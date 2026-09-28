"""Rede de taxa de Pugliese et al. em torch, em lote, para o controlador (CPU ou GPU).

    τ_i·dr_i/dt = max(r_max,i · tanh((a_i / r_max,i) · (I_i + b·Σ_j w_ij r_j − θ_i)), 0) − r_i

Mesma equação e parâmetros de `mosca.brain.pugliese` (a referência em numpy, validada contra
o código deles), com integração RK4. Com passo de 5 ms, o ritmo do DNg100 no grafo de 6
patas fica igual ao de 1 ms (correlação 0,9995), então um passo de controle de 10 ms
precisa de só 2 subpassos.

Treináveis por tipo celular, como no plano: fatores multiplicativos em a, θ e τ (em log,
começando em 0, isto é, nos valores sorteados da receita deles). Neurônios do mesmo tipo
dos dois lados compartilham os fatores, o que dá simetria esquerda/direita de graça.

O estado fica como (neurônios × lote), contíguo: é o formato que a multiplicação esparsa
CSR usa direto. Com o lote na primeira dimensão, as transpostas a cada multiplicação a
deixavam 7 a 10 vezes mais lenta na GPU.

Degrau 2 da escada de flexibilidade: um fator positivo treinável por ligação (em log,
começando em 1); o sinal e a existência de cada ligação continuam os do conectoma. Com
`edge_gains=True` vale para todas as ligações; com `gain_rows`, só para as que chegam aos
neurônios dessas linhas (no controlador, os motores das patas), somadas à parte.

Gradiente substituto (`surrogate=True`): a simulação é exatamente a equação acima, mas no
cálculo do gradiente um neurônio abaixo do limiar ganha uma inclinação que decai com a
distância até o limiar (largura θ/2 do próprio neurônio). Sem isso, neurônio calado não
passa gradiente, e uma rede que começa calada não aprende nada.

Modo exato (`exact=True`, em float64): antes de cada multiplicação esparsa, as taxas vão para
múltiplos de 2^-12 Hz e os pesos para múltiplos de 2^-18. Cada produto é então um múltiplo de 2^-30
e, enquanto a soma dos |produtos| de uma linha ficar abaixo de 2^23 (ver `exact_margin`), toda soma
parcial é exata em float64: o resultado não depende da ordem das somas (na GPU, a multiplicação
esparsa soma em ordem variável e dá resultados diferentes a cada execução), nem do tamanho do lote,
nem do algoritmo. O gradiente passa direto pelo arredondamento.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from mosca.brain.pugliese import B, NeuronParams

W_BITS, R_BITS = 18, 12  # grades do modo exato: pesos em múltiplos de 2^-18, taxas em 2^-12 Hz


def _on_grid(x: torch.Tensor, bits: int) -> torch.Tensor:
    """x arredondado para múltiplos de 2^-bits (exato: só escalas por potências de 2); o gradiente passa direto."""
    q = torch.round(x * 2.0**bits) * 2.0**-bits
    return q.detach() + (x - x.detach()) if x.requires_grad else q


def sparse_tensor(w: sp.csr_matrix, device, dtype) -> torch.Tensor:
    w = w.tocsr()
    with warnings.catch_warnings():  # "suporte a CSR em beta": usamos só sparse.mm, que é estável
        warnings.simplefilter("ignore", UserWarning)
        return torch.sparse_csr_tensor(torch.from_numpy(w.indptr.astype(np.int64)), torch.from_numpy(w.indices.astype(np.int64)),
                                       torch.from_numpy(w.data), size=w.shape, device=device, dtype=dtype,
                                       check_invariants=True)


class _SpMM(torch.autograd.Function):
    """W @ x com W fixa; a volta usa a transposta já pronta (o torch montaria W^T a cada chamada)."""

    @staticmethod
    def forward(ctx, w, wt, x):
        ctx.wt = wt
        return torch.sparse.mm(w, x)

    @staticmethod
    def backward(ctx, grad):
        return None, None, torch.sparse.mm(ctx.wt, grad)


class _Activation(torch.autograd.Function):
    """max(r_max·tanh((a/r_max)·x), 0), com derivada substituta a·exp(x/largura) para x ≤ 0."""

    @staticmethod
    def forward(ctx, x, a, r_max, width):
        t = torch.tanh((a / r_max) * x)
        ctx.save_for_backward(x, a, t, width)
        return torch.relu(r_max * t)

    @staticmethod
    def backward(ctx, grad):
        x, a, t, width = ctx.saved_tensors
        above = x > 0
        slope = 1.0 - t * t
        dx = torch.where(above, a * slope, a * torch.exp(torch.clamp(x, max=0.0) / width))
        da = torch.where(above, x * slope, torch.zeros_like(x))
        return grad * dx, (grad * da).sum(dim=1, keepdim=True), None, None


class _GainedSpMM(torch.autograd.Function):
    """W(v) @ x com v = base · exp(ganho) por ligação; a volta dá o gradiente de x (pela transposta)
    e o de cada ganho: o produto grad @ x^T amostrado nas ligações (SDDMM do cuSPARSE, 15 vezes mais
    rápido que coletar as duas matrizes por ligação)."""

    @staticmethod
    def forward(ctx, log_gain, net, x):
        values = net.edge_base * log_gain.exp()
        if net.exact:
            values = _on_grid(values, W_BITS)
        w = torch.sparse_csr_tensor(net.edge_crow, net.edge_col, values, size=net.edge_shape, check_invariants=False)
        ctx.net = net
        ctx.save_for_backward(values, x)
        return torch.sparse.mm(w, x)

    @staticmethod
    def backward(ctx, grad):
        values, x = ctx.saved_tensors
        net = ctx.net
        wt = torch.sparse_csr_tensor(net.edge_crow_t, net.edge_col_t, values[net.edge_perm_t],
                                     size=(net.edge_shape[1], net.edge_shape[0]), check_invariants=False)
        grad_x = torch.sparse.mm(wt, grad)
        pattern = torch.sparse_csr_tensor(net.edge_crow, net.edge_col, torch.zeros_like(values), size=net.edge_shape,
                                          check_invariants=False)
        grad_v = torch.sparse.sampled_addmm(pattern, grad, x.T, beta=0.0).values()
        return grad_v * values, None, grad_x


class PuglieseNet(nn.Module):
    def __init__(self, w: sp.csr_matrix, params: NeuronParams, cell_type: np.ndarray | None = None, b: float = B,
                 device: str | torch.device = "cpu", dtype=torch.float32, surrogate: bool = False,
                 gain_rows: np.ndarray | None = None, edge_gains: bool = False, exact: bool = False):
        """`w`: pós × pré com sinal × nº de sinapses (sem o b); `cell_type`: rótulo de cada neurônio
        para compartilhar os fatores treináveis (None = um fator por neurônio); `gain_rows`:
        neurônios cujas ligações de entrada ganham um fator treinável por ligação; `exact`: somas
        exatas na multiplicação esparsa (ver a documentação do módulo; exige float64)."""
        super().__init__()
        if exact and dtype != torch.float64:
            raise ValueError("o modo exato precisa de float64")
        self.exact = exact
        w = (b * w).tocsr()
        w.sort_indices()
        self.edge_gains = edge_gains
        if edge_gains:  # ganho em todas as ligações
            n_edges = w.nnz
            ids = sp.csr_matrix((np.arange(n_edges, dtype=np.float64) + 1, w.indices, w.indptr), shape=w.shape).T.tocsr()
            ids.sort_indices()

            def lt(x):
                return torch.as_tensor(np.asarray(x, dtype=np.int64), device=device)

            self.register_buffer("edge_crow", lt(w.indptr), persistent=False)
            self.register_buffer("edge_col", lt(w.indices), persistent=False)
            self.register_buffer("edge_row", lt(np.repeat(np.arange(w.shape[0]), np.diff(w.indptr))), persistent=False)
            self.register_buffer("edge_base", torch.as_tensor(w.data, dtype=dtype, device=device), persistent=False)
            self.register_buffer("edge_crow_t", lt(ids.indptr), persistent=False)
            self.register_buffer("edge_col_t", lt(ids.indices), persistent=False)
            self.register_buffer("edge_perm_t", lt(ids.data.astype(np.int64) - 1), persistent=False)
            self.edge_shape = w.shape
            self.log_edge_gain = nn.Parameter(torch.zeros(n_edges, device=device, dtype=dtype))
            gain_rows = None
        self.has_gains = gain_rows is not None and len(gain_rows) > 0
        if self.has_gains:
            rows = np.zeros(w.shape[0], dtype=bool)
            rows[np.asarray(gain_rows)] = True
            coo = w.tocoo()
            sel = rows[coo.row]
            self.register_buffer("gain_post", torch.as_tensor(coo.row[sel], dtype=torch.long, device=device), persistent=False)
            self.register_buffer("gain_pre", torch.as_tensor(coo.col[sel], dtype=torch.long, device=device), persistent=False)
            self.register_buffer("gain_val", torch.as_tensor(coo.data[sel], dtype=dtype, device=device), persistent=False)
            self.log_gain = nn.Parameter(torch.zeros(int(sel.sum()), device=device, dtype=dtype))
            w = sp.csr_matrix((coo.data[~sel], (coo.row[~sel], coo.col[~sel])), shape=w.shape)
        if exact:
            w.data = np.round(w.data * 2.0**W_BITS) / 2.0**W_BITS
        # Soma dos |pesos| fixos (sem ganho treinável) por linha, para exact_margin.
        fixed_abs = np.zeros(w.shape[0]) if edge_gains else np.asarray(abs(w).sum(axis=1)).ravel()
        self.register_buffer("fixed_abs", torch.as_tensor(fixed_abs, dtype=dtype, device=device), persistent=False)
        # Reconstruídas a partir do grafo; fora do state_dict para os checkpoints ficarem pequenos.
        self.register_buffer("w", sparse_tensor(w, device, dtype), persistent=False)
        self.register_buffer("wt", sparse_tensor(w.T.tocsr(), device, dtype), persistent=False)
        self.surrogate = surrogate

        def buf(x):  # parâmetros por neurônio como coluna (N, 1), para somar ao estado (N, lote)
            return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)[:, None]

        self.register_buffer("tau0", buf(params.tau))
        self.register_buffer("a0", buf(params.a))
        self.register_buffer("theta0", buf(params.theta))
        self.register_buffer("r_max", buf(params.r_max))
        n = len(params.tau)
        if cell_type is None:
            groups = np.arange(n)
        else:
            _, groups = np.unique(np.asarray(cell_type, dtype=str), return_inverse=True)
        self.register_buffer("group", torch.as_tensor(groups, device=device, dtype=torch.long))
        k = int(groups.max()) + 1
        self.log_a = nn.Parameter(torch.zeros(k, device=device, dtype=dtype))
        self.log_theta = nn.Parameter(torch.zeros(k, device=device, dtype=dtype))
        self.log_tau = nn.Parameter(torch.zeros(k, device=device, dtype=dtype))

    @property
    def n(self) -> int:
        return self.w.shape[0]

    def neuron_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self.group
        return (self.a0 * self.log_a[g, None].exp(), self.theta0 * self.log_theta[g, None].exp(),
                self.tau0 * self.log_tau[g, None].exp())

    def exact_margin(self, r_bound: float = 1200.0) -> float:
        """No modo exato, quantas vezes o limite 2^23 é maior que a maior soma possível dos |produtos|
        numa linha (Σ|w|·r_bound; as taxas ficam em [0, 1000], com folga para os passos do RK4).
        Acima de 1, toda soma é exata."""
        with torch.no_grad():
            rows = self.fixed_abs.clone()
            if self.edge_gains:
                rows = rows.index_add(0, self.edge_row, (self.edge_base * self.log_edge_gain.exp()).abs())
            if self.has_gains:
                rows = rows.index_add(0, self.gain_post, (self.gain_val * self.log_gain.exp()).abs())
        return float(2.0 ** (53 - W_BITS - R_BITS) / (rows.max().item() * r_bound))

    def deriv(self, r: torch.Tensor, current: torch.Tensor, a, theta, tau) -> torch.Tensor:
        """dr/dt para o estado r (N, lote) com corrente externa (N, lote)."""
        x = _on_grid(r, R_BITS) if self.exact else r
        if self.edge_gains:
            syn = _GainedSpMM.apply(self.log_edge_gain, self, x)
        else:
            syn = _SpMM.apply(self.w, self.wt, x)
        if self.has_gains:  # somado antes da corrente: no modo exato, a soma continua exata
            weight = self.gain_val * self.log_gain.exp()
            if self.exact:
                weight = _on_grid(weight, W_BITS)
            syn = syn.index_add(0, self.gain_post, weight[:, None] * x[self.gain_pre])
        drive = syn + current - theta
        if self.surrogate:
            rate = _Activation.apply(drive, a, self.r_max, 0.5 * self.theta0)
        else:
            rate = torch.relu(self.r_max * torch.tanh((a / self.r_max) * drive))
        return (rate - r) / tau

    def forward(self, r: torch.Tensor, current: torch.Tensor, dt: float, substeps: int = 1) -> torch.Tensor:
        """Avança `dt` segundos em `substeps` passos de RK4, com a corrente constante no intervalo.
        `r` e `current` têm forma (N, lote)."""
        a, theta, tau = self.neuron_params()
        h = dt / substeps
        for _ in range(substeps):
            k1 = self.deriv(r, current, a, theta, tau)
            k2 = self.deriv(r + 0.5 * h * k1, current, a, theta, tau)
            k3 = self.deriv(r + 0.5 * h * k2, current, a, theta, tau)
            k4 = self.deriv(r + h * k3, current, a, theta, tau)
            r = r + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            # Corte em [0, 1000] como na referência; o gradiente passa direto (o clamp do torch
            # zera o gradiente no limite, justamente onde fica um neurônio calado).
            r = r + (r.clamp(0.0, 1000.0) - r).detach()
        return r
