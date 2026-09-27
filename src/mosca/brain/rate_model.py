"""Rede de taxa de Pugliese et al. em torch, em lote, para o controlador (CPU ou GPU).

    τ_i·dr_i/dt = max(r_max,i · tanh((a_i / r_max,i) · (I_i + b·Σ_j w_ij r_j − θ_i)), 0) − r_i

Mesma equação e parâmetros de `mosca.brain.pugliese` (a referência em numpy, validada contra
o código deles), com integração RK4. Com passo de 5 ms, o ritmo do DNg100 no grafo de 6
patas fica igual ao de 1 ms (correlação 0,9995), então um passo de controle de 10 ms
precisa de só 2 subpassos.

Treináveis por tipo celular, como no plano: fatores multiplicativos em a, θ e τ (em log,
começando em 0, isto é, nos valores sorteados da receita deles). Neurônios do mesmo tipo
dos dois lados compartilham os fatores, o que dá simetria esquerda/direita de graça.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from mosca.brain.pugliese import B, NeuronParams


def sparse_tensor(w: sp.csr_matrix, device, dtype) -> torch.Tensor:
    w = w.tocsr()
    with warnings.catch_warnings():  # "suporte a CSR em beta": usamos só sparse.mm, que é estável
        warnings.simplefilter("ignore", UserWarning)
        return torch.sparse_csr_tensor(torch.from_numpy(w.indptr.astype(np.int64)), torch.from_numpy(w.indices.astype(np.int64)),
                                       torch.from_numpy(w.data), size=w.shape, device=device, dtype=dtype,
                                       check_invariants=True)


class PuglieseNet(nn.Module):
    def __init__(self, w: sp.csr_matrix, params: NeuronParams, cell_type: np.ndarray | None = None, b: float = B,
                 device: str | torch.device = "cpu", dtype=torch.float32):
        """`w`: pós × pré com sinal × nº de sinapses (sem o b); `cell_type`: rótulo de cada neurônio
        para compartilhar os fatores treináveis (None = um fator por neurônio)."""
        super().__init__()
        self.register_buffer("w", sparse_tensor(b * w, device, dtype))

        def buf(x):
            return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)

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
        return (self.a0 * self.log_a[g].exp(), self.theta0 * self.log_theta[g].exp(), self.tau0 * self.log_tau[g].exp())

    def deriv(self, r: torch.Tensor, current: torch.Tensor, a, theta, tau) -> torch.Tensor:
        """dr/dt para um lote r (B, N) com corrente externa (B, N)."""
        drive = torch.sparse.mm(self.w, r.T).T + current - theta
        return (torch.relu(self.r_max * torch.tanh((a / self.r_max) * drive)) - r) / tau

    def forward(self, r: torch.Tensor, current: torch.Tensor, dt: float, substeps: int = 1) -> torch.Tensor:
        """Avança `dt` segundos em `substeps` passos de RK4, com a corrente constante no intervalo."""
        a, theta, tau = self.neuron_params()
        h = dt / substeps
        for _ in range(substeps):
            k1 = self.deriv(r, current, a, theta, tau)
            k2 = self.deriv(r + 0.5 * h * k1, current, a, theta, tau)
            k3 = self.deriv(r + 0.5 * h * k2, current, a, theta, tau)
            k4 = self.deriv(r + h * k3, current, a, theta, tau)
            r = (r + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).clamp(0.0, 1000.0)
        return r
