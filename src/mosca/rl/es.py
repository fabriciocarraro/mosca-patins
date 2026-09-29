"""Estratégias evolutivas (OpenAI-ES, Salimans et al. 2017) sobre um subconjunto dos parâmetros.

Cada geração sorteia perturbações antitéticas (+ε e −ε) de uma semente própria; cada membro da
população roda uma tentativa com os parâmetros θ + σ·ε (uma coluna do lote), e o passo estima o
gradiente do retorno pela soma das perturbações pesadas pela posição de cada membro no ranking
(notas centradas em [−0,5; 0,5]). Não passa gradiente pela rede: serve para dinâmicas caóticas, em
que o gradiente pela recorrência explode (Metz et al. 2021).

Cada grupo de parâmetros tem o seu σ (nas unidades do parâmetro) e a sua taxa, proporcional ao σ:
com o Adam, a média anda ~taxa·σ por geração em cada parâmetro.
"""

from __future__ import annotations

import numpy as np
import torch


def centered_ranks(x: np.ndarray) -> np.ndarray:
    """Notas pela posição: a pior −0,5, a melhor +0,5 (insensível à escala e a valores extremos)."""
    ranks = np.empty(len(x))
    ranks[np.argsort(x)] = np.arange(len(x))
    return ranks / max(len(x) - 1, 1) - 0.5


class PopulationES:
    def __init__(self, params: dict[str, torch.nn.Parameter], sigma: dict[str, float], lr: float, seed: int):
        self.params, self.sigma, self.seed = params, sigma, seed
        self.slices, start = {}, 0
        for name, p in params.items():
            self.slices[name] = slice(start, start + p.numel())
            start += p.numel()
        self.dim = start
        self.opt = torch.optim.Adam([{"params": [p], "lr": lr * sigma[name]} for name, p in params.items()])

    def sample(self, generation: int, pop: int) -> torch.Tensor:
        """Perturbações unitárias (pop, dim), antitéticas: a segunda metade é o negativo da primeira."""
        assert pop % 2 == 0, "população par (pares antitéticos)"
        gen = torch.Generator().manual_seed(int(self.seed) * 1_000_003 + int(generation))
        eps = torch.randn(pop // 2, self.dim, generator=gen)
        return torch.cat([eps, -eps])

    def offsets(self, eps: torch.Tensor, device) -> dict[str, torch.Tensor]:
        """Deslocamentos (tamanho do parâmetro, pop) de cada grupo, para `ConnectomePolicy.set_population`."""
        return {name: (self.sigma[name] * eps[:, sl].T).reshape(*self.params[name].shape, -1).to(device)
                for name, sl in self.slices.items()}

    def step(self, eps: torch.Tensor, fitness: np.ndarray) -> float:
        """Um passo de subida pelo retorno; devolve a norma do gradiente estimado (em unidades de σ)."""
        w = torch.as_tensor(centered_ranks(np.asarray(fitness)), dtype=torch.float32)
        grad_unit = (w @ eps) / len(w)  # (dim,)
        for name, sl in self.slices.items():
            p = self.params[name]
            p.grad = -(grad_unit[sl] / self.sigma[name]).reshape(p.shape).to(p.device, p.dtype)
        self.opt.step()
        self.opt.zero_grad()
        return float(grad_unit.norm())
