"""Rede de taxa com a fiação do conectoma (forma do modelo de Pugliese et al.).

    τ·dr/dt = −r + r_max · max(tanh(a·(b·Σ_j W_ij r_j + I_i − θ_i)), 0)

W é a matriz esparsa pós × pré com o sinal do neurônio pré-sináptico vezes o número de
sinapses (congelada). a, θ, r_max e τ são por neurônio. Integração por Euler explícito.
A rede roda em lote (B trajetórias ao mesmo tempo) na CPU ou na GPU.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mosca.brain.graph import Connectome


def signed_weights(c: Connectome, device: str | torch.device = "cpu", dtype=torch.float32) -> torch.Tensor:
    """Matriz esparsa CSR (pós × pré) com sinal × nº de sinapses; ligações de neurônios sem sinal saem."""
    sign = c.sign[c.pre].astype(np.float32)
    keep = sign != 0
    values = torch.from_numpy(sign[keep] * c.count[keep].astype(np.float32))
    index = torch.from_numpy(np.stack([c.post[keep], c.pre[keep]]).astype(np.int64))
    w = torch.sparse_coo_tensor(index, values, (c.n, c.n)).coalesce()
    return w.to_sparse_csr().to(device=device, dtype=dtype)


class RateNetwork(nn.Module):
    def __init__(self, weights: torch.Tensor, a: torch.Tensor, theta: torch.Tensor, r_max: torch.Tensor,
                 tau: torch.Tensor, b: float = 0.03):
        super().__init__()
        self.register_buffer("w", weights)
        self.a = nn.Parameter(a)
        self.theta = nn.Parameter(theta)
        self.register_buffer("r_max", r_max)
        self.tau = nn.Parameter(tau)
        self.b = b

    @property
    def n(self) -> int:
        return self.w.shape[0]

    def drive(self, r: torch.Tensor) -> torch.Tensor:
        """Σ_j W_ij r_j para um lote (B, N)."""
        return torch.sparse.mm(self.w, r.T).T

    def step(self, r: torch.Tensor, current: torch.Tensor, dt: float) -> torch.Tensor:
        target = self.r_max * torch.relu(torch.tanh(self.a * (self.b * self.drive(r) + current - self.theta)))
        return r + (dt / self.tau) * (target - r)
