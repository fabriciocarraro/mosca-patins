"""Aluno MLP da destilação do M4, para controle: as mesmas entradas do conectoma, sem a fiação.

Recebe o mesmo que o codificador do conectoma (ângulo e velocidade das 7 juntas e a carga de
cada pata, nas mesmas escalas fixas) e os comandos de velocidade e giro, e devolve as 48
saídas do corpo "walk". Separa os problemas: se a MLP aprende a andar com a mesma receita, o
que falta é da rede do conectoma; se não aprende, é da receita. Opcionalmente atrasa as
entradas e filtra a saída, para imitar a lentidão da rede de taxa (neurônios com τ ~20 ms).

Mesma interface do ConnectomePolicy na destilação: estado (n, lote), `forward(r, obs, v_cmd)`
devolve o novo estado e a ação.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from mosca.brain.controller import FEATURE_CENTER, FEATURE_SCALE, leg_feature_index

N_FEATURES = 6 * len(FEATURE_CENTER) + 2  # sentidos das 6 patas + velocidade e giro pedidos


@dataclass(frozen=True)
class MLPStudentConfig:
    hidden: int = 256
    layers: int = 2
    lag_steps: int = 0  # atraso das entradas, em passos de controle
    tau: float = 0.0  # constante de tempo (s) de um filtro na saída; 0 = sem filtro
    control_dt: float = 0.002
    n_out: int = 48
    seed: int = 0


def features(obs: np.ndarray, v_cmd: np.ndarray) -> np.ndarray:
    """As entradas do aluno (lote, N_FEATURES) a partir da observação (lote, 186): as mesmas do
    codificador do conectoma, nas mesmas escalas, e os comandos de velocidade e giro."""
    idx = np.concatenate([leg_feature_index(k) for k in range(6)])
    x = (obs[:, idx] - np.tile(FEATURE_CENTER, 6)) / np.tile(FEATURE_SCALE, 6)
    return np.concatenate([x, np.asarray(v_cmd)[:, None], obs[:, 184:185]], axis=1)


class OutputLatency:
    """Atraso (em passos de controle) e filtro de primeira ordem nas saídas de uma política, por
    ambiente: y(t) = filtro(u(t − atraso)). Imita a lentidão de uma rede de neurônios de taxa."""

    def __init__(self, n_envs: int, n_out: int, delay_steps: int, tau: float, dt: float = 0.002):
        self.delay = delay_steps
        self.alpha = dt / tau if tau > 0 else 1.0
        self.queue = np.zeros((max(delay_steps, 1), n_envs, n_out))
        self.y = np.zeros((n_envs, n_out))
        self.head = 0

    def reset(self, u0: np.ndarray) -> None:
        self.queue[:] = u0
        self.y[:] = u0
        self.head = 0

    def __call__(self, u: np.ndarray) -> np.ndarray:
        if self.delay > 0:
            delayed = self.queue[self.head].copy()
            self.queue[self.head] = u
            self.head = (self.head + 1) % self.delay
        else:
            delayed = u
        self.y = self.y + self.alpha * (delayed - self.y)
        return self.y.copy()

    def state(self) -> np.ndarray:
        """Comandos ainda na fila (do mais antigo ao mais novo) e o estado do filtro: (lote, …)."""
        order = [(self.head + k) % len(self.queue) for k in range(len(self.queue))] if self.delay > 0 else []
        parts = [self.queue[k] for k in order] + [self.y]
        return np.concatenate(parts, axis=1)


class MLPStudent(nn.Module):
    def __init__(self, cfg: MLPStudentConfig = MLPStudentConfig(), device="cpu"):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.n_out = cfg.n_out
        feats = np.concatenate([leg_feature_index(k) for k in range(6)])
        self.register_buffer("feat_idx", torch.as_tensor(feats, dtype=torch.long, device=device))
        self.register_buffer("center", torch.as_tensor(np.tile(FEATURE_CENTER, 6), dtype=torch.float32, device=device))
        self.register_buffer("scale", torch.as_tensor(np.tile(FEATURE_SCALE, 6), dtype=torch.float32, device=device))
        layers, d = [], N_FEATURES
        for _ in range(cfg.layers):
            layers += [nn.Linear(d, cfg.hidden), nn.ELU()]
            d = cfg.hidden
        last = nn.Linear(d, cfg.n_out)
        with torch.no_grad():
            last.weight.mul_(0.1)
            last.bias.zero_()
        self.mlp = nn.Sequential(*layers, last).to(device)
        self.dec_bias = nn.Parameter(torch.zeros(cfg.n_out, device=device))
        self.alpha = cfg.control_dt / cfg.tau if cfg.tau > 0 else 1.0
        self.n = cfg.lag_steps * N_FEATURES + (cfg.n_out if cfg.tau > 0 else 0)

    @property
    def device(self) -> torch.device:
        return self.dec_bias.device

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"mlp": list(self.mlp.parameters()), "decodificador": [self.dec_bias]}

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(self.n, batch, device=self.device)

    def features(self, obs: torch.Tensor, v_cmd: torch.Tensor) -> torch.Tensor:
        x = (obs[:, self.feat_idx] - self.center) / self.scale
        return torch.cat([x, v_cmd[:, None], obs[:, 184:185]], dim=1)  # obs[:, 184]: giro pedido

    def forward(self, r: torch.Tensor, obs: torch.Tensor, v_cmd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(obs, v_cmd)
        batch, lag = x.shape[0], self.cfg.lag_steps
        state = []
        if lag > 0:  # fila das entradas: a mais nova primeiro; a rede vê a de `lag` passos atrás
            queue = r[: lag * N_FEATURES].T.reshape(batch, lag, N_FEATURES)
            x_in = queue[:, -1]
            state.append(torch.cat([x[:, None], queue[:, :-1]], dim=1).reshape(batch, -1))
        else:
            x_in = x
        y = self.mlp(x_in) + self.dec_bias
        if self.cfg.tau > 0:
            prev = r[lag * N_FEATURES :].T
            y = prev + self.alpha * (y - prev)
            state.append(y)
        r_new = torch.cat(state, dim=1).T if state else r
        return r_new, y
