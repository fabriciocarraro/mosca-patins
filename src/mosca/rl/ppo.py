"""PPO com tentativas completas por versão da política.

Cada iteração recomeça todos os ambientes, roda até todos caírem ou chegarem ao fim do
tempo e só então atualiza a política: cada tentativa inteira sai de uma única versão da
política, requisito da captura fiel. O crítico é assimétrico (vê também o estado
privilegiado do simulador).

Controle do tamanho do passo: a taxa de aprendizado fica fixa dentro da iteração; as
épocas param antes se a divergência KL entre a política da coleta e a atual passa de
`kl_stop` × alvo; e a taxa se ajusta uma vez por iteração pela divergência final. (Ajustar
a cada minilote, com ~35 minilotes por iteração, fazia a taxa subir até o teto, estourar a
divergência e despencar ao mínimo dentro de cada iteração.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatch: int = 4096
    lr: float = 3e-4
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    target_kl: float = 0.01
    kl_stop: float = 2.0  # para as épocas quando a divergência passa de kl_stop × alvo
    ent_coef: float = 0.0
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    reward_scale: float = 0.1


class RunningNorm:
    """Normalização das entradas por média e variância acumuladas (algoritmo paralelo de Chan)."""

    def __init__(self, dim: int, clip: float = 5.0):
        self.mean, self.var, self.count, self.clip = np.zeros(dim), np.ones(dim), 1e-4, clip

    def update(self, x: np.ndarray) -> None:
        b_mean, b_var, b_count = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta, total = b_mean - self.mean, self.count + b_count
        self.mean = self.mean + delta * b_count / total
        self.var = (self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / total) / total
        self.count = total

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


def _mlp(inp: int, hidden: tuple[int, ...], out: int) -> nn.Sequential:
    layers, last = [], inp
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ELU()]
        last = h
    layers.append(nn.Linear(last, out))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, priv_dim: int, act_dim: int, actor_hidden=(256, 256),
                 critic_hidden=(512, 256), init_std: float = 0.4):
        super().__init__()
        self.actor = _mlp(obs_dim, actor_hidden, act_dim)
        self.critic = _mlp(obs_dim + priv_dim, critic_hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), math.log(init_std)))
        with torch.no_grad():  # começa perto da postura canônica (ação zero)
            self.actor[-1].weight.mul_(0.01)
            self.actor[-1].bias.zero_()

    def dist(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs: torch.Tensor, priv: torch.Tensor) -> torch.Tensor:
        return self.critic(torch.cat([obs, priv], dim=-1)).squeeze(-1)


def compute_gae(rew: np.ndarray, val: np.ndarray, valid: np.ndarray, terminal: np.ndarray,
                last_val: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Vantagens e retornos para episódios síncronos (todos começam em t=0).

    `valid[t, i]` marca os passos executados pelo ambiente i. No último passo de cada
    episódio, o valor seguinte é 0 se ele terminou em queda, ou `last_val` (valor da última
    observação) se acabou o tempo.
    """
    steps, n = rew.shape
    lengths = valid.sum(axis=0)
    adv = np.zeros((steps, n))
    running = np.zeros(n)
    for t in reversed(range(steps)):
        is_last = t == lengths - 1
        following = val[t + 1] if t + 1 < steps else np.zeros(n)
        next_val = np.where(is_last, np.where(terminal, 0.0, last_val), following)
        delta = rew[t] + gamma * next_val - val[t]
        running = delta + gamma * lam * np.where(is_last, 0.0, running)
        running = np.where(valid[t], running, 0.0)
        adv[t] = running
    return adv, adv + val


def ppo_update(ac: ActorCritic, opt: torch.optim.Optimizer, batch: dict, cfg: PPOConfig, lr: float,
               generator: torch.Generator) -> tuple[float, dict]:
    obs, priv, act = batch["obs"], batch["priv"], batch["act"]
    old_logp, old_mean, old_log_std = batch["logp"], batch["mean"], batch["log_std"]
    ret = batch["ret"]
    adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
    n = obs.shape[0]
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "kl": [], "clip_frac": []}
    old_std = old_log_std.exp()
    for group in opt.param_groups:
        group["lr"] = lr
    kl, epochs_done, stopped = 0.0, 0, False
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, cfg.minibatch):
            idx = perm[start : start + cfg.minibatch]
            dist = ac.dist(obs[idx])
            logp = dist.log_prob(act[idx]).sum(-1)
            ratio = (logp - old_logp[idx]).exp()
            surrogate = torch.min(ratio * adv[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv[idx])
            value = ac.value(obs[idx], priv[idx])
            policy_loss = -surrogate.mean()
            value_loss = (value - ret[idx]).pow(2).mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                new_std = ac.log_std.exp()
                kl = (torch.log(new_std / old_std) + (old_std**2 + (old_mean[idx] - ac.actor(obs[idx])) ** 2)
                      / (2 * new_std**2) - 0.5).sum(-1).mean().item()
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.item())
            stats["kl"].append(kl)
            stats["clip_frac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
            if kl > cfg.kl_stop * cfg.target_kl:
                stopped = True
                break
        epochs_done += 1
        if stopped:
            break
    if stopped or kl > 2 * cfg.target_kl:
        lr = max(cfg.lr_min, lr / 1.5)
    elif kl < cfg.target_kl / 2:
        lr = min(cfg.lr_max, lr * 1.5)
    out = {k: float(np.mean(v)) for k, v in stats.items()}
    out.update(kl_final=kl, epochs=epochs_done)
    return lr, out
