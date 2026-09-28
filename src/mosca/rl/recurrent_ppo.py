"""PPO com ator recorrente (o controlador de conectoma).

A coleta roda tentativas completas com uma única versão da política, como em ppo.py, e grava
o estado da rede a cada `chunk` passos. Na atualização, cada trecho de `chunk` passos
recomeça do estado gravado e o gradiente atravessa o trecho (retropropagação truncada no
tempo). O crítico é uma MLP sem memória que vê a observação e o estado privilegiado.

O passo é controlado como em ppo.py: taxa fixa dentro da iteração, épocas interrompidas se
a divergência KL passa de `kl_stop` × alvo, e ajuste da taxa uma vez por iteração. Aqui o
ajuste vale só para o ator (cada grupo com o seu multiplicador "mult", porque a saída é
muito mais sensível a uns parâmetros que a outros): o crítico tem taxa própria (grupo
"critic" do otimizador) e o
corte de gradiente é separado, para o gradiente grande do crítico no começo não encolher o
do ator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from mosca.rl.ppo import _mlp


@dataclass(frozen=True)
class RecurrentPPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 3
    chunk: int = 16  # passos por trecho da retropropagação
    minibatch_chunks: int = 32
    lr: float = 3e-4
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    target_kl: float = 0.02
    kl_stop: float = 2.0
    # Média móvel da KL dos minilotes usada para parar a época e ajustar a taxa (1 = sem média). A KL
    # de um minilote de 16 trechos salta: no PPO de patins, picos de 0,1–0,2 com média de 0,002–0,009
    # paravam a época cedo e derrubavam a taxa sem a política ter mudado de fato.
    kl_smoothing: float = 1.0
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    reward_scale: float = 0.1


class Critic(nn.Module):
    def __init__(self, obs_dim: int, priv_dim: int, hidden=(512, 256)):
        super().__init__()
        self.net = _mlp(obs_dim + priv_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, priv: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, priv], dim=-1)).squeeze(-1)


@dataclass
class Rollout:
    """Uma leva de tentativas (T passos × B ambientes), já no dispositivo do ator."""

    obs: torch.Tensor  # (T, B, D) observação crua, entrada do ator
    obs_norm: torch.Tensor  # (T, B, D) normalizada, entrada do crítico
    priv_norm: torch.Tensor  # (T, B, P)
    act: torch.Tensor  # (T, B, A)
    logp: torch.Tensor  # (T, B)
    mean: torch.Tensor  # (T, B, A) ação média na coleta
    valid: torch.Tensor  # (T, B) bool
    v_cmd: torch.Tensor  # (B,)
    states: torch.Tensor  # (C, N, B) estado da rede no começo de cada trecho
    adv: torch.Tensor  # (T, B)
    ret: torch.Tensor  # (T, B)
    log_std: torch.Tensor  # (A,) da coleta


def chunks_with_data(valid: torch.Tensor, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(trecho, ambiente) de todos os trechos com pelo menos um passo executado."""
    steps, n = valid.shape
    starts = torch.arange(0, steps, chunk, device=valid.device)
    has = valid[starts]  # um trecho tem dados se o seu primeiro passo foi executado
    k, b = torch.nonzero(has, as_tuple=True)
    return k, b


def recurrent_ppo_update(policy: nn.Module, critic: Critic, opt: torch.optim.Optimizer, ro: Rollout,
                         cfg: RecurrentPPOConfig, lr: float, generator: torch.Generator,
                         extra_loss=None, critic_only: bool = False) -> tuple[float, dict]:
    """`extra_loss`: função sem argumentos somada à perda de cada minilote (por exemplo, uma âncora nos
    parâmetros de partida). `critic_only`: só o crítico aprende (aquecimento), sem re-executar o ator."""
    steps = ro.valid.shape[0]
    valid_adv = ro.adv[ro.valid]
    adv_all = (ro.adv - valid_adv.mean()) / (valid_adv.std() + 1e-8)
    ks, bs = chunks_with_data(ro.valid, cfg.chunk)
    offsets = torch.arange(cfg.chunk, device=ro.valid.device)
    old_std = ro.log_std.exp()
    for group in opt.param_groups:
        if group.get("name") != "critic":
            group["lr"] = lr * group.get("mult", 1.0)
    actor_params = list(policy.parameters())
    critic_params = list(critic.parameters())
    stats = {"policy_loss": [], "value_loss": [], "kl": [], "clip_frac": [], "grad_actor": []}
    kl, kl_raw, epochs_done, stopped = 0.0, 0.0, 0, False
    smoothing_started = False
    for _ in range(cfg.epochs):
        perm = torch.randperm(len(ks), generator=generator).to(ks.device)
        for start in range(0, len(ks), cfg.minibatch_chunks):
            idx = perm[start : start + cfg.minibatch_chunks]
            k, b = ks[idx], bs[idx]
            t_abs = k[None, :] * cfg.chunk + offsets[:, None]  # (L, M)
            inside = t_abs < steps
            t_c = t_abs.clamp(max=steps - 1)
            mask = inside & ro.valid[t_c, b[None, :]]
            if critic_only:
                count = mask.sum().clamp(min=1)
                value = critic(ro.obs_norm[t_c, b[None, :]], ro.priv_norm[t_c, b[None, :]])
                value_loss = ((value - ro.ret[t_c, b[None, :]]).pow(2) * mask).sum() / count
                opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
                opt.step()
                stats["value_loss"].append(value_loss.item())
                continue
            r = ro.states[k, :, b].T.contiguous()  # (N, M)
            means = []
            for t in range(cfg.chunk):
                r, m = policy(r, ro.obs[t_c[t], b], ro.v_cmd[b])
                means.append(m)
            mean = torch.stack(means)  # (L, M, A)
            dist = Normal(mean, policy.log_std.exp().expand_as(mean))
            act = ro.act[t_c, b[None, :]]
            logp = dist.log_prob(act).sum(-1)
            ratio = (logp - ro.logp[t_c, b[None, :]]).exp()
            adv = adv_all[t_c, b[None, :]]
            surrogate = torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv)
            count = mask.sum().clamp(min=1)
            policy_loss = -(surrogate * mask).sum() / count
            value = critic(ro.obs_norm[t_c, b[None, :]], ro.priv_norm[t_c, b[None, :]])
            value_loss = ((value - ro.ret[t_c, b[None, :]]).pow(2) * mask).sum() / count
            loss = policy_loss + cfg.vf_coef * value_loss
            if extra_loss is not None:
                loss = loss + extra_loss()
            opt.zero_grad()
            loss.backward()
            stats["grad_actor"].append(float(nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)))
            nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():  # KL da coleta até os parâmetros de antes deste passo
                new_std = policy.log_std.exp()
                old_mean = ro.mean[t_c, b[None, :]]
                per = (torch.log(new_std / old_std) + (old_std**2 + (old_mean - mean) ** 2) / (2 * new_std**2) - 0.5).sum(-1)
                kl_raw = ((per * mask).sum() / count).item()
            kl = kl_raw if not smoothing_started or cfg.kl_smoothing >= 1.0 else kl + cfg.kl_smoothing * (kl_raw - kl)
            smoothing_started = True
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["kl"].append(kl_raw)
            stats["clip_frac"].append(((((ratio - 1).abs() > cfg.clip) & mask).sum() / count).item())
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
    out = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
    out.update(kl_final=kl, kl_last=kl_raw, epochs=epochs_done, minibatches=len(stats["kl"]))
    return lr, out
