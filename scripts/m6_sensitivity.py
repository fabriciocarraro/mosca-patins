"""M6: sensibilidade da ação do conectoma a cada grupo de parâmetros, para as taxas do PPO de patins.

No primeiro teste de patinação partindo do conectoma que anda (patina_a), a divergência KL passou
do alvo já na primeira época em toda iteração e a taxa caiu ao piso. No Adam, cada parâmetro anda
~lr no primeiro passo, então um grupo com um milhão de parâmetros (os ganhos por ligação) mexe na
ação muito mais que um grupo de poucos. Este script mede, para cada grupo, a divergência KL média
entre a política antes e depois de um passo de tamanho `--lr` na direção do sinal do gradiente de
um objetivo do PPO com vantagens sorteadas (o primeiro passo do Adam), e sugere multiplicadores que
igualam a contribuição dos grupos.

Uso:
    python scripts/m6_sensitivity.py --init-from runs/anda_r5F/best.pt --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from connectome_train import load_walking  # noqa: E402
from mosca.brain.controller import ConnectomePolicy, ControllerConfig  # noqa: E402
from mosca.brain.graph import Connectome  # noqa: E402
from mosca.env.skate_env import EnvConfig, SkateVecEnv  # noqa: E402
from mosca.paths import MALECNS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-from", required=True, help="checkpoint do conectoma que anda")
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=100, help="passos de controle (10 ms) coletados")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--init-std", type=float, default=0.6)
    p.add_argument("--substeps", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    src = torch.load(args.init_from, weights_only=False, map_location="cpu")
    wc = dict(src["controller"])
    env = SkateVecEnv(EnvConfig(n_envs=args.envs, n_threads=args.threads))
    cfg = ControllerConfig(size_ref=wc["size_ref"], walk_gain=wc["walk_gain"], turn_gain=wc["turn_gain"],
                           enc_std=wc["enc_std"], prop_offset=wc["prop_offset"], motor_tone=wc["motor_tone"],
                           dec_gain=wc["dec_gain"], init_std=args.init_std, control_dt=env.cfg.control_dt,
                           substeps=args.substeps, tau_scale=wc.get("tau_scale", 1.0),
                           all_synapse_gains=wc.get("all_synapse_gains", False),
                           haltere_input=wc.get("haltere_input", False), haltere_scale=wc.get("haltere_scale", 2.0),
                           haltere_offset=wc.get("haltere_offset"), turn_cells=tuple(wc.get("turn_cells", ("DNa01", "DNa02"))))
    graph = Path(src["args"].get("graph", "")).name or "controller_graph_min5.npz"
    pol = ConnectomePolicy(Connectome.load(MALECNS_DIR / graph), cfg, device=device)
    load_walking(pol, args.init_from, device)
    obs, _ = env.reset(np.arange(env.n) + 10**8, np.full(env.n, 1.0))
    pol.calibrate_rest(torch.as_tensor(obs, dtype=torch.float32, device=device), 0.65)

    # Coleta com a ação média: estados da rede e observações de cada passo.
    obs, _ = env.reset(np.arange(env.n) + 2 * 10**8, np.full(env.n, 1.0))
    r = pol.initial_state(env.n)
    vc = torch.full((env.n,), 1.0, device=device)
    states, observations = [], []
    with torch.no_grad():
        for _ in range(args.steps):
            o = torch.as_tensor(obs, dtype=torch.float32, device=device)
            states.append(r.clone())
            observations.append(o)
            r, mean = pol(r, o, vc)
            obs, *_ = env.step(mean.cpu().numpy())
    S, O = torch.stack(states), torch.stack(observations)  # (T, N, B), (T, B, D)
    T = len(S)

    def means() -> torch.Tensor:
        out = []
        for t in range(T):
            _, m = pol(S[t], O[t], vc)
            out.append(m)
        return torch.stack(out)  # (T, B, A)

    gen = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        mu0 = means()
    std = pol.log_std.exp().detach()
    act = mu0 + std * torch.randn(mu0.shape, generator=gen).to(device)
    adv = torch.randn(mu0.shape[:2], generator=gen).to(device)
    pol.zero_grad()
    loss = -(adv * Normal(means(), std).log_prob(act).sum(-1)).mean()
    loss.backward()

    groups = pol.param_groups()
    print(f"passo de sinal com lr {args.lr:g} (o primeiro passo do Adam), KL média por passo de controle:")
    results = {}
    for name, params in groups.items():
        if name == "exploracao":
            continue
        backup = [p.detach().clone() for p in params]
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.sub_(args.lr * p.grad.sign())
            mu1 = means()
            for p, b in zip(params, backup):
                p.copy_(b)
        kl = float((((mu1 - mu0) ** 2) / (2 * std**2)).sum(-1).mean())
        results[name] = kl
        print(f"   {name:14s} {sum(p.numel() for p in params):>10,} parâmetros  KL {kl:.2e}")
    target = 0.02 / max(len(results), 1)  # alvo do PPO dividido igualmente entre os grupos
    print(f"multiplicadores que dão ~{target:.1e} de KL por grupo num passo com lr {args.lr:g}:")
    for name, kl in results.items():
        mult = float(np.sqrt(target / kl)) if kl > 0 else float("inf")
        print(f"   {name:14s} x{mult:.3g}")


if __name__ == "__main__":
    main()
