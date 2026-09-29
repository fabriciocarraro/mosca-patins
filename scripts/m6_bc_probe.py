"""M6: o conectoma consegue imitar a MLP patinadora do M2? Teste offline (sem DAgger).

Coleta tentativas conduzidas pela MLP do M2 (ação média, sem ruído) nos patins, com velocidades
pedidas sorteadas em [--v-min, --v-max], e treina o conectoma (partindo do que anda, como o PPO
de patins) a reproduzir as ações dela, com as observações dela e retropropagação truncada. Mede
o erro normalizado (1 − R²) das ações, cru e depois do filtro dos atuadores (o que o corpo
sente), em tentativas separadas para teste. Serve para decidir se a destilação da MLP é um
caminho para o M6 (no M4, a caminhada, erro filtrado ~0,27 na imitação offline previu que o
DAgger funcionaria).

Uso (no notebook, CPU, algumas horas; ou no Spark com --device cuda):
    python scripts/m6_bc_probe.py --teacher runs/m2_mlp_e/checkpoints/it00055_m2_ok.pt \
        --init-from runs/anda_r5I/best_it6.pt --out runs/m6_bc_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.brain.controller import ConnectomePolicy, ControllerConfig, load_walking  # noqa: E402
from mosca.brain.graph import Connectome  # noqa: E402
from mosca.env.skate_env import EnvConfig, RewardConfig, SkateVecEnv  # noqa: E402
from mosca.paths import MALECNS_DIR  # noqa: E402
from mosca.rl.ppo import ActorCritic, RunningNorm  # noqa: E402

PROBE_ATTEMPT_BASE = 2 * 10**9  # tentativas que não aparecem nem no treino nem nos testes fixos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", required=True, help="checkpoint da MLP patinadora (m2_train.py)")
    p.add_argument("--init-from", required=True, help="conectoma que anda (m4_distill.py)")
    p.add_argument("--graph", default=str(MALECNS_DIR / "controller_graph_min5.npz"))
    p.add_argument("--train", type=int, default=16, help="tentativas de treino")
    p.add_argument("--test", type=int, default=8, help="tentativas de teste")
    p.add_argument("--seconds", type=float, default=0.0, help="duração das tentativas (padrão: a da professora)")
    p.add_argument("--v-min", type=float, default=1.0)
    p.add_argument("--v-max", type=float, default=3.5)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=8, help="tentativas por minilote")
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mults", default="rede=0.3,codificador=3,tonus=10,decodificador=1,sinapses=0.1,comando=0",
                   help="multiplicadores da taxa por grupo")
    p.add_argument("--tau-scale", type=float, default=0.25)
    p.add_argument("--substeps", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default="")
    return p.parse_args()


def collect(env: SkateVecEnv, ac: ActorCritic, norm: RunningNorm, v_cmd: np.ndarray):
    """Tentativas conduzidas pela professora: observações, ações dela e passos válidos (T, n, ...)."""
    obs, _ = env.reset(PROBE_ATTEMPT_BASE + np.arange(env.n), v_cmd)
    O, A, V = [], [], []
    while env.alive.any():
        with torch.no_grad():
            act = ac.actor(torch.from_numpy(norm(obs).astype(np.float32))).numpy()
        O.append(obs.astype(np.float32))
        A.append(act)
        V.append(env.alive.copy())
        obs, *_ = env.step(act)
    return np.array(O), np.array(A), np.array(V)


def filtered(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Filtro de primeira ordem dos atuadores ao longo do tempo (T, n, A), partindo de zero."""
    y = np.zeros_like(x)
    prev = np.zeros(x.shape[1:])
    for t in range(len(x)):
        prev = prev + alpha * (x[t] - prev)
        y[t] = prev
    return y


def nmse(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    """1 − R² nos passos válidos, somando todas as saídas."""
    p, y = pred[valid], target[valid]
    return float(((p - y) ** 2).sum() / ((y - y.mean(axis=0)) ** 2).sum())


def student_actions(pol: ConnectomePolicy, obs: np.ndarray, v_cmd: np.ndarray, device) -> np.ndarray:
    n = obs.shape[1]
    r = pol.initial_state(n)
    v = torch.as_tensor(v_cmd, dtype=torch.float32, device=device)
    out = []
    with torch.no_grad():
        for t in range(len(obs)):
            r, mean = pol(r, torch.as_tensor(obs[t], device=device), v)
            out.append(mean.cpu().numpy())
    return np.array(out)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    ck = torch.load(args.teacher, weights_only=False)
    env_cfg = dict(ck["env_cfg"])
    env_cfg["reward"] = RewardConfig(**env_cfg["reward"])
    n = args.train + args.test
    env_cfg.update(n_envs=n, n_threads=args.threads)
    if args.seconds > 0:
        env_cfg["episode_seconds"] = args.seconds
    env = SkateVecEnv(EnvConfig(**env_cfg))
    ac = ActorCritic(env.obs_dim, env.priv_dim, env.act_dim)
    ac.load_state_dict(ck["ac"])
    norm = RunningNorm(env.obs_dim)
    norm.load_state_dict(ck["norm_obs"])
    v_cmd = np.random.default_rng(0).uniform(args.v_min, args.v_max, n)
    t0 = time.perf_counter()
    obs, act, valid = collect(env, ac, norm, v_cmd)
    stats = env.episode_stats()
    print(f"professora: {n} tentativas em {time.perf_counter() - t0:.0f} s; velocidade média "
          f"{np.mean([s['speed'] for s in stats]):.2f} cm/s, deslizando {np.mean([s['glide_frac'] for s in stats]):.0%}, "
          f"quedas {np.mean([s['fell'] for s in stats]):.0%}")
    m = env.model
    tau = m.actuator_dynprm[env.leg_act, 0]
    alpha = 1.0 - (1.0 - m.opt.timestep / tau) ** env.substeps
    tr, te = np.arange(args.train), np.arange(args.train, n)
    clip = env.cfg.action_clip
    print(f"ações da professora: média {act[valid].mean():+.2f}, desvio no tempo {act[valid].std(axis=0).mean():.2f}, "
          f"fora de ±{clip:g}: {(np.abs(act[valid]) > clip).mean():.1%} (cortadas, como no ambiente)")
    act = np.clip(act, -clip, clip)
    act_f = filtered(act, alpha)

    cfg = ControllerConfig(control_dt=env.cfg.control_dt, substeps=args.substeps, tau_scale=args.tau_scale,
                           all_synapse_gains=True, haltere_input=True, haltere_scale=0.5, haltere_offset=0.0,
                           turn_cells=("DNa02",), turn_gain=1500.0, enc_std=2.0, dec_gain=0.01)
    pol = ConnectomePolicy(Connectome.load(Path(args.graph)), cfg, device=device)
    print(f"partindo de {args.init_from}: {load_walking(pol, args.init_from, device)} tensores/ganhos copiados")
    obs_rest, _ = env.reset(np.arange(env.n) + 10**8, np.full(env.n, 1.0))
    pol.calibrate_rest(torch.as_tensor(obs_rest[:8], dtype=torch.float32, device=device), 0.65)
    env.close()
    # A postura da professora é outra: o viés do decodificador começa na média das ações dela, e o erro
    # mede a parte que muda no tempo (com a taxa do treino, só o deslocamento levaria dezenas de épocas).
    pred = student_actions(pol, obs[:, tr], v_cmd[tr], device)
    with torch.no_grad():
        pol.dec_bias += torch.as_tensor(act[:, tr][valid[:, tr]].mean(axis=0) - pred[valid[:, tr]].mean(axis=0),
                                        dtype=torch.float32, device=device)
    mults = {"rede": 1.0, "codificador": 1.0, "tonus": 1.0, "decodificador": 1.0, "comando": 0.0, "exploracao": 0.0,
             "sinapses": 1.0}
    for item in filter(None, args.mults.split(",")):
        k, val = item.split("=")
        mults[k] = float(val)
    opt = torch.optim.Adam([{"params": ps, "lr": args.lr * mults[k]} for k, ps in pol.param_groups().items()])

    def evaluate(tag: str) -> dict:
        pred = student_actions(pol, obs[:, te], v_cmd[te], device)
        res = {"raw": nmse(pred, act[:, te], valid[:, te]),
               "filtered": nmse(filtered(pred, alpha), act_f[:, te], valid[:, te])}
        print(f"{tag}: erro no teste (1 − R²) cru {res['raw']:.3f}, filtrado {res['filtered']:.3f}", flush=True)
        return res

    history = [{"epoch": 0, **evaluate("época 0 (conectoma que anda, só com o viés alinhado)")}]
    T = obs.shape[0]
    obs_t = torch.as_tensor(obs, device=device)
    act_t = torch.as_tensor(act, device=device)
    valid_t = torch.as_tensor(valid, device=device)
    v_t = torch.as_tensor(v_cmd, dtype=torch.float32, device=device)
    gen = np.random.default_rng(1)
    for epoch in range(1, args.epochs + 1):
        t0, losses = time.perf_counter(), []
        order = gen.permutation(tr)
        for s in range(0, len(order), args.batch):
            idx = torch.as_tensor(order[s : s + args.batch], device=device)
            r = pol.initial_state(len(idx))
            for c0 in range(0, T, args.chunk):
                r = r.detach()
                loss, count = 0.0, valid_t[c0 : c0 + args.chunk][:, idx].sum().clamp(min=1)
                for t in range(c0, min(T, c0 + args.chunk)):
                    r, mean = pol(r, obs_t[t, idx], v_t[idx])
                    err = ((mean - act_t[t, idx]) ** 2).sum(-1)
                    loss = loss + (err * valid_t[t, idx]).sum()
                loss = loss / count
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
        res = evaluate(f"época {epoch} ({time.perf_counter() - t0:.0f} s, perda de treino {np.mean(losses):.3f})")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **res})
        if args.out:
            Path(args.out).write_text(json.dumps({"args": vars(args), "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
