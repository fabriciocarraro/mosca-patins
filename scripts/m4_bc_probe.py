"""M4: triagem de configurações do conectoma por imitação offline, com o erro filtrado.

Coleta uma vez caminhadas conduzidas pela professora e treina cada variante do aluno por
algumas épocas de retropropagação truncada. Mede o erro normalizado das ações depois do filtro
dos atuadores em moscas separadas para teste, e a resposta da rede treinada a desvios pequenos
das juntas de uma pata (tamanho e atraso). É bem mais barato que a destilação completa e
separa o que a rede consegue representar do que o DAgger consegue ensinar. Referência: a MLP
com as mesmas entradas chega a ~0,2 de erro filtrado já na primeira iteração da destilação, e
anda.

Cada variante é "nome:campo=valor,campo=valor" sobre o ControllerConfig (corpo "walk").

Uso (no Spark):
    python scripts/m4_bc_probe.py --device cuda --variants "base:" "tonico:prop_offset=2" "rapido:tau_scale=0.25"
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.brain.controller import ConnectomePolicy, ControllerConfig, leg_feature_index  # noqa: E402
from mosca.brain.graph import Connectome  # noqa: E402
from mosca.paths import MALECNS_DIR, RUNS  # noqa: E402
from mosca.walking.teacher import FUTURE_STEPS, WalkingTeacher, straight_trajectory  # noqa: E402
from mosca.walking.vec_env import WalkingVecEnv  # noqa: E402

BASE = dict(body="walk", control_dt=0.002, substeps=1, enc_std=2.0, dec_gain=0.01, all_synapse_gains=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--test-envs", type=int, default=8)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--syn-lr-mult", type=float, default=3.0)
    p.add_argument("--raw-loss", action="store_true", help="treina no erro cru (sem o filtro dos atuadores)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-gpu-mem-gb", type=float, default=6.0)
    p.add_argument("--graph", default=str(MALECNS_DIR / "controller_graph_min5.npz"))
    return p.parse_args()


def parse_variant(spec: str) -> tuple[str, ControllerConfig]:
    name, _, rest = spec.partition(":")
    types = {f.name: f.type for f in fields(ControllerConfig)}
    kw = dict(BASE)
    for item in filter(None, rest.split(",")):
        key, value = item.split("=")
        kw[key] = value.lower() in ("1", "true", "sim") if types[key] in (bool, "bool") else float(value)
    return name, ControllerConfig(**kw)


def collect(args, alpha):
    path = RUNS / f"bc_probe_data_{args.envs}x{round(args.seconds / 0.002)}.npz"
    if path.exists():
        d = np.load(path)
        return d["obs"], d["tgt"], d["tgt_f"], d["valid"], d["v"]
    rng = np.random.default_rng(11)
    n, steps = args.envs, round(args.seconds / 0.002)
    env, teacher = WalkingVecEnv(n, n_threads=args.threads), WalkingTeacher().to(args.device)
    v = rng.uniform(0.5, 3.0, n)
    yaw = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(-1.0, 1.0, n))
    env.reset([straight_trajectory(steps + FUTURE_STEPS + 1, a, yaw_speed=b, heading=rng.uniform(-np.pi, np.pi))
               for a, b in zip(v, yaw)], v, yaw)
    obs, tgt, valid = [], [], []
    for _ in range(steps):
        o, a = env.student_obs(), teacher(env.teacher_obs())
        obs.append(o), tgt.append(env.student_target(a)), valid.append(env.alive.copy())
        env.step(env.teacher_ctrl(a))
    env.close()
    obs, tgt, valid = np.array(obs, np.float32), np.array(tgt, np.float32), np.array(valid)
    tgt_f = np.empty_like(tgt)
    tgt_f[0] = tgt[0]
    for t in range(1, len(tgt)):
        tgt_f[t] = tgt_f[t - 1] + alpha * (tgt[t] - tgt_f[t - 1])
    RUNS.mkdir(exist_ok=True)
    np.savez(path, obs=obs, tgt=tgt, tgt_f=tgt_f, valid=valid, v=v.astype(np.float32))
    return obs, tgt, tgt_f, valid, v.astype(np.float32)


@torch.no_grad()
def responsiveness(pol: ConnectomePolicy, device) -> tuple[float, float]:
    """Mediana, nas 14 grandezas da pata T1 esquerda (±0,1 rad; ±5 rad/s), da soma de |Δação| nas
    42 saídas das juntas, e do tempo até metade da resposta máxima (ms), com a mosca parada."""
    n = 14
    obs = torch.zeros(2 * n + 1, 186, device=device)
    obs[:, 4 * 42 + 9 : 4 * 42 + 15] = 1 / 6
    v = torch.full((2 * n + 1,), 2.0, device=device)
    r = pol.initial_state(2 * n + 1)
    for _ in range(500):
        r, _ = pol(r, obs, v)
    feats = leg_feature_index(0)[:n]
    for j in range(n):
        d = 0.1 if j < 7 else 0.5
        obs[j, feats[j]] += d
        obs[n + j, feats[j]] -= d
    resp = []
    for _ in range(100):
        r, out = pol(r, obs, v)
        resp.append((out[: 2 * n, :42] - out[-1:, :42]).abs().sum(dim=1).cpu().numpy())
    resp = np.array(resp)
    peak = resp.max(axis=0)
    t50 = 2 * np.argmax(resp >= 0.5 * np.maximum(peak, 1e-9)[None], axis=0)
    return float(np.median(peak)), float(np.median(t50))


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", device.index or 0)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_mem_gb * 2**30 / total), device)
    env_alpha = WalkingVecEnv(1, n_threads=1).act_alpha.astype(np.float32)
    obs, tgt, tgt_f, valid, v = collect(args, env_alpha)
    T, n = tgt.shape[:2]
    tr, te = np.arange(n - args.test_envs), np.arange(n - args.test_envs, n)
    warm = 100  # passos iniciais fora do erro (a rede ainda está saindo do repouso)
    std = np.maximum(tgt[warm:, tr][valid[warm:, tr]].std(0), 0.05)
    std_f = np.maximum(tgt_f[warm:, tr][valid[warm:, tr]].std(0), 0.05)
    to = lambda x: torch.as_tensor(x, device=device)  # noqa: E731
    OBS, TGT, TGT_F, VALID = to(obs), to(tgt), to(tgt_f), to(valid.astype(np.float32))
    STD, STD_F, ALPHA, V = to(std.astype(np.float32)), to(std_f.astype(np.float32)), to(env_alpha), to(v)
    graph = Connectome.load(Path(args.graph))
    print(f"dados: {n} moscas × {T} passos (professora); treino {len(tr)}, teste {len(te)}; erro {'cru' if args.raw_loss else 'filtrado'}")
    mults = {"rede": 1.0, "codificador": 10.0, "tonus": 3.0, "decodificador": 10.0, "comando": 10.0, "exploracao": 0.0,
             "sinapses": args.syn_lr_mult}

    def run(pol, idx, train, opt=None):
        """Uma passada pela sequência inteira; devolve os erros médios (cru, filtrado) nos passos válidos."""
        idx_t = to(idx)
        r = pol.initial_state(len(idx))
        out_f = None
        sums = np.zeros(2)
        count = 0.0
        for s0 in range(0, T, args.chunk):
            loss_c, loss_f, cnt = 0.0, 0.0, 0.0
            with torch.set_grad_enabled(train):
                for t in range(s0, min(s0 + args.chunk, T)):
                    r, out = pol(r, OBS[t, idx_t], V[idx_t])
                    out_f = out if out_f is None else out_f + ALPHA * (out - out_f)
                    if t < warm:
                        continue
                    m = VALID[t, idx_t]
                    loss_c = loss_c + ((((out - TGT[t, idx_t]) / STD) ** 2).mean(dim=1) * m).sum()
                    loss_f = loss_f + ((((out_f - TGT_F[t, idx_t]) / STD_F) ** 2).mean(dim=1) * m).sum()
                    cnt += float(m.sum())
            if cnt > 0:
                sums += [loss_c.item(), loss_f.item()]
                count += cnt
                if train:
                    loss = (loss_c if args.raw_loss else loss_f) / cnt
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                    opt.step()
            r, out_f = r.detach(), out_f.detach()
        return sums / max(count, 1.0)

    for spec in args.variants:
        name, cfg = parse_variant(spec)
        pol = ConnectomePolicy(graph, cfg, device=device)
        with torch.no_grad():
            pol.dec_bias.copy_(to(tgt[warm:, tr][valid[warm:, tr]].mean(0)))
        opt = torch.optim.Adam([{"params": ps, "lr": args.lr * mults[k]} for k, ps in pol.param_groups().items()])
        mag, lat = responsiveness(pol, device)
        raw, filt = run(pol, te, train=False)
        print(f"[{name}] {spec}\n  início: teste cru {raw:.3f}, filtrado {filt:.3f}; resposta {mag:.3f} em {lat:.0f} ms", flush=True)
        t0 = time.perf_counter()
        for ep in range(args.epochs):
            train_raw, train_filt = run(pol, tr, train=True, opt=opt)
            raw, filt = run(pol, te, train=False)
            print(f"  época {ep + 1}: treino {train_raw:.3f}/{train_filt:.3f} | teste cru {raw:.3f}, filtrado {filt:.3f} "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
        mag, lat = responsiveness(pol, device)
        _, _, tau = pol.net.neuron_params()
        print(f"  fim: resposta {mag:.3f} em {lat:.0f} ms; τ mediano {1000 * tau.median().item():.1f} ms", flush=True)


if __name__ == "__main__":
    main()
