"""M4: o conectoma aprende a andar imitando a política de caminhada do flybody (DAgger).

A professora (mosca.walking.teacher) segue uma trajetória de referência; o aluno (o
controlador de conectoma, corpo "walk") só recebe a velocidade e o giro pedidos e os
sentidos das próprias patas. A cada iteração:
1. Uma fração β das moscas é conduzida pela professora e o resto pelo aluno (β cai de 1 a
   `--beta-min` em `--beta-iters` iterações). Nas do aluno, a cada passo a ação executada é a
   da professora com probabilidade `--mix` (caindo até `--mix-min`): o aluno vê os próprios
   erros, e as intervenções evitam a queda imediata. Em todas, a professora diz, a cada 2 ms,
   o que faria naquele estado; esse é o alvo do aluno (a ação dela já cortada nos limites).
2. O aluno treina por retropropagação em trechos de `--chunk` passos, sorteados das últimas
   `--buffer-iters` iterações: cada trecho recomeça do estado da rede gravado na coleta, com
   `--burn-in` passos sem gradiente para atualizar o estado, e o erro quadrático normalizado
   (por saída) nos passos restantes.

Critério do M4 (docs/plano.md): ≥90% de caminhadas de 5 s bem-sucedidas a 1–3 cm/s, e o
DNa02 faz virar.

Uso (no Spark):
    python scripts/m4_distill.py --run anda_a --iters 60 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.brain.controller import ConnectomePolicy, ControllerConfig  # noqa: E402
from mosca.brain.graph import Connectome  # noqa: E402
from mosca.capture import library_versions  # noqa: E402
from mosca.env.skate_env import attempt_seed  # noqa: E402
from mosca.paths import MALECNS_DIR, RUNS  # noqa: E402
from mosca.walking.teacher import FUTURE_STEPS, WalkingTeacher, straight_trajectory  # noqa: E402
from mosca.walking.vec_env import WalkingVecEnv  # noqa: E402

TEST_COMMANDS = [(v, y) for v in (1.0, 2.0, 3.0) for y in (0.0, 1.0, -1.0)]


class PuglieseNetKeys:
    @staticmethod
    def edge_ids_for_rows(pol, graph) -> torch.Tensor:
        """Índices, na lista de todas as ligações (ordem CSR), das ligações que chegam aos motores das
        patas, na mesma ordem em que o modo "só motores" guarda os seus ganhos (ordem COO por linha)."""
        from mosca.body.fly import LEGS as _LEGS
        from mosca.brain.pugliese import signed_matrix
        w = (0.03 * signed_matrix(graph)).tocsr()
        w.sort_indices()
        rows = np.zeros(w.shape[0], dtype=bool)
        rows[np.concatenate([graph.groups[f"motor_{leg}"] for leg in _LEGS])] = True
        edge_rows = np.repeat(np.arange(w.shape[0]), np.diff(w.indptr))
        coo = w.tocoo()
        order_csr = np.flatnonzero(rows[edge_rows])
        order_coo = np.flatnonzero(rows[coo.row])
        # COO de uma CSR ordenada percorre as ligações na mesma ordem da CSR
        assert np.array_equal(coo.row[order_coo], edge_rows[order_csr])
        return torch.as_tensor(order_csr, device=pol.device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-gpu-mem-gb", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--graph", default=str(MALECNS_DIR / "controller_graph_min5.npz"),
                   help="grafo do controlador (o embaralhado, para o controle do M7)")
    p.add_argument("--episode-seconds", type=float, default=2.0)
    p.add_argument("--v-min", type=float, default=0.5)
    p.add_argument("--v-max", type=float, default=3.0)
    p.add_argument("--yaw-max", type=float, default=1.0)
    p.add_argument("--beta-iters", type=int, default=15)
    p.add_argument("--beta-min", type=float, default=0.1)
    p.add_argument("--mix", type=float, default=0.0, help="probabilidade inicial de a professora intervir a cada passo")
    p.add_argument("--mix-min", type=float, default=0.0)
    p.add_argument("--mix-iters", type=int, default=30)
    p.add_argument("--init-from", default="", help="checkpoint para partir (parâmetros com o mesmo nome e forma; "
                                                  "ganhos dos motores viram ganhos das mesmas ligações)")
    p.add_argument("--chunk", type=int, default=48)
    p.add_argument("--burn-in", type=int, default=16)
    p.add_argument("--minibatch-chunks", type=int, default=16)
    p.add_argument("--updates", type=int, default=60, help="passos do otimizador por iteração")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dec-lr-mult", type=float, default=10.0,
                   help="multiplicador da taxa do decodificador (os alvos têm médias de várias unidades)")
    p.add_argument("--buffer-iters", type=int, default=4)
    p.add_argument("--motor-synapse-gains", action="store_true",
                   help="degrau 2 da escada: ganho treinável por ligação nas entradas dos motores das patas")
    p.add_argument("--all-synapse-gains", action="store_true",
                   help="degrau 2 completo: ganho treinável por ligação em toda a rede (sinal fixo)")
    p.add_argument("--syn-lr-mult", type=float, default=10.0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-seconds", type=float, default=5.0)
    p.add_argument("--resume", action="store_true")
    cc = ControllerConfig()
    for name in ("size_ref", "walk_gain", "turn_gain", "enc_std", "prop_offset", "motor_tone", "dec_gain"):
        p.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(cc, name))
    return p.parse_args()


def to_dev(x, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def rollout(env, pol, teacher, refs, v_cmd, yaw, driver_teacher, chunk, device, mix=0.0, rng=None):
    """Uma leva de tentativas; devolve observações, alvos, máscara e os estados da rede por trecho."""
    env.reset(refs, v_cmd, yaw)
    T = min(len(r) for r in refs) - FUTURE_STEPS - 1
    n = env.n
    obs_buf = np.zeros((T, n, 186), np.float32)
    tgt_buf = np.zeros((T, n, pol.n_out), np.float32)
    valid = np.zeros((T, n), bool)
    states = torch.zeros(((T + chunk - 1) // chunk, pol.n, n), device=device)
    r = pol.initial_state(n)
    vc = to_dev(v_cmd, device)
    start = env.positions()
    steps = 0
    with torch.no_grad():
        for t in range(T):
            if not env.alive.any():
                break
            if t % chunk == 0:
                states[t // chunk] = r
            obs = env.student_obs()
            r, out = pol(r, to_dev(obs, device), vc)
            out_np = out.cpu().numpy()
            a_teacher = teacher(env.teacher_obs())
            target = env.student_target(a_teacher)
            use_teacher = driver_teacher | ((rng.random(n) < mix) if mix > 0 else False)
            ctrl = np.where(use_teacher[:, None], env.teacher_ctrl(a_teacher), env.ctrl_from_student(out_np))
            obs_buf[t], tgt_buf[t], valid[t] = obs, target, env.alive
            env.step(ctrl, out_np)
            steps = t + 1
    end = env.positions()
    return {"obs": obs_buf[:steps], "tgt": tgt_buf[:steps], "valid": valid[:steps], "states": states,
            "v_cmd": np.asarray(v_cmd, np.float32), "fell": env.fell.copy(), "t": env.t.copy(),
            "dist": np.linalg.norm(end[:, :2] - start[:, :2], axis=1)}


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", device.index or 0)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_mem_gb * 2**30 / total), device)
    run_dir = RUNS / args.run
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    env = WalkingVecEnv(args.envs, n_threads=args.threads)
    teacher = WalkingTeacher().to(device)
    graph = Connectome.load(Path(args.graph))
    cfg = ControllerConfig(body="walk", control_dt=0.002, substeps=1, size_ref=args.size_ref, walk_gain=args.walk_gain,
                           turn_gain=args.turn_gain, enc_std=args.enc_std, prop_offset=args.prop_offset,
                           motor_tone=args.motor_tone, dec_gain=args.dec_gain, seed=args.seed,
                           motor_synapse_gains=args.motor_synapse_gains, all_synapse_gains=args.all_synapse_gains)
    pol = ConnectomePolicy(graph, cfg, device=device)
    mults = {"rede": 1.0, "codificador": 10.0, "tonus": 3.0, "decodificador": args.dec_lr_mult, "comando": 10.0,
             "exploracao": 0.0, "sinapses": args.syn_lr_mult}
    opt = torch.optim.Adam([{"params": ps, "lr": args.lr * mults[name], "name": name}
                            for name, ps in pol.param_groups().items()])
    state = {"it": 0}
    tgt_std = None
    latest = run_dir / "latest.pt"
    if args.init_from and not (args.resume and latest.exists()):
        src = torch.load(args.init_from, weights_only=False, map_location=device)
        own = pol.state_dict()
        copied = {k: v for k, v in src["policy"].items() if k in own and own[k].shape == v.shape}
        own.update(copied)
        pol.load_state_dict(own)
        tgt_std = src.get("tgt_std")
        if "net.log_gain" in src["policy"] and pol.net.edge_gains:  # ganhos dos motores -> mesmas ligações
            old = PuglieseNetKeys.edge_ids_for_rows(pol, graph)
            with torch.no_grad():
                pol.net.log_edge_gain[old] = src["policy"]["net.log_gain"].to(device)
        print(f"partindo de {args.init_from}: {len(copied)} tensores copiados", flush=True)
    if args.resume and latest.exists():
        ckpt = torch.load(latest, weights_only=False, map_location=device)
        pol.load_state_dict(ckpt["policy"])
        opt.load_state_dict(ckpt["opt"])
        state, tgt_std = ckpt["state"], ckpt["tgt_std"]
        print(f"retomando da iteração {state['it']}")
    else:
        (run_dir / "config.json").write_text(json.dumps(
            {"args": vars(args), "controller": asdict(cfg), "neurons": graph.n,
             "trainable": sum(p.numel() for p in pol.parameters()), "versions": library_versions()}, indent=2),
            encoding="utf-8")

    buffer: list[dict] = []
    ref_steps = round(args.episode_seconds / 0.002) + FUTURE_STEPS + 1
    print(f"conectoma (corpo walk): {graph.n:,} neurônios, {sum(p.numel() for p in pol.parameters()):,} parâmetros; {device}")
    print("  it   beta  erro_norm (início→fim)  quedas(aluno)  vel_aluno/pedida  coleta  treino")
    for it in range(state["it"], args.iters):
        t0 = time.perf_counter()
        rng = np.random.default_rng(attempt_seed(args.seed, it))
        n = env.n
        v_cmd = rng.uniform(args.v_min, args.v_max, n)
        yaw = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(-args.yaw_max, args.yaw_max, n))
        refs = [straight_trajectory(ref_steps, v, yaw_speed=y, heading=rng.uniform(-np.pi, np.pi)) for v, y in zip(v_cmd, yaw)]
        beta = max(args.beta_min, 1.0 - it / max(args.beta_iters, 1))
        driver_teacher = np.zeros(n, bool)
        driver_teacher[rng.permutation(n)[: round(beta * n)]] = True
        mix = max(args.mix_min, args.mix * (1.0 - it / max(args.mix_iters, 1))) if args.mix > 0 else 0.0
        ro = rollout(env, pol, teacher, refs, v_cmd, yaw, driver_teacher, args.chunk, device, mix=mix, rng=rng)
        t_collect = time.perf_counter() - t0
        buffer = (buffer + [ro])[-args.buffer_iters :]
        if tgt_std is None:  # escala de cada saída, fixada na primeira coleta (só a professora)
            tgt_std = np.maximum(ro["tgt"][ro["valid"]].std(axis=0), 0.05).astype(np.float32)
            with torch.no_grad():  # o viés do decodificador começa na média dos alvos
                pol.dec_bias.copy_(to_dev(ro["tgt"][ro["valid"]].mean(axis=0), device))
        std_t = to_dev(tgt_std, device)

        # Treino: trechos sorteados do buffer.
        t1 = time.perf_counter()
        losses = []
        segs = [(b, k, i) for b, r_ in enumerate(buffer) for k in range(r_["states"].shape[0]) for i in range(n)
                if k * args.chunk < len(r_["valid"]) and r_["valid"][k * args.chunk, i]]
        for u in range(args.updates):
            pick = [segs[j] for j in rng.choice(len(segs), size=min(args.minibatch_chunks, len(segs)), replace=False)]
            r = torch.stack([buffer[b]["states"][k][:, i] for b, k, i in pick], dim=1)
            loss_sum, count = 0.0, 0.0
            for t in range(args.chunk):
                obs_t = np.stack([buffer[b]["obs"][min(k * args.chunk + t, len(buffer[b]["obs"]) - 1), i] for b, k, i in pick])
                v_t = to_dev([buffer[b]["v_cmd"][i] for b, k, i in pick], device)
                if t < args.burn_in:
                    with torch.no_grad():
                        r, _ = pol(r, to_dev(obs_t, device), v_t)
                    continue
                r, out = pol(r, to_dev(obs_t, device), v_t)
                tgt_t = np.stack([buffer[b]["tgt"][min(k * args.chunk + t, len(buffer[b]["tgt"]) - 1), i] for b, k, i in pick])
                mask = np.array([k * args.chunk + t < len(buffer[b]["valid"]) and buffer[b]["valid"][k * args.chunk + t, i]
                                 for b, k, i in pick], dtype=np.float32)
                err = ((out - to_dev(tgt_t, device)) / std_t).pow(2).mean(dim=1)
                m_t = to_dev(mask, device)
                loss_sum = loss_sum + (err * m_t).sum()
                count += mask.sum()
            loss = loss_sum / max(count, 1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        t_train = time.perf_counter() - t1

        student = ~driver_teacher
        speed = ro["dist"] / np.maximum(ro["t"] * 0.002, 1e-6)
        ratio = float(np.mean(speed[student] / v_cmd[student])) if student.any() else float("nan")
        falls = float(np.mean(ro["fell"][student])) if student.any() else float("nan")
        metrics = {"it": it, "beta": beta, "mix": mix, "loss": float(np.mean(losses)), "loss_first": float(np.mean(losses[:5])),
                   "loss_last": float(np.mean(losses[-5:])), "student_falls": falls,
                   "student_speed_ratio": ratio, "time_collect": t_collect, "time_train": t_train}
        if args.eval_every and (it + 1) % args.eval_every == 0:
            cmds = [TEST_COMMANDS[j % len(TEST_COMMANDS)] for j in range(n)]
            ev_refs = [straight_trajectory(round(args.eval_seconds / 0.002) + FUTURE_STEPS + 1, v, yaw_speed=y) for v, y in cmds]
            ev = rollout(env, pol, teacher, ev_refs, np.array([c[0] for c in cmds]), np.array([c[1] for c in cmds]),
                         np.zeros(n, bool), args.chunk, device)
            ok = (~ev["fell"]) & (ev["t"] >= round(args.eval_seconds / 0.002) - 1)
            ev_speed = ev["dist"] / np.maximum(ev["t"] * 0.002, 1e-6)
            metrics.update(eval_success=float(ok.mean()), eval_falls=float(ev["fell"].mean()),
                           eval_speed_ratio=float(np.mean(ev_speed / np.array([c[0] for c in cmds]))))
            print(f"      teste sem professora ({args.eval_seconds:g} s): {ok.mean():.0%} sem cair até o fim, "
                  f"velocidade/pedida {metrics['eval_speed_ratio']:.2f}", flush=True)
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        print(f"{it:4d} {beta:6.2f} {metrics['loss']:10.3f} ({metrics['loss_first']:.2f}→{metrics['loss_last']:.2f}) "
              f"{falls:14.2f} {ratio:17.2f} {t_collect:7.1f} {t_train:7.1f}", flush=True)
        state["it"] = it + 1
        ckpt = {"policy": pol.state_dict(), "opt": opt.state_dict(), "state": state, "tgt_std": tgt_std,
                "controller": asdict(cfg), "args": vars(args)}
        torch.save(ckpt, latest.with_suffix(".tmp"))
        latest.with_suffix(".tmp").replace(latest)
        if (it + 1) % 10 == 0:
            torch.save(ckpt, run_dir / "checkpoints" / f"it{it + 1:05d}.pt")
    env.close()


if __name__ == "__main__":
    main()
