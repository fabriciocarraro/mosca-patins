"""M8: re-simula tentativas gravadas e guarda o que o vídeo precisa (roda onde a execução rodou).

Para cada tentativa ("Tentativa #N", numerada a partir de 1 no vídeo e de 0 aqui):
- física refeita sozinha com as ações gravadas, conferida contra os estados gravados (desvio 0 = bit a bit), com a
  postura (qpos) guardada a cada subpasso de física (0,2 ms), para a câmera lenta de verdade;
- cérebro re-simulado na leva inteira da geração (como no m5_rehearsal), com a atividade de todos os neurônios da
  tentativa guardada a cada passo de controle, e a maior diferença entre a ação recalculada e a gravada.

Com `--checkpoint`, roda um teste fixo (as mesmas sementes dos testes do M2, sem variação nem ruído) a partir do
checkpoint, com intervenção opcional: `--silence DNg100` (inibição contínua, como no `m6_eval.py --probes`) ou
`--stim DNa02_L` (corrente de um comando de giro de 1 rad/s nos DNa02 de um lado, vezes `--stim-scale`). Um teste é
re-simulável por construção: é o mesmo cálculo do `m6_eval.py`, a partir do mesmo checkpoint. Com `--tests N`, os N
primeiros testes de uma vez (a grade da pista reta), só com as posturas, a cada `--every` subpassos. Com `--walk`, o
conectoma que anda (checkpoint do M4) no corpo de caminhada, sem patins, partindo parado e com rumo 0.

Renderizar não exige a mesma máquina: o `m8_render.py` só desenha as posturas guardadas.

Saída: runs/<execução>/video/tentativa_<N+1>.npz, teste<K+1>_v<velocidade>_<intervenção>_g<geração>.npz ou
grade<N>_v<velocidade>_<intervenção>_g<geração>.npz.

Uso (no Spark):
    python scripts/m8_trace.py --run final_s0 --attempts 0,43903 --device cuda
    python scripts/m8_trace.py --checkpoint runs/final_s0/checkpoints/gen02000.pt --test 0 --speed 3.5 --silence DNg100
    python scripts/m8_trace.py --checkpoint runs/final2_s0/checkpoints/gen01400.pt --tests 20 --speed 3.5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.brain.controller import ConnectomePolicy, ControllerConfig  # noqa: E402
from mosca.brain.graph import Connectome  # noqa: E402
from mosca.capture import IterationCapture, noise_beta, replay, replay_brain  # noqa: E402
from mosca.env.skate_env import EnvConfig, RewardConfig, attempt_seed  # noqa: E402
from mosca.paths import MALECNS_DIR, RUNS  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run_test(pol, graph, ckpt, args, test: int, device, every: int = 1, keep_rates: bool = True):
    """Um teste fixo (sementes dos testes do M2, sem variação nem ruído), com a intervenção de `args`; devolve as
    posturas a cada `every` subpassos, a atividade e a ação média a cada passo de controle, o giro pedido, as
    estatísticas da tentativa e o rótulo da intervenção."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from m6_eval import TEST_ATTEMPT_BASE, make_env
    from mosca.walking.evaluate import side_current

    env = make_env(ckpt, 1, 1, args.seconds)
    poses, rates, means, yaws = [], [], [], []
    count = [0]

    def keep_pose(e, sub):
        if count[0] % every == 0:
            poses.append(e.datas[0].qpos.astype(np.float32))
        count[0] += 1

    env.substep_callback = keep_pose
    extra, silence, tag = None, None, "base"
    if args.silence:
        silence = torch.as_tensor(np.concatenate([graph.groups[f"{args.silence}_L"], graph.groups[f"{args.silence}_R"]]),
                                  device=device)
        extra = torch.zeros(pol.n, 1, device=device)
        extra[silence] = -1e4  # inibição contínua (zerar a taxa só no fim do passo vaza pelos subpassos de RK4)
        tag = f"{args.silence}_calado"
    elif args.stim:
        cell, side = args.stim.rsplit("_", 1)
        extra = side_current(pol, graph.groups, cell, np.array([1.0 if side == "L" else -1.0]), device) * args.stim_scale
        tag = args.stim + (f"_x{args.stim_scale:g}" if args.stim_scale != 1 else "")
    obs, _ = env.reset(np.array([TEST_ATTEMPT_BASE + test]), np.array([args.speed]), yaw_cmd=np.array([args.yaw]))
    v = torch.full((1,), float(args.speed), device=device)
    r = pol.initial_state(1)
    with torch.no_grad():
        while env.alive.any():
            current = pol.currents(torch.as_tensor(obs, dtype=torch.float32, device=device), v)
            if extra is not None:
                current = current + extra
            r = pol.step_net(r, current)
            if silence is not None:
                r[silence] = 0.0
            mean = pol.decode(r)
            if keep_rates:
                rates.append(r[:, 0].float().cpu().numpy().astype(np.float16))
            means.append(mean[0].float().cpu().numpy())
            yaws.append(float(env.yaw_cmd[0]))
            obs, *_ = env.step(mean.cpu().numpy())
    stats = env.episode_stats()[0]
    substeps = env.substeps
    env.close()
    return poses, rates, means, yaws, stats, tag, substeps


def clean(stats: dict) -> dict:
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer, bool)) else v) for k, v in stats.items()}


def rerun_args(args) -> dict:
    """Os argumentos que refazem um teste (gravados no traço, para a conferência do m8_verify.py)."""
    return {k: getattr(args, k) for k in ("test", "tests", "every", "speed", "yaw", "seconds", "silence", "stim",
                                          "stim_scale", "walk")}


def trace_grid(args, device, save: bool = True) -> np.ndarray:
    """Os `--tests` primeiros testes fixos a partir de um checkpoint (a grade da pista reta), sem atividade do
    cérebro e com as posturas a cada `--every` subpassos."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from m6_eval import load

    ckpt_path = Path(args.checkpoint)
    pol, graph, ckpt, train_args = load(str(ckpt_path), device)
    all_poses, steps, stats = [], [], []
    for test in range(args.tests):
        poses, _, means, _, s, tag, substeps = run_test(pol, graph, ckpt, args, test, device, args.every, keep_rates=False)
        all_poses.append(np.stack(poses))
        steps.append(len(means))
        stats.append(clean(s))
        print(f"teste {test + 1}: {s['seconds']:.2f} s, {s['speed']:.2f} cm/s, giro médio {s['yaw_rate']:+.2f} rad/s, "
              f"{'caiu' if s['fell'] else 'não caiu'}", flush=True)
    length = max(len(p) for p in all_poses)
    qpos = np.stack([np.concatenate([p, np.repeat(p[-1:], length - len(p), axis=0)]) for p in all_poses])
    if not save:
        return qpos
    state = ckpt["state"]
    gen = state.get("gen", state.get("it"))
    run = ckpt_path.parent.parent.name if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent.name
    out_dir = RUNS / run / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"grade{args.tests}_v{args.speed:g}_{tag}_g{gen}.npz"
    meta = {"mode": "grade", "run": run, "graph": Path(train_args["graph"]).name, "checkpoint": str(ckpt_path),
            "checkpoint_sha256": sha256(ckpt_path), "generation": gen, "tests": args.tests, "intervention": tag,
            "toe_out_deg": float(ckpt["env_cfg"].get("toe_out_deg", 0.0)), "v_cmd": float(args.speed),
            "yaw": float(args.yaw), "control_dt": ckpt["env_cfg"]["control_dt"], "substeps": substeps,
            "every": args.every, "steps": steps, "device": str(device), "args": rerun_args(args), "stats": stats}
    np.savez_compressed(out, qpos=qpos, meta=json.dumps(meta, ensure_ascii=False))
    fast = sum(not s["fell"] and s["speed"] >= 3.0 for s in stats)
    print(f"{args.tests} testes (geração {gen}, pedido {args.speed:g} cm/s): {fast} com média ≥3 cm/s sem cair -> {out}")
    return qpos


def trace_walk(args, device, save: bool = True) -> np.ndarray:
    """Um teste do conectoma que anda (M4) no corpo de caminhada, sem patins, como no m4_eval.py: parte parado, rumo 0,
    com o comando de velocidade e giro e a intervenção opcional (a mesma do modo teste)."""
    from mosca.walking.evaluate import DT, side_current
    from mosca.walking.teacher import FUTURE_STEPS, heading_of, straight_trajectory
    from mosca.walking.vec_env import WalkingVecEnv

    ckpt_path = Path(args.checkpoint)
    ck = torch.load(ckpt_path, weights_only=False, map_location=device)
    ctrl = dict(ck["controller"])
    if "turn_cells" in ctrl:
        ctrl["turn_cells"] = tuple(ctrl["turn_cells"])
    graph_path = Path(ck["args"].get("graph") or MALECNS_DIR / "controller_graph_min5.npz")
    if not graph_path.exists():
        graph_path = MALECNS_DIR / graph_path.name
    graph = Connectome.load(graph_path)
    pol = ConnectomePolicy(graph, ControllerConfig(**ctrl), device=device)
    policy = dict(ck["policy"])
    if ck.get("ema"):  # como no m4_eval.py: checkpoint de iteração com a média móvel dos parâmetros
        policy.update({k: v.to(device) for k, v in ck["ema"].items()})
    pol.load_state_dict(policy)
    env = WalkingVecEnv(1, n_threads=1)
    poses = []
    env.substep_callback = lambda e, i: poses.append(e.datas[0].qpos.astype(np.float32))
    steps = round((args.seconds or 5.0) / DT)
    env.reset([straight_trajectory(steps + FUTURE_STEPS + 1, args.speed, yaw_speed=args.yaw, heading=0.0)],
              np.array([args.speed]), np.array([args.yaw]))
    extra, silence, tag = None, None, "base"
    if args.silence:
        silence = torch.as_tensor(np.concatenate([graph.groups[f"{args.silence}_L"], graph.groups[f"{args.silence}_R"]]),
                                  device=device)
        extra = torch.zeros(pol.n, 1, device=device)
        extra[silence] = -1e4
        tag = f"{args.silence}_calado"
    elif args.stim:
        cell, side = args.stim.rsplit("_", 1)
        extra = side_current(pol, graph.groups, cell, np.array([1.0 if side == "L" else -1.0]), device) * args.stim_scale
        tag = args.stim + (f"_x{args.stim_scale:g}" if args.stim_scale != 1 else "")
    r = pol.initial_state(1)
    vc = torch.full((1,), float(args.speed), device=device)
    rates, means = [], []
    with torch.no_grad():
        for _ in range(steps):
            if not env.alive.any():
                break
            current = pol.currents(torch.as_tensor(env.student_obs(), dtype=torch.float32, device=device), vc)
            if extra is not None:
                current = current + extra
            r = pol.step_net(r, current)
            if silence is not None:
                r[silence] = 0.0
            out = pol.decode(r)
            rates.append(r[:, 0].float().cpu().numpy().astype(np.float16))
            means.append(out[0].float().cpu().numpy())
            o = out.cpu().numpy()
            env.step(env.ctrl_from_student(o), o)
    root = env.obs_fn.root_qadr
    q = np.stack(poses)
    seconds = len(rates) * DT
    yaw0, yaw1 = (heading_of(q[k, root + 3:root + 7]) for k in (0, -1))
    stats = {"seconds": seconds, "fell": bool(env.fell[0]), "speed": float(np.linalg.norm(q[-1, root:root + 2] -
             q[0, root:root + 2]) / max(seconds, 1e-9)), "yaw_rate": float(np.arctan2(np.sin(yaw1 - yaw0),
                                                                                       np.cos(yaw1 - yaw0)) / max(seconds, 1e-9))}
    env.close()
    if not save:
        return q
    state = ck.get("state", {})
    gen = state.get("it", state.get("gen", 0))
    run = ckpt_path.parent.parent.name if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent.name
    out_dir = RUNS / run / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"anda_v{args.speed:g}_{tag}_{ckpt_path.stem}.npz"
    meta = {"mode": "teste", "body": "walk", "run": run, "graph": graph_path.name, "checkpoint": str(ckpt_path),
            "checkpoint_sha256": sha256(ckpt_path), "generation": gen, "test": 0, "video_number": None,
            "intervention": tag, "toe_out_deg": 0.0, "v_cmd": float(args.speed), "yaw": float(args.yaw), "push": 0.0,
            "steps": len(rates), "control_dt": DT, "substeps": env.substeps, "root_qadr": int(root),
            "device": str(device), "args": rerun_args(args), "stats": clean(stats)}
    np.savez_compressed(out, qpos=q, rates=np.stack(rates), mean_action=np.stack(means),
                        yaw_cmd=np.full(len(rates), args.yaw, np.float32), body_id=graph.body_id,
                        meta=json.dumps(meta, ensure_ascii=False))
    print(f"andando ({tag}, {ckpt_path.name}, pedido {args.speed:g} cm/s): {seconds:.2f} s, {stats['speed']:.2f} cm/s, "
          f"giro médio {stats['yaw_rate']:+.2f} rad/s, {'caiu' if stats['fell'] else 'não caiu'} -> {out}", flush=True)
    return q


def trace_test(args, device, save: bool = True) -> np.ndarray:
    """Um teste fixo a partir de um checkpoint, com intervenção opcional."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from m6_eval import load

    ckpt_path = Path(args.checkpoint)
    pol, graph, ckpt, train_args = load(str(ckpt_path), device)
    poses, rates, means, yaws, stats, tag, substeps = run_test(pol, graph, ckpt, args, args.test, device)
    if not save:
        return np.stack(poses)
    state = ckpt["state"]
    gen = state.get("gen", state.get("it"))
    run = ckpt_path.parent.parent.name if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent.name
    out_dir = RUNS / run / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"teste{args.test + 1}_v{args.speed:g}_{tag}_g{gen}.npz"
    meta = {"mode": "teste", "run": run, "graph": Path(train_args["graph"]).name, "checkpoint": str(ckpt_path),
            "checkpoint_sha256": sha256(ckpt_path), "generation": gen, "test": args.test, "video_number": None,
            "intervention": tag, "toe_out_deg": float(ckpt["env_cfg"].get("toe_out_deg", 0.0)), "v_cmd": float(args.speed),
            "yaw": float(args.yaw), "push": 0.0, "steps": len(rates), "control_dt": ckpt["env_cfg"]["control_dt"],
            "substeps": substeps, "device": str(device), "args": rerun_args(args), "stats": clean(stats)}
    np.savez_compressed(out, qpos=np.stack(poses), rates=np.stack(rates), mean_action=np.stack(means),
                        yaw_cmd=np.array(yaws, np.float32), body_id=graph.body_id,
                        meta=json.dumps(meta, ensure_ascii=False))
    print(f"teste {args.test + 1} ({tag}, geração {gen}, pedido {args.speed:g} cm/s): {stats['seconds']:.2f} s, "
          f"{stats['speed']:.2f} cm/s, giro médio {stats['yaw_rate']:+.2f} rad/s, "
          f"{'caiu' if stats['fell'] else 'não caiu'} -> {out}", flush=True)
    return np.stack(poses)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="execução gravada (modo captura)")
    p.add_argument("--attempts", help="números das tentativas, a partir de 0, separados por vírgula")
    p.add_argument("--checkpoint", help="modo teste: checkpoint do m6_es / connectome_train")
    p.add_argument("--test", type=int, default=0, help="número do teste fixo, a partir de 0")
    p.add_argument("--tests", type=int, default=0, help="modo grade: os N primeiros testes fixos, sem atividade do cérebro")
    p.add_argument("--every", type=int, default=10, help="modo grade: guarda a postura a cada N subpassos")
    p.add_argument("--walk", action="store_true", help="conectoma que anda (M4), no corpo de caminhada, sem patins")
    p.add_argument("--speed", type=float, default=3.5)
    p.add_argument("--yaw", type=float, default=0.0, help="giro pedido (rad/s)")
    p.add_argument("--seconds", type=float, default=0.0, help="duração do teste (s; padrão: a do treino)")
    p.add_argument("--silence", default="", help="cala um par de neurônios de comando (ex.: DNg100)")
    p.add_argument("--stim", default="", help="estimula um lado (ex.: DNa02_L)")
    p.add_argument("--stim-scale", type=float, default=1.0, help="múltiplo da corrente do comando de giro")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-gpu-mem-gb", type=float, default=8.0)
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", device.index or 0)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_mem_gb * 2**30 / total), device)
    if args.checkpoint:
        if args.walk:
            trace_walk(args, device)
        elif args.tests > 0:
            trace_grid(args, device)
        else:
            trace_test(args, device)
        return
    if not (args.run and args.attempts):
        p.error("use --run e --attempts (captura) ou --checkpoint (teste)")
    run_dir = RUNS / args.run
    manifest = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    env_dict = dict(manifest["env"])
    env_dict["reward"] = RewardConfig(**env_dict["reward"])
    env_cfg = EnvConfig(**env_dict)
    ctrl = dict(manifest["controller"])
    ctrl["turn_cells"] = tuple(ctrl["turn_cells"])
    run_args = manifest["args"]
    graph_path = Path(run_args["graph"])
    if not graph_path.exists():
        graph_path = MALECNS_DIR / graph_path.name
    graph = Connectome.load(graph_path)
    pol = ConnectomePolicy(graph, ControllerConfig(**ctrl), device=device)
    evolution = "es_dim" in manifest
    beta = 0.0 if evolution else noise_beta(env_cfg.control_dt, run_args["noise_tau"])
    if evolution:
        from mosca.rl.es import PopulationES

        es = PopulationES(pol.es_parameters(), manifest["sigma"], 0.0, run_args["seed"])

    n = env_cfg.n_envs
    by_gen: dict[int, list[int]] = {}
    for a in sorted(int(x) for x in args.attempts.split(",")):
        by_gen.setdefault(a // n, []).append(a)
    out_dir = run_dir / "video"
    out_dir.mkdir(exist_ok=True)
    for gen, attempts in by_gen.items():
        cap_path, pol_path = run_dir / "capture" / f"it{gen:05d}.npz", run_dir / "capture" / f"policy_it{gen:05d}.pt"
        cap = IterationCapture.load(cap_path)
        index = [a - gen * n for a in attempts]
        rates = {i: [] for i in index}
        means = {i: [] for i in index}

        def keep(env, t, r, mean):
            for i in index:
                rates[i].append(r[:, i].float().cpu().numpy().astype(np.float16))
                means[i].append(mean[i].float().cpu().numpy())

        pol.load_state_dict(torch.load(pol_path, map_location=device, weights_only=True))
        population = es.offsets(es.sample(gen, n), device) if evolution else None
        brain = replay_brain(cap, env_cfg, pol, attempt_seed(run_args["seed"], -1 - gen), beta, on_step=keep,
                             population=population, action_noise=not evolution)
        for a, i in zip(attempts, index):
            assert cap.attempts[i] == a, "a numeração das tentativas não confere"
            poses = []
            phys = replay(cap, i, env_cfg, on_substep=lambda env, t, sub: poses.append(env.datas[0].qpos.astype(np.float32)))
            steps = int(cap.steps[i])
            out = out_dir / f"tentativa_{a + 1}.npz"
            meta = {"mode": "captura", "run": args.run, "graph": graph_path.name, "toe_out_deg": float(env_cfg.toe_out_deg),
                    "attempt": a, "video_number": a + 1, "generation": gen, "env": i,
                    "v_cmd": float(cap.v_cmd[i]), "push": float(cap.push[i]), "steps": steps,
                    "control_dt": env_cfg.control_dt, "substeps": len(poses) // max(steps, 1),
                    "physics_max_deviation": phys["max_deviation"], "brain_max_action_diff": float(brain[i]),
                    "capture_sha256": sha256(cap_path), "policy_sha256": sha256(pol_path),
                    "stats": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer, bool)) else v)
                              for k, v in phys.items() if k != "max_deviation"}}
            np.savez_compressed(out, qpos=np.stack(poses), rates=np.stack(rates[i][:steps]),
                                mean_action=np.stack(means[i][:steps]),
                                yaw_cmd=np.array([cap.yaw_at(t)[i] for t in range(steps)], np.float32),
                                body_id=graph.body_id,
                                meta=json.dumps(meta, ensure_ascii=False))
            ok = phys["max_deviation"] == 0.0 and brain[i] < 1e-4
            print(f"tentativa #{a + 1} (geração {gen}, ambiente {i}): {phys['seconds']:.2f} s, {phys['speed']:.2f} cm/s, "
                  f"{'caiu' if phys['fell'] else 'não caiu'} | física: desvio {phys['max_deviation']:.3g} | cérebro: "
                  f"diferença na ação {brain[i]:.3g} -> {'ok' if ok else 'FALHOU'} | {out}", flush=True)


if __name__ == "__main__":
    main()
