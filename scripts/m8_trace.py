"""M8: re-simula tentativas gravadas e guarda o que o vídeo precisa (roda onde a execução rodou).

Para cada tentativa ("Tentativa #N", numerada a partir de 1 no vídeo e de 0 aqui):
- física refeita sozinha com as ações gravadas, conferida contra os estados gravados (desvio 0 = bit a bit), com a
  postura (qpos) guardada a cada subpasso de física (0,2 ms), para a câmera lenta de verdade;
- cérebro re-simulado na leva inteira da geração (como no m5_rehearsal), com a atividade de todos os neurônios da
  tentativa guardada a cada passo de controle, e a maior diferença entre a ação recalculada e a gravada.

Renderizar não exige a mesma máquina: o `m8_render.py` só desenha as posturas guardadas.

Saída: runs/<execução>/video/tentativa_<N+1>.npz.

Uso (no Spark):
    python scripts/m8_trace.py --run final_s0 --attempts 0,43903 --device cuda
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


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--attempts", required=True, help="números das tentativas, a partir de 0, separados por vírgula")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-gpu-mem-gb", type=float, default=8.0)
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", device.index or 0)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_mem_gb * 2**30 / total), device)
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
            meta = {"run": args.run, "graph": graph_path.name, "attempt": a, "video_number": a + 1, "generation": gen, "env": i,
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
