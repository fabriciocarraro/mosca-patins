"""Ensaio da captura (M5): regenera tentativas sorteadas de uma execução a partir do que foi gravado.

Para cada tentativa sorteada:
- a física é refeita sozinha (lote 1) com as ações gravadas e conferida contra os estados
  gravados a cada 0,5 s (critério: idêntica bit a bit);
- o cérebro é re-simulado com a versão da política usada na coleta, na leva inteira da iteração
  (mesmo lote da coleta), recalculando as ações como na coleta (critério: diferença < 1e-4).

O manifesto é o config.json da execução: argumentos, configurações e versões das bibliotecas. A
física só é garantida bit a bit com as mesmas versões e a mesma arquitetura, e o cérebro, na mesma
máquina da coleta (a rede amplifica diferenças de arredondamento entre CPU e GPU).

Uso (no Spark, onde a execução rodou, depois de um treino com --capture --deterministic):
    python scripts/m5_rehearsal.py --run patina_f --count 3 --device cuda
    python scripts/m5_rehearsal.py --run patina_f --attempts 0,777,12345
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
from mosca.capture import IterationCapture, library_versions, noise_beta, replay, replay_brain  # noqa: E402
from mosca.env.skate_env import EnvConfig, RewardConfig, attempt_seed  # noqa: E402
from mosca.paths import MALECNS_DIR, RUNS  # noqa: E402

BRAIN_TOL = 1e-4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--count", type=int, default=3, help="quantas tentativas sortear")
    p.add_argument("--seed", type=int, default=None, help="semente do sorteio (padrão: aleatória, impressa)")
    p.add_argument("--attempts", default="", help="números das tentativas (a partir de 0), em vez do sorteio")
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
    ctrl_cfg = ControllerConfig(**ctrl)
    run_args = manifest["args"]

    now = library_versions()
    changed = {k: (v, now.get(k)) for k, v in manifest["versions"].items() if now.get(k) != v}
    print(f"execução {args.run}: {manifest['neurons']:,} neurônios, controle a {env_cfg.control_dt * 1e3:g} ms, "
          f"controlador {'determinístico' if ctrl_cfg.deterministic else 'NÃO determinístico'}")
    print("versões iguais às da coleta" if not changed else f"versões diferentes da coleta: {changed}")

    n = env_cfg.n_envs
    iterations = sorted(int(f.stem[2:]) for f in (run_dir / "capture").glob("it[0-9]*.npz"))
    if not iterations:
        sys.exit("nenhuma iteração gravada (o treino rodou com --capture?)")
    if args.attempts:
        chosen = [int(a) for a in args.attempts.split(",")]
    else:
        seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(2**31))
        pool = np.array([it * n + i for it in iterations for i in range(n)])
        chosen = sorted(int(a) for a in np.random.default_rng(seed).choice(pool, size=args.count, replace=False))
        print(f"sorteio com a semente {seed}: tentativas {[a + 1 for a in chosen]} (numeradas a partir de 1)")

    graph_path = Path(run_args["graph"])
    if not graph_path.exists():
        graph_path = MALECNS_DIR / graph_path.name
    pol = ConnectomePolicy(Connectome.load(graph_path), ctrl_cfg, device=device)
    beta = noise_beta(env_cfg.control_dt, run_args["noise_tau"])

    report, brain_cache = [], {}
    for attempt in chosen:
        it, index = divmod(attempt, n)
        cap_path = run_dir / "capture" / f"it{it:05d}.npz"
        pol_path = run_dir / "capture" / f"policy_it{it:05d}.pt"
        cap = IterationCapture.load(cap_path)
        assert cap.attempts[index] == attempt, "a numeração das tentativas não confere"
        phys = replay(cap, index, env_cfg)
        if it not in brain_cache:
            pol.load_state_dict(torch.load(pol_path, map_location=device, weights_only=True))
            brain_cache[it] = replay_brain(cap, env_cfg, pol, attempt_seed(run_args["seed"], -1 - it), beta)
        brain = float(brain_cache[it][index])
        ok = phys["max_deviation"] == 0.0 and brain < BRAIN_TOL
        report.append({"attempt": attempt, "iteration": it, "env": index, "physics_max_deviation": phys["max_deviation"],
                       "brain_max_action_diff": brain, "batch_brain_max_action_diff": float(brain_cache[it].max()),
                       "seconds": phys["seconds"], "fell": bool(phys["fell"]), "speed": phys["speed"],
                       "capture_sha256": sha256(cap_path), "policy_sha256": sha256(pol_path), "ok": bool(ok)})
        print(f"tentativa #{attempt + 1} (iteração {it}, ambiente {index}): {phys['seconds']:.2f} s, "
              f"{'caiu' if phys['fell'] else 'não caiu'}, {phys['speed']:.2f} cm/s | física: desvio "
              f"{phys['max_deviation']:.3g} | cérebro: maior diferença na ação {brain:.3g} "
              f"(leva inteira {brain_cache[it].max():.3g}) -> {'ok' if ok else 'FALHOU'}")
    passed = all(r["ok"] for r in report)
    out = run_dir / "m5_rehearsal.json"
    out.write_text(json.dumps({"run": args.run, "versions_now": now, "versions_changed": changed,
                               "brain_tolerance": BRAIN_TOL, "attempts": report, "passed": passed}, indent=2),
                   encoding="utf-8")
    print(f"{'M5 ok' if passed else 'M5 FALHOU'}: física bit a bit e cérebro com diferença < {BRAIN_TOL:g} "
          f"em {sum(r['ok'] for r in report)}/{len(report)} tentativas; relatório em {out}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
