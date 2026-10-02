"""M7: avalia os checkpoints de uma execução nos testes fixos registrados (docs/m7_controles.md).

Serve para os três braços: conectoma real e embaralhado (checkpoints do m6_es.py) e MLP (do m2_train.py).
Em cada checkpoint, com a ação média da política (sem ruído), a partir do repouso:

- critério do M2: as 20 tentativas de teste pedindo 3,5 cm/s (≥80% com média ≥3 cm/s em 5 s, ≥25% do tempo
  deslizando, <10% de quedas);
- curvas: as mesmas 20 tentativas pedindo 3 cm/s, metade com giro de +0,5 rad/s e metade de −0,5; registra o
  giro médio de cada lado, a diferença entre eles (ideal 1,0 rad/s) e o acerto (sem cair, até o fim, erro médio
  de giro abaixo da metade do pedido, 0,25 rad/s, e velocidade média a menos de 25% da pedida; uma mosca que não
  gira fica com erro ~0,5);
- parada: as mesmas 20 tentativas pedindo 0 cm/s (sem cair e com |velocidade| <0,5 cm/s).

As 60 tentativas rodam num lote só, com as mesmas sementes de teste do M2 (10⁹ + 0…19) em cada bloco.

Uso:
    python scripts/m7_eval.py m7p_real_s0 --every 50 --device cuda
    python scripts/m7_eval.py m7p_mlp_s0 --every 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.paths import RUNS  # noqa: E402

TEST_ATTEMPT_BASE = 10**9  # os mesmos testes do M2
TESTS, SPEED, MIN_SPEED, TURN_SPEED, TURN_YAW = 20, 3.5, 3.0, 3.0, 0.5


def commands() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tentativas, velocidade e giro pedidos dos três blocos (M2, curvas, parada)."""
    k = np.arange(TESTS)
    attempts = TEST_ATTEMPT_BASE + np.tile(k, 3)
    speed = np.concatenate([np.full(TESTS, SPEED), np.full(TESTS, TURN_SPEED), np.zeros(TESTS)])
    yaw = np.concatenate([np.zeros(TESTS), np.where(k % 2 == 0, TURN_YAW, -TURN_YAW), np.zeros(TESTS)])
    return attempts, speed, yaw


def run_connectome(path: Path, device, threads: int) -> tuple[list[dict], float]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from m6_eval import load, make_env

    pol, _, ckpt, _ = load(str(path), device)
    attempts, speed, yaw = commands()
    env = make_env(ckpt, len(attempts), threads)
    obs, _ = env.reset(attempts, speed, yaw_cmd=yaw)
    v = torch.as_tensor(speed, dtype=torch.float32, device=device)
    r = pol.initial_state(len(attempts))
    with torch.no_grad():
        while env.alive.any():
            r = pol.step_net(r, pol.currents(torch.as_tensor(obs, dtype=torch.float32, device=device), v))
            obs, *_ = env.step(pol.decode(r).cpu().numpy())
    eps = env.episode_stats()
    horizon = env.cfg.episode_seconds
    env.close()
    return eps, horizon


def run_mlp(path: Path, threads: int) -> tuple[list[dict], float]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from m2_eval import load, policy

    attempts, speed, yaw = commands()
    env, ac, norm, _ = load(str(path), len(attempts), threads)
    obs, _ = env.reset(attempts, speed, yaw_cmd=yaw)
    while env.alive.any():
        obs, *_ = env.step(policy(ac, norm, obs))
    eps = env.episode_stats()
    horizon = env.cfg.episode_seconds
    env.close()
    return eps, horizon


def summarize(eps: list[dict], seconds: float) -> dict:
    m2, turn, stand = eps[:TESTS], eps[TESTS:2 * TESTS], eps[2 * TESTS:]
    done = [not e["fell"] and e["seconds"] >= seconds - 1e-9 for e in eps]
    fast = float(np.mean([d and e["speed"] >= MIN_SPEED for d, e in zip(done[:TESTS], m2)]))
    glide = float(np.mean([e["glide_frac"] for e in m2]))
    falls = float(np.mean([e["fell"] for e in m2]))
    left = [e["yaw_rate"] for e in turn if e["yaw_cmd"] > 0]
    right = [e["yaw_rate"] for e in turn if e["yaw_cmd"] < 0]
    turn_ok = float(np.mean([d and abs(e["speed"] - TURN_SPEED) < 0.25 * TURN_SPEED
                             and e["yaw_error"] < 0.5 * TURN_YAW for d, e in zip(done[TESTS:2 * TESTS], turn)]))
    return {"m2_fast": fast, "m2_glide": glide, "m2_falls": falls, "m2_ok": fast >= 0.8 and glide >= 0.25 and falls < 0.1,
            "m2_speed": float(np.mean([e["speed"] for e in m2])),
            "turn_left": float(np.mean(left)), "turn_right": float(np.mean(right)),
            "turn_range": float(np.mean(left) - np.mean(right)), "turn_ok": turn_ok,
            "turn_falls": float(np.mean([e["fell"] for e in turn])),
            "stand_ok": float(np.mean([not e["fell"] and abs(e["speed"]) < 0.5 for e in stand]))}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--every", type=int, default=50, help="avalia os checkpoints múltiplos deste número")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-gpu-mem-gb", type=float, default=6.0)
    args = p.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, args.max_gpu_mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))

    run_dir = RUNS / args.run
    out = run_dir / "m7_eval.json"
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    ckpts = []
    for f in sorted((run_dir / "checkpoints").glob("*.pt")):
        m = re.fullmatch(r"(?:gen|it)(\d+)\.pt", f.name)
        if m and int(m.group(1)) % args.every == 0 and str(int(m.group(1))) not in results:
            ckpts.append((int(m.group(1)), f))
    print(f"{args.run}: {len(ckpts)} checkpoints a avaliar ({len(results)} já avaliados)")
    print("  ger.   M2   ≥3cm/s  vel  desliza quedas | giro +0,5  −0,5  diferença acerto | parada")
    for number, f in ckpts:
        mlp = "ac" in torch.load(f, weights_only=False, map_location="cpu")
        eps, seconds = run_mlp(f, args.threads) if mlp else run_connectome(f, device, args.threads)
        s = summarize(eps, seconds)
        results[str(number)] = s
        out.write_text(json.dumps(dict(sorted(results.items(), key=lambda kv: int(kv[0]))), indent=1), encoding="utf-8")
        print(f"  {number:5d}  {'OK ' if s['m2_ok'] else 'não'}  {s['m2_fast']:5.0%}  {s['m2_speed']:5.2f}  {s['m2_glide']:5.0%}"
              f"  {s['m2_falls']:5.0%} |  {s['turn_left']:+.2f}  {s['turn_right']:+.2f}  {s['turn_range']:+.2f}"
              f"  {s['turn_ok']:5.0%} | {s['stand_ok']:5.0%}", flush=True)


if __name__ == "__main__":
    main()
