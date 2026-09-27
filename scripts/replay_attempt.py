"""Re-simula uma tentativa gravada pelo número ("Tentativa #N") e, se pedido, grava um GIF.

A tentativa N (contada a partir de 0) está na iteração N // ambientes, posição
N % ambientes. A física é refeita com as ações gravadas e conferida contra os estados
gravados a cada 0,5 s (desvio máximo 0 = bit a bit).

Uso:
    python scripts/replay_attempt.py --run conectoma_a --attempt 1234
    python scripts/replay_attempt.py --run conectoma_a --attempt 1234 --gif outputs/tentativa_1235.gif --slow 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.capture import IterationCapture, replay  # noqa: E402
from mosca.env.skate_env import EnvConfig, RewardConfig  # noqa: E402
from mosca.paths import RUNS  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--attempt", type=int, required=True, help="número da tentativa, a partir de 0")
    parser.add_argument("--gif", default="")
    parser.add_argument("--slow", type=float, default=4.0, help="câmera lenta do GIF")
    args = parser.parse_args()

    run_dir = RUNS / args.run
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    env_dict = dict(config["env"])
    env_dict["reward"] = RewardConfig(**env_dict["reward"])
    env_cfg = EnvConfig(**env_dict)
    n = env_cfg.n_envs
    it, index = divmod(args.attempt, n)
    cap = IterationCapture.load(run_dir / "capture" / f"it{it:05d}.npz")
    assert cap.attempts[index] == args.attempt

    frames, render = [], None
    if args.gif:
        import mujoco
        from PIL import Image, ImageDraw

        state = {}

        def render(env, t):
            if "renderer" not in state:
                state["renderer"] = mujoco.Renderer(env.model, height=360, width=640)
                cam = mujoco.MjvCamera()
                cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, env.thorax
                cam.distance, cam.elevation, cam.azimuth = 0.75, -18, 150
                state["cam"] = cam
            every = max(1, round(1.0 / (25 * env.cfg.control_dt * args.slow)))  # GIF a 25 quadros/s
            if t % every:
                return
            state["renderer"].update_scene(env.datas[0], camera=state["cam"])
            img = Image.fromarray(state["renderer"].render())
            ImageDraw.Draw(img).text((10, 8), f"Tentativa #{args.attempt + 1}  t = {(t + 1) * env.cfg.control_dt:.2f} s  "
                                              f"(câmera lenta {args.slow:g}x)  {args.run}", fill=(20, 20, 20))
            frames.append(img.quantize(colors=128))

    res = replay(cap, index, env_cfg, on_step=render)
    print(f"tentativa #{args.attempt + 1} (iteração {it}, ambiente {index}): {res['seconds']:.2f} s, "
          f"{'caiu' if res['fell'] else 'não caiu'}, {res['speed']:.2f} cm/s, deslizando {res['glide_frac']:.0%}; "
          f"desvio máximo nos estados gravados: {res['max_deviation']:.3g}")
    if args.gif and frames:
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=40, loop=0, optimize=True)
        print(f"GIF: {args.gif} ({len(frames)} quadros)")
    if not np.isfinite(res["max_deviation"]) or res["max_deviation"] > 0:
        sys.exit("a re-simulação divergiu dos estados gravados")


if __name__ == "__main__":
    main()
