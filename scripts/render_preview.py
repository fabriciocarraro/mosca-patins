"""Renderiza uma prévia da mosca numa pose real, depois de segurar a pose por um tempo.

Uso:
    python scripts/render_preview.py --snippet 0 --hold 0.5 --out outputs/preview.png
    python scripts/render_preview.py --skates --camera side --show-runners
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.body.fly import build_fly  # noqa: E402
from mosca.body.poses import apply_frame, hold_pose_ctrl, load_walking_frame, set_ctrl, settle_on_floor  # noqa: E402
from mosca.body.skates import build_skater  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snippet", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--hold", type=float, default=0.5, help="segundos simulados segurando a pose")
    parser.add_argument("--skates", action="store_true")
    parser.add_argument("--show-runners", action="store_true", help="mostra as lâminas de contato (grupo 3)")
    parser.add_argument("--camera", default="hero")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out", default="outputs/preview.png")
    args = parser.parse_args()

    t0 = time.perf_counter()
    model = build_skater() if args.skates else build_fly()
    print(f"modelo: nq={model.nq} nv={model.nv} nu={model.nu} ngeom={model.ngeom} ({time.perf_counter() - t0:.1f} s)")
    print(f"massa total: {model.body_subtreemass[model.body('thorax').id] * 1e3:.3f} mg")

    data = mujoco.MjData(model)
    apply_frame(model, data, load_walking_frame(args.snippet, args.frame))
    shift = settle_on_floor(model, data)
    set_ctrl(model, data, hold_pose_ctrl(model, data))
    start_height = data.qpos[2]
    for _ in range(int(args.hold / model.opt.timestep)):
        mujoco.mj_step(model, data)
    unstable = data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
    print(f"ajuste inicial {shift * 1e4:+.1f} µm; tórax {start_height:.4f} -> {data.qpos[2]:.4f} cm; "
          f"deslocamento horizontal {abs(data.qpos[0]) * 1e4:.1f}/{abs(data.qpos[1]) * 1e4:.1f} µm; "
          f"contatos {data.ncon}; instável={unstable}")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    options = mujoco.MjvOption()
    options.geomgroup[3] = args.show_runners
    renderer.update_scene(data, camera=args.camera, scene_option=options)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(renderer.render()).save(out)
    print(f"imagem: {out}")


if __name__ == "__main__":
    main()
