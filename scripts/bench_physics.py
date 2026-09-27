"""M0: velocidade da física e determinismo do replay.

Mede passos de física por segundo (1 thread), passos de controle por segundo com vários
ambientes em threads (um MjData por ambiente, `mj_step` com o GIL liberado), e confere que
rodar duas vezes a mesma sequência de controles dá trajetórias idênticas bit a bit.

Uso:
    python scripts/bench_physics.py --seconds 1 --threads 1 4 8 16
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.body.fly import build_fly  # noqa: E402
from mosca.body.poses import apply_frame, hold_pose_ctrl, load_walking_frame, set_ctrl  # noqa: E402

CONTROL_DT = 2e-3  # s, controle a 500 Hz


def make_data(model: mujoco.MjModel, base_ctrl: np.ndarray) -> mujoco.MjData:
    data = mujoco.MjData(model)
    apply_frame(model, data, load_walking_frame(0, 0))
    set_ctrl(model, data, base_ctrl)
    return data


def wiggle(base_ctrl: np.ndarray, step: int, phase: float = 0.0) -> np.ndarray:
    """Controles determinísticos que mexem as patas, para não medir uma mosca parada."""
    t = step * CONTROL_DT
    return base_ctrl + 0.15 * np.sin(2 * np.pi * 8.0 * t + phase + np.arange(base_ctrl.size))


def run(model, data, base_ctrl, n_control, substeps, record=False):
    traj = []
    for k in range(n_control):
        data.ctrl[:] = np.clip(wiggle(base_ctrl, k), *model.actuator_ctrlrange.T)
        mujoco.mj_step(model, data, nstep=substeps)
        if record:
            traj.append(data.qpos.copy())
    return np.array(traj) if record else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=1.0, help="tempo simulado por medição")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--replay-seconds", type=float, default=5.0)
    args = parser.parse_args()

    model = build_fly()
    substeps = round(CONTROL_DT / model.opt.timestep)
    probe = mujoco.MjData(model)
    apply_frame(model, probe, load_walking_frame(0, 0))
    base_ctrl = hold_pose_ctrl(model, probe)
    n_control = round(args.seconds / CONTROL_DT)
    print(f"{platform.processor() or platform.machine()} | mujoco {mujoco.__version__} | "
          f"física {model.opt.timestep * 1e3:.1f} ms, {substeps} subpassos por controle")

    data = make_data(model, base_ctrl)
    t0 = time.perf_counter()
    run(model, data, base_ctrl, n_control, substeps)
    dt = time.perf_counter() - t0
    print(f"1 thread: {n_control * substeps / dt:,.0f} passos de física/s = "
          f"{n_control / dt:,.0f} passos de controle/s ({args.seconds / dt:.2f}x tempo real)")

    for n_threads in args.threads:
        n_envs = 2 * n_threads
        datas = [make_data(model, base_ctrl) for _ in range(n_envs)]
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            t0 = time.perf_counter()
            list(pool.map(lambda d: run(model, d, base_ctrl, n_control, substeps), datas))
            dt = time.perf_counter() - t0
        print(f"{n_threads:2d} threads, {n_envs:2d} ambientes: {n_envs * n_control / dt:,.0f} passos de controle/s")

    n_replay = round(args.replay_seconds / CONTROL_DT)
    a = run(model, make_data(model, base_ctrl), base_ctrl, n_replay, substeps, record=True)
    b = run(model, make_data(model, base_ctrl), base_ctrl, n_replay, substeps, record=True)
    identical = np.array_equal(a, b)
    print(f"replay de {args.replay_seconds:.0f} s: {'idêntico bit a bit' if identical else 'DIVERGIU'} "
          f"(diferença máxima {np.max(np.abs(a - b)):.3g})")
    sys.exit(0 if identical else 1)


if __name__ == "__main__":
    main()
