"""M1: procura uma marcha programada que faça a mosca andar de patins (≥ 1 cm/s).

Não há rede neural: a marcha é periódica e fixa (src/mosca/body/gait.py), com 16
parâmetros (período e, para cada par de patas, giro do patim, varredura lateral, altura do
pé, fração de apoio e fase). O CMA-ES procura os parâmetros que maximizam a velocidade
média para a frente. Serve para responder se a física dos patins permite propulsão.

"Rolamento" mede se os patins apoiados deslizam junto com o corpo (1) ou ficam plantados
enquanto o corpo passa por cima (0, que é andar de patins em passinhos, não patinar). Com
`--require-rolling`, a nota é velocidade × min(1, rolamento / 0,7).

Uso:
    python scripts/m1_gait_search.py --generations 15 --workers 8 --require-rolling
    python scripts/m1_gait_search.py --play outputs/m1_best_gait.json --gif outputs/m1_best_gait.gif
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.body.fly import LEGS as _LEGS  # noqa: E402
from mosca.body.gait import CONTROL_DT, Gait, PairStroke, cycle_controls, plan_cycle  # noqa: E402
from mosca.body.ik import LegIK  # noqa: E402
from mosca.body.poses import hold_pose_ctrl, set_ctrl  # noqa: E402
from mosca.body.skates import build_skater  # noqa: E402
from mosca.body.stance import skating_stance  # noqa: E402

# (mínimo, máximo) de cada parâmetro; o CMA-ES trabalha no intervalo [0, 1].
PERIOD = (0.05, 0.25)
PAIR_BOUNDS = {
    "yaw_deg": (-15.0, 15.0),
    "sweep": (-0.03, 0.03),
    "lift": (0.003, 0.03),
    "duty": (0.3, 0.8),
    "phase": (0.0, 1.0),
}
WARMUP, MEASURE = 0.3, 1.0  # s
MIN_LOAD = 0.02  # fração do peso para um patim contar como apoiado
ROLLING_TARGET = 0.7

_WORLD = {}


def decode(x: np.ndarray) -> Gait:
    x = np.clip(x, 0.0, 1.0)

    def scale(v, lo_hi):
        return lo_hi[0] + v * (lo_hi[1] - lo_hi[0])

    pairs, i = [], 1
    for _ in range(3):
        kw = {}
        for name, bounds in PAIR_BOUNDS.items():
            kw[name] = float(scale(x[i], bounds))
            i += 1
        pairs.append(PairStroke(**kw))
    return Gait(float(scale(x[0], PERIOD)), *pairs)


def _init_worker(servo_gain: float = 1.0) -> None:
    model = build_skater()
    if servo_gain != 1.0:  # força = k·(alvo − ângulo): servos mais fortes deixam a pata menos flexível
        model.actuator_gainprm[:, 0] *= servo_gain
        model.actuator_biasprm[:, 1] *= servo_gain
    ik = LegIK(model)
    _WORLD.update(model=model, ik=ik, stance=skating_stance(model, ik))


def simulate(gait: Gait, seconds: float, frames_every: int = 0, renderer=None) -> dict:
    model, ik, stance = _WORLD["model"], _WORLD["ik"], _WORLD["stance"]
    targets, worst = plan_cycle(model, ik, stance, gait)
    ctrl = cycle_controls(model, targets)
    data = mujoco.MjData(model)
    data.qpos[:] = stance.qpos
    set_ctrl(model, data, hold_pose_ctrl(model, data))
    substeps = round(CONTROL_DT / model.opt.timestep)
    thorax = model.body("thorax").id
    weight = model.body_subtreemass[thorax] * 981.0
    skates = [(model.sensor(f"touch_skate_{leg}").id, model.body(f"skate_{leg}").id) for leg in _LEGS]
    frames = []
    n_warm = round(WARMUP / CONTROL_DT)
    for k in range(n_warm):
        mujoco.mj_step(model, data, nstep=substeps)
    x0, lowest = data.subtree_com[thorax].copy(), data.qpos[2]
    rolled = moved = 0.0
    vel = np.zeros(6)
    n = round(seconds / CONTROL_DT)
    for k in range(n):
        data.ctrl[:] = ctrl[k % len(ctrl)]
        mujoco.mj_step(model, data, nstep=substeps)
        lowest = min(lowest, data.qpos[2])
        mujoco.mj_subtreeVel(model, data)
        body_speed = np.linalg.norm(data.subtree_linvel[thorax][:2])
        for sensor, body in skates:
            if data.sensordata[model.sensor_adr[sensor]] > MIN_LOAD * weight:
                # XBODY = referencial do patim (x ao longo dele); mjOBJ_BODY seria o de inércia.
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, vel, 1)
                rolled += min(abs(vel[3]), 1.2 * body_speed)
                moved += body_speed
        if renderer is not None and k % frames_every == 0:
            renderer.update_scene(data, camera="track1")
            frames.append(renderer.render())
    disp = data.subtree_com[thorax] - x0
    unstable = data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
    return {
        "speed": float(disp[0] / seconds),
        "drift": float(abs(disp[1]) / seconds),
        "rolling": float(rolled / moved) if moved > 0 else 0.0,
        "lowest": float(lowest),
        "ik_err_deg": float(np.degrees(worst)),
        "unstable": bool(unstable),
        "frames": frames,
    }


def score(r: dict, require_rolling: bool) -> float:
    if r["unstable"] or r["lowest"] < 0.08:
        return -10.0
    s = r["speed"] - 0.5 * r["drift"]
    if require_rolling:
        s *= min(1.0, r["rolling"] / ROLLING_TARGET)
    return s


def fitness(args: tuple[np.ndarray, bool]) -> float:
    x, require_rolling = args
    try:
        return -score(simulate(decode(x), MEASURE), require_rolling)  # CMA-ES minimiza
    except Exception:  # noqa: BLE001 — marcha inviável conta como a pior
        return 10.0


def search(generations: int, workers: int, seed: int, out: Path, require_rolling: bool, servo_gain: float) -> None:
    import cma

    x0 = np.full(1 + 3 * len(PAIR_BOUNDS), 0.5)
    es = cma.CMAEvolutionStrategy(x0, 0.3, {"bounds": [0, 1], "seed": seed, "popsize": 16, "verbose": -9})
    best_x, best_f = None, np.inf
    t0 = time.perf_counter()
    with Pool(workers, initializer=_init_worker, initargs=(servo_gain,)) as pool:
        for gen in range(generations):
            xs = es.ask()
            fs = pool.map(fitness, [(x, require_rolling) for x in xs])
            es.tell(xs, fs)
            i = int(np.argmin(fs))
            if fs[i] < best_f:
                best_x, best_f = np.array(xs[i]), fs[i]
            print(f"geração {gen + 1:2d}: melhor nota {-min(fs):5.2f} | até agora {-best_f:5.2f} ({time.perf_counter() - t0:.0f} s)")
    gait = decode(best_x)
    _init_worker(servo_gain)
    r = simulate(gait, MEASURE)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: r[k] for k in ("speed", "drift", "rolling", "lowest", "ik_err_deg", "unstable")}
    out.write_text(json.dumps({"score": -best_f, "require_rolling": require_rolling, "servo_gain": servo_gain,
                               **summary, "gait": gait.to_dict()}, indent=2), encoding="utf-8")
    print(f"melhor marcha: {out} | velocidade {r['speed']:.2f} cm/s, rolamento {r['rolling']:.2f}")


def play(path: Path, seconds: float, gif: str) -> None:
    saved = json.loads(path.read_text(encoding="utf-8"))
    _init_worker(saved.get("servo_gain", 1.0))
    gait = Gait.from_dict(saved["gait"])
    renderer = mujoco.Renderer(_WORLD["model"], height=360, width=640) if gif else None
    r = simulate(gait, seconds, frames_every=8, renderer=renderer)
    print(f"velocidade {r['speed']:.2f} cm/s ({r['speed'] / 0.297:.1f} corpos/s), deriva lateral {r['drift']:.2f} cm/s, "
          f"rolamento {r['rolling']:.2f}, tórax mínimo {r['lowest']:.4f} cm, erro da IK no apoio {r['ik_err_deg']:.1f}°, "
          f"instável={r['unstable']}")
    if gif:
        from PIL import Image

        images = [Image.fromarray(f) for f in r["frames"]]
        Path(gif).parent.mkdir(parents=True, exist_ok=True)
        images[0].save(gif, save_all=True, append_images=images[1:], duration=40, loop=0)
        print(f"GIF: {gif} ({len(images)} quadros, ~2,5× mais lento que o real)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="outputs/m1_best_gait.json")
    parser.add_argument("--play", default="")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--gif", default="")
    parser.add_argument("--require-rolling", action="store_true", help="só conta velocidade com os patins rolando")
    parser.add_argument("--servo-gain", type=float, default=1.0, help="multiplica a força dos servos das patas")
    args = parser.parse_args()
    if args.play:
        play(Path(args.play), args.seconds, args.gif)
    else:
        search(args.generations, args.workers, args.seed, Path(args.out), args.require_rolling, args.servo_gain)


if __name__ == "__main__":
    main()
