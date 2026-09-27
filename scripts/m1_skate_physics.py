"""M1: testes de física dos patins com a mosca inteira.

A precisão do contato é testada no trenó (tests/test_skate_contact.py). Aqui a pergunta é
se, na mosca inteira, os patins rolam em vez de arrastar. Com atrito lateral 100 vezes
maior que o de rolamento, um desalinhamento de frações de grau entre seis patins já dobra
a freada (o patim raspa de lado), então os limites são de "rolando" (freada de 10 a 20
cm/s²) contra "arrastando" (acima de 500 cm/s²), e não de precisão.

Os testes de rolamento usam as patas enrijecidas (ganho dos servos ×50). Com as patas
normais, a mosca "mole" cede e fecha os patins da frente em limpa-neve ao deslizar; isso,
a força lateral e o giro alcançável são registrados como informação.

1. Giro alcançável de cada patim (a marcha programada decide se basta).
2. Postura canônica: todos os patins planos e apontando para a frente.
3. Parada por 0,5 s: quanto as lâminas afundam no chão, e se a simulação fica estável.
4. Força lateral de 1/4 do peso no tórax por 0,3 s (quanto a mosca escorrega de lado).
5. Deslize a partir de 3 cm/s: freada abaixo de 2× atrito ao longo × g.
6. Rampa de 5°: aceleração acima de 70% de g·(sen θ − μ·cos θ).

Uso:
    python scripts/m1_skate_physics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.body.fly import LEGS  # noqa: E402
from mosca.body.ik import LegIK  # noqa: E402
from mosca.body.poses import hold_pose_ctrl, set_ctrl  # noqa: E402
from mosca.body.skates import SkateConfig, build_skater  # noqa: E402
from mosca.body.stance import (  # noqa: E402
    STANCE_CLAW_TIP,
    STANCE_JOINTS,
    STANCE_POS_SCALE,
    skating_stance,
    thorax_rotation,
)

SKATES = SkateConfig()
YAW_LIMIT, YAW_STEP = 60, 5
MAX_FOOT_SHIFT = 0.015  # cm
STIFF_GAIN = 50.0
MIN_LOAD = 0.05  # fração do peso para um patim contar como apoiado
HEADING = np.array([1.0, 0.0, 0.0])
LATERAL = np.array([0.0, 1.0, 0.0])


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'ok' if ok else 'FALHOU'}] {label}: {detail}")
    return ok


def info(label: str, detail: str) -> None:
    print(f"  [info] {label}: {detail}")


def runner_penetration(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    worst = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if model.geom(c.geom1).name.startswith("skate_runner") or model.geom(c.geom2).name.startswith("skate_runner"):
            worst = max(worst, -c.dist)
    return worst


def loaded_lateral_slip(model: mujoco.MjModel, data: mujoco.MjData, weight: float) -> float:
    """Maior velocidade lateral (no referencial do próprio patim) entre os patins com carga."""
    worst = 0.0
    vel = np.zeros(6)
    for leg in LEGS:
        if data.sensor(f"touch_skate_{leg}").data[0] < MIN_LOAD * weight:
            continue
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, model.body(f"skate_{leg}").id, vel, 1)
        worst = max(worst, abs(vel[4]))
    return worst


def skate_yaws(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    out = {}
    for leg in LEGS:
        x = data.xmat[model.body(f"skate_{leg}").id].reshape(3, 3)[:, 0]
        out[leg] = np.degrees(np.arctan2(x[1], x[0]))
    return out


def com_velocity(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    mujoco.mj_subtreeVel(model, data)
    return data.subtree_linvel[model.body("thorax").id].copy()


def stiffen(model: mujoco.MjModel, factor: float) -> None:
    """Multiplica o ganho dos servos de posição (força = k·(alvo − ângulo))."""
    model.actuator_gainprm[:, 0] *= factor
    model.actuator_biasprm[:, 1] *= factor


def settle(model: mujoco.MjModel, qpos: np.ndarray, seconds: float) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    set_ctrl(model, data, hold_pose_ctrl(model, data))
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return data


def glide(model: mujoco.MjModel, rest: mujoco.MjData, seconds: float = 0.15) -> tuple[float, mujoco.MjData]:
    data = mujoco.MjData(model)
    data.qpos[:], data.act[:], data.ctrl[:] = rest.qpos, rest.act, rest.ctrl
    data.qvel[:3] = 3.0 * HEADING
    mujoco.mj_forward(model, data)
    v0 = com_velocity(model, data) @ HEADING
    n = int(seconds / model.opt.timestep)
    for _ in range(n):
        mujoco.mj_step(model, data)
    return (v0 - com_velocity(model, data) @ HEADING) / (n * model.opt.timestep), data


def main() -> None:
    model = build_skater(skates=SKATES)
    ik = LegIK(model)
    thorax = model.body("thorax").id
    weight = model.body_subtreemass[thorax] * 981.0
    dt = model.opt.timestep
    results = []

    print("1. Giro alcançável de cada patim (plano, erro < 3°, pé a < 150 µm da posição mediana):")
    rot = thorax_rotation()
    up = rot.T @ np.array([0.0, 0.0, 1.0])
    for leg in LEGS:
        q_ref = np.asarray(STANCE_JOINTS[leg])
        reach = []
        for direction in (1, -1):
            q0 = q_ref
            for yaw in range(0, direction * (YAW_LIMIT + 1), direction * YAW_STEP):
                forward = rot.T @ np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0.0])
                res = ik.solve(leg, np.asarray(STANCE_CLAW_TIP[leg]), forward, up=up, q0=q0, q_ref=q_ref, pos_scale=STANCE_POS_SCALE)
                if res.angle_err > np.radians(3) or res.pos_err > MAX_FOOT_SHIFT:
                    break
                reach.append(yaw)
                q0 = res.q
        info(leg, f"{min(reach):+d}° a {max(reach):+d}° (positivo = para a esquerda da mosca)" if reach else "nenhum")

    print("2-3. Postura canônica, parada por 0,5 s:")
    stance = skating_stance(model, ik)
    results.append(check("patins planos e para a frente", stance.worst_angle < np.radians(3),
                         f"pior erro {np.degrees(stance.worst_angle):.2f}°, maior desvio de pé {stance.worst_shift * 1e4:.0f} µm"))
    rest = settle(model, stance.qpos, 0.5)
    pen = runner_penetration(model, rest)
    limit = 0.05 * SKATES.runner_radius
    stable = rest.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0 and abs(rest.qpos[2] - stance.qpos[2]) < 0.2 * stance.qpos[2]
    results.append(check("afundamento das lâminas", pen < limit, f"{pen * 1e4:.3f} µm (limite {limit * 1e4:.1f} µm)"))
    results.append(check("estável parada", stable, f"tórax {stance.qpos[2]:.4f} -> {rest.qpos[2]:.4f} cm"))

    decel_limp, limp = glide(model, rest)
    toe = skate_yaws(model, limp)
    info("deslize com patas normais", f"desaceleração {decel_limp:.1f} cm/s²; patins da frente em {toe['T1_left']:+.1f}° / {toe['T1_right']:+.1f}° (limpa-neve)")

    stiffen(model, STIFF_GAIN)
    rest = settle(model, stance.qpos, 0.5)

    print(f"4. Força lateral de 1/4 do peso por 0,3 s (patas ×{STIFF_GAIN:.0f}):")
    data = mujoco.MjData(model)
    data.qpos[:], data.act[:], data.ctrl[:] = rest.qpos, rest.act, rest.ctrl
    mujoco.mj_forward(model, data)
    y0 = data.subtree_com[thorax][1]
    worst_slip = 0.0
    for _ in range(int(0.3 / dt)):
        data.xfrc_applied[thorax, :3] = 0.25 * weight * LATERAL
        mujoco.mj_step(model, data)
        worst_slip = max(worst_slip, loaded_lateral_slip(model, data, weight))
    info("escorregar de lado", f"centro de massa andou {(data.subtree_com[thorax][1] - y0) * 1e4:.0f} µm; "
         f"pico de derrapagem de um patim com carga {worst_slip * 1e4:.0f} µm/s")

    print(f"5. Deslize a partir de 3 cm/s (patas ×{STIFF_GAIN:.0f}):")
    decel, _ = glide(model, rest)
    expected = SKATES.friction_along * 981.0
    results.append(check("patins rolando", decel < 2 * expected, f"freada {decel:.1f} cm/s² (atrito × g = {expected:.2f}; arrastando seria > 500)"))

    print(f"6. Rampa de 5° a partir do repouso (patas ×{STIFF_GAIN:.0f}):")
    data = mujoco.MjData(model)
    data.qpos[:], data.act[:], data.ctrl[:] = rest.qpos, rest.act, rest.ctrl
    theta = np.radians(5)
    model.opt.gravity[:] = 981.0 * (np.sin(theta) * HEADING - np.cos(theta) * np.array([0.0, 0.0, 1.0]))
    n = int(0.15 / dt)
    for _ in range(n):
        mujoco.mj_step(model, data)
    acc = (com_velocity(model, data) @ HEADING) / (n * dt)
    expected = 981.0 * (np.sin(theta) - SKATES.friction_along * np.cos(theta))
    results.append(check("aceleração na rampa", acc > 0.7 * expected, f"{acc:.1f} cm/s² ({acc / expected:.0%} do ideal {expected:.1f})"))
    model.opt.gravity[:] = [0.0, 0.0, -981.0]

    print(f"\n{sum(results)}/{len(results)} checagens passaram")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
