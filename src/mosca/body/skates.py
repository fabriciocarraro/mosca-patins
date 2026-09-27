"""Patins para a mosca.

Em cada pata, as juntas 2 a 5 do tarso saem do modelo e o tarso distal vira uma peça
rígida (a "bota"); a junta `tarsus` continua livre e funciona como tornozelo. Adesão,
tendão do tarso e sensor de toque da garra também saem: patins não grudam.

Modelo "lâmina" (padrão): cada patim tem quatro lâminas curtas (cápsulas) nas posições das
quatro rodas de um patins quad. O atrito é anisotrópico, definido num par de contato
explícito com o chão: baixo ao longo da lâmina, alto de lado. A cápsula é a única forma do
MuJoCo cujo referencial de atrito acompanha o eixo do objeto. Lâminas curtas e separadas
impedem que o patim role de lado como um rolo de massa e evitam o erro de lâminas longas,
que empinam ao frear e passam a tocar só pela ponta. As rodas aparecem só no visual.

O patim é montado centrado sob a ponta da garra, paralelo ao chão da postura típica de
apoio (SKATE_DOWN) e apontando para a frente do corpo nessa postura (SKATE_FORWARD). Nas
patas do meio e de trás o pé aponta para o lado ou para trás, e as juntas não conseguem
girar um patim alinhado ao pé até a frente; por isso a montagem segue o corpo, não o pé.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mosca.body.fly import LEGS, PhysicsConfig, build_fly_spec

# Direção "para baixo" no referencial do tarso, média dos quadros de apoio de moscas reais
# (gerada por scripts/derive_skate_mounts.py).
SKATE_DOWN = {
    "T1_left": (-0.483, 0.744, -0.461),
    "T1_right": (0.555, -0.742, 0.375),
    "T2_left": (0.143, 0.711, -0.689),
    "T2_right": (-0.130, -0.709, 0.693),
    "T3_left": (-0.059, 0.520, -0.852),
    "T3_right": (0.071, -0.508, 0.858),
}
# Frente do corpo (projetada no chão), no referencial do tarso, nos mesmos quadros de apoio.
SKATE_FORWARD = {
    "T1_left": (0.539, 0.664, 0.519),
    "T1_right": (-0.656, -0.664, -0.360),
    "T2_left": (0.940, -0.318, -0.126),
    "T2_right": (-0.946, 0.299, 0.123),
    "T3_left": (0.430, -0.760, -0.488),
    "T3_right": (-0.441, 0.759, 0.479),
}
DISTAL_TARSUS = ("tarsus2", "tarsus3", "tarsus4", "claw")


@dataclass(frozen=True)
class SkateConfig:
    length: float = 0.03  # cm (300 µm), da frente das rodas da frente ao fundo das de trás
    width: float = 0.008  # cm, distância entre as rodas da esquerda e da direita
    runner_radius: float = 0.002  # cm (20 µm)
    runner_length: float = 0.008  # cm (80 µm), comprimento de cada lâmina
    clearance: float = 0.0005  # cm, quanto a base do patim fica abaixo da ponta da garra
    friction_along: float = 0.01
    friction_across: float = 1.0
    solref: tuple[float, float] = (0.0004, 1.0)  # 2 passos de física: o mais rígido que é seguro
    solimp: tuple[float, ...] = (0.95, 0.99, 0.01, 0.5, 2.0)
    mass: float = 5e-6  # g por patim (0,005 mg; uma pata tem ~0,016 mg)
    toe_out_deg: float = 0.0  # ponta para fora, espelhada entre os lados
    impratio: float = 10.0


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def runner_segments(cfg: SkateConfig) -> list[tuple[str, np.ndarray]]:
    """Nome e `fromto` de cada lâmina, no referencial do patim (x para a frente, z para cima)."""
    r, half_l, half_w = cfg.runner_radius, 0.5 * cfg.length, 0.5 * cfg.width
    axis_half = 0.5 * cfg.runner_length - r
    out = []
    for side, y in (("l", half_w), ("r", -half_w)):
        for end, x in (("f", half_l - 0.5 * cfg.runner_length), ("b", -half_l + 0.5 * cfg.runner_length)):
            out.append((f"{side}{end}", np.array([x - axis_half, y, r, x + axis_half, y, r])))
    return out


def _claw_tip(spec: mujoco.MjSpec, leg: str) -> np.ndarray:
    """Ponta da garra no referencial do tarso (juntas distais em zero)."""
    pos = np.zeros(3)
    for segment in DISTAL_TARSUS:
        body = spec.body(f"{segment}_{leg}")
        assert np.allclose(body.quat, [1, 0, 0, 0]), f"{body.name} deveria estar alinhado ao pai"
        pos = pos + np.asarray(body.pos)
    return pos + np.asarray(spec.geom(f"tarsal_claw_{leg}_collision").fromto[3:])


def _mount_frame(spec: mujoco.MjSpec, leg: str, cfg: SkateConfig) -> tuple[np.ndarray, np.ndarray]:
    """Posição e orientação do patim no referencial do tarso: x para a frente, z para cima."""
    tip = _claw_tip(spec, leg)
    down = _normalize(np.asarray(SKATE_DOWN[leg]))
    body_forward = np.asarray(SKATE_FORWARD[leg])
    forward = _normalize(body_forward - np.dot(body_forward, down) * down)
    if cfg.toe_out_deg:
        sign = 1.0 if leg.endswith("left") else -1.0
        angle = np.radians(cfg.toe_out_deg) * sign
        quat = np.array([np.cos(angle / 2), *(np.sin(angle / 2) * -down)])
        rotated = np.zeros(3)
        mujoco.mju_rotVecQuat(rotated, forward, quat)
        forward = _normalize(rotated)
    up = -down
    lateral = np.cross(up, forward)
    rot = np.column_stack([forward, lateral, up])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    center = tip + cfg.clearance * down
    return center, quat


def _strip_foot(spec: mujoco.MjSpec, leg: str) -> None:
    spec.delete(spec.actuator(f"adhere_claw_{leg}"))
    spec.delete(spec.actuator(f"tarsus2_{leg}"))
    spec.delete(spec.tendon(f"tarsus2_{leg}"))
    spec.delete(spec.sensor(f"touch_claw_{leg}"))
    for segment in ("tarsus2", "tarsus3", "tarsus4", "tarsus5"):
        spec.delete(spec.joint(f"{segment}_{leg}"))
    for name in (f"tarsus2_{leg}", f"tarsus3_{leg}", f"tarsus4_{leg}", f"tarsal_claw_{leg}"):
        geom = spec.geom(f"{name}_collision")
        geom.contype = 0
        geom.conaffinity = 0


def _add_skate_materials(spec: mujoco.MjSpec) -> None:
    spec.add_material(name="skate_boot", rgba=[0.85, 0.12, 0.16, 1.0], shininess=0.5, specular=0.4)
    spec.add_material(name="skate_wheel", rgba=[0.97, 0.93, 0.78, 1.0], shininess=0.3)
    spec.add_material(name="skate_runner", rgba=[0.2, 0.6, 1.0, 0.5])


def _add_skate(spec: mujoco.MjSpec, leg: str, cfg: SkateConfig) -> None:
    center, quat = _mount_frame(spec, leg, cfg)
    skate = spec.body(f"tarsus_{leg}").add_body(name=f"skate_{leg}", pos=center, quat=quat)
    r, half_l, half_w = cfg.runner_radius, 0.5 * cfg.length, 0.5 * cfg.width
    segments = runner_segments(cfg)

    for tag, fromto in segments:
        name = f"skate_runner_{tag}_{leg}"
        skate.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[r, 0, 0],
            fromto=fromto.tolist(),
            mass=cfg.mass / len(segments),
            contype=0,
            conaffinity=0,
            group=3,
            material="skate_runner",
        )
        spec.add_pair(
            name=f"floor_{name}",
            geomname1="floor",
            geomname2=name,
            condim=3,
            friction=[cfg.friction_along, cfg.friction_across, 0.005, 0.0001, 0.0001],
            solref=list(cfg.solref),
            solimp=list(cfg.solimp),
        )
        wheel_r = 0.5 * cfg.runner_length
        center = 0.5 * (fromto[:3] + fromto[3:])
        skate.add_geom(
            name=f"skate_wheel_{tag}_{leg}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[wheel_r, 0.3 * r, 0],
            pos=[center[0], center[1], wheel_r],
            quat=[np.sqrt(0.5), np.sqrt(0.5), 0, 0],
            contype=0,
            conaffinity=0,
            density=0,
            group=1,
            material="skate_wheel",
        )

    wheel_r = 0.5 * cfg.runner_length
    boot_h = 1.5 * r
    skate.add_geom(
        name=f"skate_boot_{leg}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[half_l, half_w + r, 0.5 * boot_h],
        pos=[0, 0, 1.8 * wheel_r + 0.5 * boot_h],
        contype=0,
        conaffinity=0,
        density=0,
        group=1,
        material="skate_boot",
    )
    skate.add_site(
        name=f"skate_{leg}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[half_l + r, half_w + 2 * r, 2 * r],
        pos=[0, 0, r],
        group=3,
    )
    spec.add_sensor(name=f"touch_skate_{leg}", type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"skate_{leg}")


def add_skates(spec: mujoco.MjSpec, cfg: SkateConfig = SkateConfig()) -> mujoco.MjSpec:
    spec.option.impratio = cfg.impratio
    _add_skate_materials(spec)
    for leg in LEGS:
        _strip_foot(spec, leg)
        _add_skate(spec, leg, cfg)
    return spec


def build_skater_spec(physics: PhysicsConfig = PhysicsConfig(), skates: SkateConfig = SkateConfig()) -> mujoco.MjSpec:
    return add_skates(build_fly_spec(physics), skates)


def build_skater(physics: PhysicsConfig = PhysicsConfig(), skates: SkateConfig = SkateConfig()) -> mujoco.MjModel:
    return build_skater_spec(physics, skates).compile()
