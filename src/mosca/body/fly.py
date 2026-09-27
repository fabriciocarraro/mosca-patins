"""Modelo base da mosca (flybody), montado com MjSpec, sem dm_control.

Reproduz o modo de andar do flybody (física a 0,2 ms, filtro de 10 ms nos atuadores das
juntas, chão com atrito 0,5, garras com atrito 1,0, sem colisão entre patas e asas).
Diferença deliberada: asas, boca, antenas e halteres ficam rígidos. As juntas deles saem
do modelo e a pose de repouso das molas (a mesma que o flybody usa para recolher as asas)
fica embutida na orientação do corpo. São 20 graus de liberdade passivos a menos.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mosca.paths import FLYBODY_XML

LEGS = ("T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right")
LEG_SEGMENTS = ("coxa", "femur", "tibia", "tarsus", "tarsus2", "tarsus3", "tarsus4", "claw")
RIGID_BODIES = (
    "wing_left", "wing_right",
    "rostrum", "haustellum", "labrum_left", "labrum_right",
    "antenna_left", "antenna_right",
    "haltere_left", "haltere_right",
)
SPAWN_HEIGHT = 0.1278  # cm, altura de pé usada pelo flybody


@dataclass(frozen=True)
class PhysicsConfig:
    timestep: float = 2e-4  # s
    integrator: str = "euler"  # "euler" ou "implicitfast"
    impratio: float = 1.0
    noslip_iterations: int = 3
    joint_filter: float = 0.01  # s, passa-baixa dos atuadores de junta (0 desliga)
    adhesion_filter: float = 0.007  # s, passa-baixa dos atuadores de adesão (0 desliga)
    floor_friction: float = 0.5
    floor_solref: tuple[float, float] = (0.001, 1.0)
    floor_solimp: tuple[float, ...] = (0.95, 0.99, 0.01, 0.5, 2.0)
    claw_friction: float = 1.0


_INTEGRATORS = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return out


def _rigidify(spec: mujoco.MjSpec, body: mujoco.MjsBody) -> None:
    """Remove as juntas do corpo, embutindo a pose de repouso das molas na orientação dele."""
    quat = np.asarray(body.quat, dtype=float)
    for joint in list(body.joints):
        quat = _quat_mul(quat, _axis_angle_quat(joint.axis, joint.springref))
        actuator = spec.actuator(joint.name)
        if actuator is not None:
            spec.delete(actuator)
        spec.delete(joint)
    body.quat = quat / np.linalg.norm(quat)


def _add_floor(spec: mujoco.MjSpec, cfg: PhysicsConfig) -> None:
    spec.add_texture(
        name="grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.83, 0.84, 0.86],
        rgb2=[0.74, 0.76, 0.79],
        mark=mujoco.mjtMark.mjMARK_EDGE,
        markrgb=[0.62, 0.64, 0.67],
        width=512,
        height=512,
    )
    material = spec.add_material(name="grid", texrepeat=[40, 40], reflectance=0.08)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[20.0, 20.0, 0.1],
        material="grid",
        friction=[cfg.floor_friction, 0.005, 0.0001],
        solref=list(cfg.floor_solref),
        solimp=list(cfg.floor_solimp),
        condim=3,
    )
    spec.worldbody.add_light(
        name="sun", pos=[0, 0, 3], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=0
    )


def build_fly_spec(cfg: PhysicsConfig = PhysicsConfig()) -> mujoco.MjSpec:
    """Mosca de pé no chão, pronta para andar (ainda sem patins)."""
    spec = mujoco.MjSpec.from_file(str(FLYBODY_XML))
    spec.modelname = "mosca"

    opt = spec.option
    opt.timestep = cfg.timestep
    opt.integrator = _INTEGRATORS[cfg.integrator]
    opt.impratio = cfg.impratio
    opt.noslip_iterations = cfg.noslip_iterations

    # Boca e partes passivas: sem adesão do lábio, juntas removidas.
    for name in ("adhere_labrum_left", "adhere_labrum_right"):
        spec.delete(spec.actuator(name))
    for name in RIGID_BODIES:
        _rigidify(spec, spec.body(name))

    # Filtros de primeira ordem nos atuadores, como no modo de andar do flybody.
    for actuator in spec.actuators:
        is_adhesion = actuator.name.startswith("adhere")
        tau = cfg.adhesion_filter if is_adhesion else cfg.joint_filter
        if tau > 0:
            actuator.dyntype = mujoco.mjtDyn.mjDYN_FILTER
            actuator.dynprm[0] = tau

    for leg in LEGS:
        spec.geom(f"tarsal_claw_{leg}_collision").friction[0] = cfg.claw_friction
        for segment in LEG_SEGMENTS:
            for wing in ("wing_left", "wing_right"):
                spec.add_exclude(bodyname1=f"{segment}_{leg}", bodyname2=wing)

    _add_floor(spec, cfg)
    return spec


def build_fly(cfg: PhysicsConfig = PhysicsConfig()) -> mujoco.MjModel:
    return build_fly_spec(cfg).compile()
