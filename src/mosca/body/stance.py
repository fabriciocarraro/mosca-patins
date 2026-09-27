"""Postura canônica "parada de patins".

O tórax fica na altura e na inclinação medianas das moscas reais em apoio, e cada patim
fica plano e apontando para a frente, com o pé o mais perto possível da posição mediana de
apoio. Orientação é restrição forte; posição é preferência (a cinemática inversa pode
mover o pé algumas dezenas de µm para achatar o patim). Os valores medianos vêm de
scripts/derive_skate_mounts.py, sobre os 100 trechos da amostra de caminhada.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mosca.body.fly import LEGS
from mosca.body.ik import LegIK
from mosca.body.poses import settle_on_floor

STANCE_THORAX_HEIGHT = 0.1382  # cm acima da garra mais baixa
STANCE_THORAX_PITCH = 0.1552  # rad, nariz para cima
STANCE_CLAW_TIP = {  # ponta da garra no referencial do tórax (cm)
    "T1_left": (0.1204, 0.0601, -0.1586),
    "T1_right": (0.1300, -0.0677, -0.1598),
    "T2_left": (-0.0733, 0.1472, -0.1307),
    "T2_right": (-0.0715, -0.1489, -0.1307),
    "T3_left": (-0.2124, 0.1047, -0.1100),
    "T3_right": (-0.2147, -0.1083, -0.1095),
}
STANCE_JOINTS = {  # coxa_abduct, coxa_twist, coxa, femur_twist, femur, tibia, tarsus
    "T1_left": (-0.125, 0.238, 0.018, -0.091, 0.576, 0.339, 0.131),
    "T1_right": (-0.169, 0.248, 0.084, -0.091, 0.714, 0.462, 0.180),
    "T2_left": (-0.035, -0.314, -0.049, 0.279, 0.058, -0.137, -0.018),
    "T2_right": (-0.036, -0.315, -0.048, 0.277, 0.074, -0.101, -0.013),
    "T3_left": (0.020, -0.021, 0.058, 0.066, 0.134, 0.279, -0.029),
    "T3_right": (0.005, -0.009, 0.062, 0.066, 0.170, 0.338, -0.031),
}
STANCE_POS_SCALE = 0.01  # cm: 100 µm de desvio do pé pesam como 1 rad de erro de orientação


@dataclass(frozen=True)
class Stance:
    qpos: np.ndarray
    rot: np.ndarray  # orientação do tórax no mundo
    worst_angle: float  # rad, pior erro de orientação entre os patins
    worst_shift: float  # cm, maior desvio de um pé em relação à posição mediana


def thorax_rotation(pitch: float = STANCE_THORAX_PITCH, heading: float = 0.0) -> np.ndarray:
    """Tórax com a frente em `heading` (rad, em torno do eixo vertical) e o nariz `pitch` para cima."""
    ch, sh, cp, sp = np.cos(heading), np.sin(heading), np.cos(pitch), np.sin(pitch)
    yaw = np.array([[ch, -sh, 0.0], [sh, ch, 0.0], [0.0, 0.0, 1.0]])
    nose_up = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
    return yaw @ nose_up


def skate_yaws(toe_out_deg: float = 0.0) -> dict[str, float]:
    """Giro de cada patim em relação à frente do corpo, com a ponta para fora espelhada."""
    return {leg: toe_out_deg if leg.endswith("left") else -toe_out_deg for leg in LEGS}


def skating_stance(model: mujoco.MjModel, ik: LegIK, yaws_deg: dict[str, float] | None = None) -> Stance:
    yaws_deg = yaws_deg or skate_yaws(0.0)
    rot = thorax_rotation()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    data = mujoco.MjData(model)
    free = model.joint("free").qposadr[0]
    data.qpos[free : free + 7] = [0.0, 0.0, STANCE_THORAX_HEIGHT, *quat]
    up = rot.T @ np.array([0.0, 0.0, 1.0])
    worst_angle = worst_shift = 0.0
    for leg in LEGS:
        yaw = np.radians(yaws_deg[leg])
        forward = rot.T @ np.array([np.cos(yaw), np.sin(yaw), 0.0])
        q_ref = np.asarray(STANCE_JOINTS[leg])
        res = ik.solve(leg, np.asarray(STANCE_CLAW_TIP[leg]), forward, up=up, q0=q_ref, q_ref=q_ref, pos_scale=STANCE_POS_SCALE)
        data.qpos[ik.qadr[leg]] = res.q
        worst_angle, worst_shift = max(worst_angle, res.angle_err), max(worst_shift, res.pos_err)
    settle_on_floor(model, data)
    return Stance(data.qpos.copy(), rot, worst_angle, worst_shift)
