"""Marchas periódicas programadas à mão, para testar a física dos patins (M1).

Cada par de patas (frente, meio, trás) segue o mesmo traçado, espelhado entre os lados:
no apoio o pé varre de lado com o patim girado; no ar o pé volta subindo e descendo.
Um patim que não pode derrapar de lado, girado de ψ e empurrado de lado à velocidade u,
leva o corpo para a frente a u / tan ψ. A cinemática inversa converte o traçado de um
ciclo (no referencial do tórax) em alvos para os servos de posição.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from mosca.body.fly import LEGS
from mosca.body.ik import LegIK
from mosca.body.poses import hold_pose_ctrl
from mosca.body.stance import STANCE_JOINTS, STANCE_POS_SCALE, Stance

CONTROL_DT = 2e-3  # s
PAIRS = {"front": "T1", "middle": "T2", "hind": "T3"}


@dataclass(frozen=True)
class PairStroke:
    yaw_deg: float = 0.0  # giro do patim no apoio; positivo = ponta para fora
    sweep: float = 0.0  # cm de varredura lateral no apoio; positivo = para fora
    lift: float = 0.0  # cm, altura do pé no ar
    duty: float = 0.5  # fração do ciclo com o pé no chão
    phase: float = 0.0  # fração do ciclo em que o apoio começa

    @property
    def moves(self) -> bool:
        return bool(self.yaw_deg or self.sweep or self.lift)


@dataclass(frozen=True)
class Gait:
    period: float
    front: PairStroke
    middle: PairStroke
    hind: PairStroke

    def stroke(self, leg: str) -> PairStroke:
        pair = next(name for name, prefix in PAIRS.items() if leg.startswith(prefix))
        return getattr(self, pair)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Gait":
        return Gait(d["period"], *(PairStroke(**d[p]) for p in ("front", "middle", "hind")))


def pair_offset(stroke: PairStroke, t: float) -> tuple[float, float]:
    """Deslocamento lateral para fora e altura do pé na fase t ∈ [0, 1)."""
    s = (t - stroke.phase) % 1.0
    if s < stroke.duty:
        return stroke.sweep * (s / stroke.duty - 0.5), 0.0
    u = (s - stroke.duty) / (1.0 - stroke.duty)
    return stroke.sweep * (0.5 - u), stroke.lift * np.sin(np.pi * u)


def plan_cycle(model: mujoco.MjModel, ik: LegIK, stance: Stance, gait: Gait) -> tuple[np.ndarray, float]:
    """Alvos de qpos para cada passo de controle de um ciclo; devolve também o pior erro de orientação no apoio."""
    n = max(4, round(gait.period / CONTROL_DT))
    rot = stance.rot
    data = mujoco.MjData(model)
    data.qpos[:] = stance.qpos
    mujoco.mj_kinematics(model, data)
    thorax = model.body("thorax").id
    base = {leg: rot.T @ (data.xpos[model.body(f"skate_{leg}").id] - data.xpos[thorax]) for leg in LEGS}
    up = rot.T @ np.array([0.0, 0.0, 1.0])
    targets = np.tile(stance.qpos, (n, 1))
    worst = 0.0
    for leg in LEGS:
        stroke = gait.stroke(leg)
        if not stroke.moves:
            continue
        side = 1.0 if leg.endswith("left") else -1.0
        yaw = np.radians(side * stroke.yaw_deg)
        forward = rot.T @ np.array([np.cos(yaw), np.sin(yaw), 0.0])
        q_ref = np.asarray(STANCE_JOINTS[leg])
        q_prev = stance.qpos[ik.qadr[leg]]
        for k in range(n):
            lateral, height = pair_offset(stroke, k / n)
            pos = base[leg] + rot.T @ np.array([0.0, side * lateral, height])
            res = ik.solve(leg, pos, forward, up=up, q0=q_prev, q_ref=q_ref, pos_scale=STANCE_POS_SCALE)
            targets[k, ik.qadr[leg]] = q_prev = res.q
            if height == 0.0:
                worst = max(worst, res.angle_err)
    return targets, worst


def cycle_controls(model: mujoco.MjModel, targets: np.ndarray) -> np.ndarray:
    """Controles dos servos que seguram cada qpos-alvo do ciclo."""
    data = mujoco.MjData(model)
    out = []
    for q in targets:
        data.qpos[:] = q
        out.append(hold_pose_ctrl(model, data))
    return np.array(out)
