"""Cinemática inversa das patas: posiciona e orienta cada patim no referencial do tórax.

Cada pata tem 7 juntas (coxa ×3, fêmur ×2, tíbia, tornozelo) para 6 restrições (posição e
orientação do patim); a sobra é resolvida puxando de leve para uma postura de referência.
Serve para os testes de física e para os movimentos programados à mão do M1.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares

from mosca.body.fly import LEGS

LEG_JOINTS = ("coxa_abduct", "coxa_twist", "coxa", "femur_twist", "femur", "tibia", "tarsus")
POS_SCALE = 0.001  # cm: 10 µm de erro pesam como 1 rad


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    pos_err: float  # cm
    angle_err: float  # rad (pior entre direção da frente e direção de cima)


def _normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class LegIK:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        free = model.joint("free").qposadr[0]
        self.data.qpos[free : free + 7] = [0, 0, 0, 1, 0, 0, 0]  # tórax na origem: mundo = tórax
        self.qadr = {}
        self.bounds = {}
        for leg in LEGS:
            ids = [model.joint(f"{j}_{leg}").id for j in LEG_JOINTS]
            self.qadr[leg] = model.jnt_qposadr[ids]
            self.bounds[leg] = (model.jnt_range[ids, 0], model.jnt_range[ids, 1])
        self.skate = {leg: model.body(f"skate_{leg}").id for leg in LEGS}

    def skate_pose(self, leg: str, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[self.qadr[leg]] = q
        mujoco.mj_kinematics(self.model, self.data)
        body = self.skate[leg]
        return self.data.xpos[body].copy(), self.data.xmat[body].reshape(3, 3).copy()

    def solve(
        self,
        leg: str,
        pos: np.ndarray,
        forward: np.ndarray,
        up: np.ndarray = (0.0, 0.0, 1.0),
        q0: np.ndarray | None = None,
        q_ref: np.ndarray | None = None,
        reg: float = 1e-3,
        pos_scale: float = POS_SCALE,
    ) -> IKResult:
        """`pos_scale` maior deixa a posição mais frouxa em favor da orientação."""
        forward, up = _normalize(forward), _normalize(up)
        lo, hi = self.bounds[leg]
        q0 = np.clip(np.zeros(len(LEG_JOINTS)) if q0 is None else q0, lo + 1e-9, hi - 1e-9)
        q_ref = q0 if q_ref is None else q_ref

        def residual(q):
            p, rot = self.skate_pose(leg, q)
            return np.concatenate([(p - pos) / pos_scale, rot[:, 0] - forward, rot[:, 2] - up, reg * (q - q_ref)])

        sol = least_squares(residual, q0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, gtol=1e-12)
        p, rot = self.skate_pose(leg, sol.x)
        angle = max(np.arccos(np.clip(rot[:, 0] @ forward, -1, 1)), np.arccos(np.clip(rot[:, 2] @ up, -1, 1)))
        return IKResult(sol.x, float(np.linalg.norm(p - pos)), float(angle))
