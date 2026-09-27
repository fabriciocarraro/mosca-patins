"""Ambiente de caminhada (M4): a mosca base, com pés e adesão, controlada a cada 2 ms.

É o ambiente da professora (a política de caminhada do flybody) e da destilação para o
conectoma. Segue a receita do `walk_imitation` do flybody: física a 0,2 ms, 10 subpassos
por passo de controle, sensores de força, toque, acelerômetro, giroscópio e velocímetro
como a média dos 10 subpassos, partida com o tórax a 0,1278 cm, orientação da referência e
juntas no ângulo padrão do modelo.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mosca.body.fly import PhysicsConfig, build_fly_spec
from mosca.walking.teacher import CONTROL_DT, FUTURE_STEPS, TeacherObservation, action_to_ctrl_index


class WalkingEnv:
    def __init__(self, model: mujoco.MjModel | None = None):
        self.model = model if model is not None else build_fly_spec(PhysicsConfig()).compile()
        self.data = mujoco.MjData(self.model)
        self.substeps = round(CONTROL_DT / self.model.opt.timestep)
        self.obs_fn = TeacherObservation(self.model)
        self.thorax = self.model.body("thorax").id
        self.ctrl_lo, self.ctrl_hi = self.model.actuator_ctrlrange.T
        self.ctrl_index = action_to_ctrl_index(self.model)  # ação da política (ordem do flybody) -> atuador
        self.sensor_mean = np.zeros(self.model.nsensordata)
        self.ref = None
        self.t = 0

    def reset(self, ref_qpos: np.ndarray) -> np.ndarray:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.obs_fn.root_qadr : self.obs_fn.root_qadr + 7] = ref_qpos[0]
        mujoco.mj_forward(m, d)
        self.sensor_mean = d.sensordata.copy()
        self.ref, self.t = ref_qpos, 0
        return self.observation()

    def observation(self) -> np.ndarray:
        return self.obs_fn(self.data, self.ref, self.t, sensordata=self.sensor_mean)

    def step(self, action: np.ndarray) -> np.ndarray:
        """`action` (59) na ordem da política do flybody (ACTION_ORDER)."""
        m, d = self.model, self.data
        ctrl = np.empty(m.nu)
        ctrl[self.ctrl_index] = action
        d.ctrl[:] = np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)
        acc = np.zeros(m.nsensordata)
        for _ in range(self.substeps):
            mujoco.mj_step(m, d)
            acc += d.sensordata
        self.sensor_mean = acc / self.substeps
        self.t += 1
        return self.observation()

    @property
    def steps_left(self) -> int:
        return len(self.ref) - FUTURE_STEPS - 1 - self.t
