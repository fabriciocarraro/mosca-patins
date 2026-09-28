"""Ambiente de caminhada em lote (M4): física em threads, professora e aluno lado a lado.

Cada ambiente segue a receita de `mosca.walking.env` (controle a 2 ms, sensores médios dos
10 subpassos) com a sua trajetória de referência, a partir da qual a professora calcula o
que faria. O aluno (o conectoma) não vê a referência: recebe a velocidade e o giro pedidos
e uma observação no mesmo formato do ambiente de patinação (SkateVecEnv), para o
codificador passar direto da caminhada para os patins. Nela, os ângulos das juntas são
relativos à postura mediana de moscas reais paradas (STANCE_JOINTS) e a "carga no patim"
é o toque na garra.

Saída do aluno (48): por pata, os 7 servos das juntas em unidades de 0,15 rad em torno da
postura mediana e a adesão da garra (−1 solta, +1 presa). O tarso distal (tarsus2) fica em 0
e a cabeça e o abdômen ficam parados: a professora só "chacoalha" o tarsus2 (±0,9 sem
relação com o apoio), e o conectoma controla só as patas.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

from mosca.body.fly import LEG_SEGMENTS, LEGS, PhysicsConfig, build_fly_spec
from mosca.body.ik import LEG_JOINTS
from mosca.body.stance import STANCE_JOINTS
from mosca.walking.teacher import (CONTROL_DT, FUTURE_STEPS, REF_HEIGHT, TeacherObservation, action_to_ctrl_index,
                                   heading_of, straight_trajectory)

ACTION_SCALE = 0.15  # rad por unidade, como na patinação
BODY_PARTS = ("thorax", "head", "rostrum", "haustellum", "labrum", "antenna", "wing", "abdomen", "haltere")
N_OUT = 6 * len(LEG_JOINTS) + 6  # 42 servos das juntas + 6 adesões


class WalkingVecEnv:
    def __init__(self, n_envs: int, n_threads: int = 8, fixed_ctrl: dict[str, float] | None = None,
                 min_height: float = 0.07, max_tilt_deg: float = 60.0, ref_leak_tau: float = 0.0):
        self.model = build_fly_spec(PhysicsConfig()).compile()
        m = self.model
        self.n = n_envs
        # 0: referência fixa (a da reinicialização). > 0: a referência avança com os comandos e é puxada
        # para a posição e o rumo da mosca com esta constante de tempo (s), e o atraso acumulado que a
        # professora tenta recuperar fica limitado a ~velocidade × constante.
        self.ref_leak_tau = ref_leak_tau
        self.substeps = round(CONTROL_DT / m.opt.timestep)
        self.datas = [mujoco.MjData(m) for _ in range(n_envs)]
        self.pool = ThreadPoolExecutor(max_workers=n_threads)
        self.obs_fn = TeacherObservation(m)
        self.teacher_ctrl_index = action_to_ctrl_index(m)
        self.ctrl_lo, self.ctrl_hi = m.actuator_ctrlrange.T
        self.thorax = m.body("thorax").id
        self.weight = m.body_subtreemass[self.thorax] * float(np.linalg.norm(m.opt.gravity))
        self.min_height, self.min_up = min_height, np.cos(np.radians(max_tilt_deg))

        names = [f"{j}_{leg}" for leg in LEGS for j in LEG_JOINTS]
        self.leg_act = np.array([m.actuator(n).id for n in names])
        self.leg_qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in names])
        self.leg_dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in names])
        self.stance = np.concatenate([STANCE_JOINTS[leg] for leg in LEGS])
        self.adhesion_act = np.array([m.actuator(f"adhere_claw_{leg}").id for leg in LEGS])
        # Os atuadores filtram o comando (τ de 10 ms nas juntas, 7 ms na adesão): fração do caminho até o
        # novo comando percorrida a cada passo de controle, na ordem das saídas do aluno.
        tau_out = m.actuator_dynprm[np.concatenate([self.leg_act, self.adhesion_act]), 0]
        self.act_alpha = 1.0 - (1.0 - m.opt.timestep / tau_out) ** self.substeps
        self.fixed_ctrl = np.zeros(m.nu)  # cabeça, abdômen e tarsus2 (fixos)
        self.fixed_idx = np.array([i for i in range(m.nu) if i not in set(self.leg_act) | set(self.adhesion_act)])
        for name, value in (fixed_ctrl or {}).items():
            self.fixed_ctrl[m.actuator(name).id] = value

        def sensor(name):
            s = m.sensor(name)
            return np.arange(s.adr[0], s.adr[0] + s.dim[0])

        self.gyro, self.velocimeter = sensor("gyro"), sensor("velocimeter")
        self.touch = np.concatenate([sensor(f"touch_claw_{leg}") for leg in LEGS])
        self.floor = m.geom("floor").id
        self.body_geom = np.zeros(m.ngeom, dtype=bool)
        for g in range(m.ngeom):
            body = m.body(m.geom_bodyid[g]).name
            if not any(body.startswith(f"{seg}_") for seg in LEG_SEGMENTS) and any(body.startswith(p) for p in BODY_PARTS):
                self.body_geom[g] = True

        self.sensor_mean = np.zeros((n_envs, m.nsensordata))
        self.refs: list[np.ndarray] = [np.zeros((1, 7))] * n_envs
        self.t = np.zeros(n_envs, dtype=int)
        self.alive = np.zeros(n_envs, dtype=bool)
        self.fell = np.zeros(n_envs, dtype=bool)
        self.v_cmd = np.zeros(n_envs)
        self.yaw_cmd = np.zeros(n_envs)
        self.prev_out = np.zeros((n_envs, N_OUT))
        self.ref_xy = np.zeros((n_envs, 2))
        self.ref_heading = np.zeros(n_envs)

    # --------------------------------------------------------------- reset / step

    def reset(self, refs: list[np.ndarray], v_cmd: np.ndarray, yaw_cmd: np.ndarray) -> None:
        m = self.model
        for i, d in enumerate(self.datas):
            mujoco.mj_resetData(m, d)
            d.qpos[self.obs_fn.root_qadr : self.obs_fn.root_qadr + 7] = refs[i][0]
            mujoco.mj_forward(m, d)
            self.sensor_mean[i] = d.sensordata
        self.refs = list(refs)
        self.t[:] = 0
        self.alive[:] = True
        self.fell[:] = False
        self.v_cmd[:], self.yaw_cmd[:] = v_cmd, yaw_cmd
        self.prev_out[:] = 0.0
        self.ref_xy[:] = [r[0, :2] for r in refs]
        self.ref_heading[:] = [heading_of(r[0, 3:7]) for r in refs]

    def ctrl_from_student(self, out: np.ndarray) -> np.ndarray:
        """Saída do aluno (lote, 48) -> ctrl (lote, nu) na ordem dos atuadores."""
        ctrl = np.tile(self.fixed_ctrl, (len(out), 1))
        ctrl[:, self.leg_act] = self.stance + ACTION_SCALE * out[:, : 6 * len(LEG_JOINTS)]
        ctrl[:, self.adhesion_act] = 0.5 + 0.5 * out[:, 6 * len(LEG_JOINTS) :]
        return np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)

    def student_target(self, teacher_action: np.ndarray) -> np.ndarray:
        """Ação da professora (lote, 59; ordem da política) -> alvo do aluno (lote, 48), já cortada."""
        ctrl = np.zeros((len(teacher_action), self.model.nu))
        ctrl[:, self.teacher_ctrl_index] = teacher_action
        ctrl = np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)
        joints = (ctrl[:, self.leg_act] - self.stance) / ACTION_SCALE
        adhesion = (ctrl[:, self.adhesion_act] - 0.5) / 0.5
        return np.concatenate([joints, adhesion], axis=1)

    def teacher_ctrl(self, teacher_action: np.ndarray) -> np.ndarray:
        ctrl = np.zeros((len(teacher_action), self.model.nu))
        ctrl[:, self.teacher_ctrl_index] = teacher_action
        return np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)

    def step(self, ctrl: np.ndarray, out: np.ndarray | None = None) -> None:
        """Avança 2 ms com `ctrl` (lote, nu) nos ambientes vivos; `out` é a saída do aluno (para a observação)."""
        m = self.model
        alive = np.flatnonzero(self.alive)

        def physics(i):
            d = self.datas[i]
            d.ctrl[:] = ctrl[i]
            acc = np.zeros(m.nsensordata)
            for _ in range(self.substeps):
                mujoco.mj_step(m, d)
                acc += d.sensordata
            self.sensor_mean[i] = acc / self.substeps

        list(self.pool.map(physics, alive))
        if out is not None:
            self.prev_out[alive] = out[alive]
        root = self.obs_fn.root_qadr
        k = min(1.0, CONTROL_DT / self.ref_leak_tau) if self.ref_leak_tau > 0 else 0.0
        for i in alive:
            d = self.datas[i]
            self.t[i] += 1
            if k > 0:
                h = self.ref_heading[i]
                self.ref_xy[i] += self.v_cmd[i] * CONTROL_DT * np.array([np.cos(h), np.sin(h)])
                self.ref_heading[i] += self.yaw_cmd[i] * CONTROL_DT
                self.ref_xy[i] += k * (d.qpos[root : root + 2] - self.ref_xy[i])
                dh = heading_of(d.qpos[root + 3 : root + 7]) - self.ref_heading[i]
                self.ref_heading[i] += k * np.arctan2(np.sin(dh), np.cos(dh))
            up = d.xmat[self.thorax][8]
            geoms = d.contact.geom[: d.ncon]
            floor_hits = geoms[(geoms == self.floor).any(axis=1)]
            others = np.where(floor_hits[:, 0] == self.floor, floor_hits[:, 1], floor_hits[:, 0])
            fell = d.xpos[self.thorax][2] < self.min_height or up < self.min_up or bool(self.body_geom[others].any())
            fell = fell or not np.isfinite(d.qacc).all()
            done = self.t[i] >= len(self.refs[i]) - FUTURE_STEPS - 1
            if fell:
                self.fell[i] = True
            if fell or done:
                self.alive[i] = False

    # ------------------------------------------------------------- observações

    def teacher_obs(self) -> np.ndarray:
        out = np.zeros((self.n, 741))
        for i in np.flatnonzero(self.alive):
            d = self.datas[i]
            if self.ref_leak_tau > 0:
                ref = straight_trajectory(FUTURE_STEPS + 1, self.v_cmd[i], self.yaw_cmd[i],
                                          init_pos=(*self.ref_xy[i], REF_HEIGHT), heading=self.ref_heading[i])
                out[i] = self.obs_fn(d, ref, 0, sensordata=self.sensor_mean[i])
            else:
                out[i] = self.obs_fn(d, self.refs[i], int(self.t[i]), sensordata=self.sensor_mean[i])
        return out

    def student_obs(self) -> np.ndarray:
        """Observação do aluno no formato do SkateVecEnv (186 posições)."""
        obs = np.zeros((self.n, 4 * 42 + 3 + 3 + 3 + 6 + 2 + 1))
        for i in range(self.n):
            d = self.datas[i]
            rot = d.xmat[self.thorax].reshape(3, 3)
            qpos = d.qpos[self.leg_qadr] - self.stance
            qvel = d.qvel[self.leg_dadr]
            loads = self.sensor_mean[i][self.touch] / self.weight
            gravity = rot.T @ np.array([0.0, 0.0, -1.0])
            obs[i] = np.concatenate([qpos, 0.1 * qvel, np.zeros(42), self.prev_out[i, :42], gravity,
                                     self.sensor_mean[i][self.gyro], self.sensor_mean[i][self.velocimeter], loads,
                                     [self.v_cmd[i], self.yaw_cmd[i]], [0.0]])
        return obs

    def positions(self) -> np.ndarray:
        return np.array([d.xpos[self.thorax].copy() for d in self.datas])

    def close(self) -> None:
        self.pool.shutdown()
