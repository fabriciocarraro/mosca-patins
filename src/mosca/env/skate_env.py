"""Ambiente de RL da mosca de patins: vários ambientes em lote, física em threads.

Cada ambiente tem o seu MjData e todos compartilham o mesmo MjModel. O MuJoCo libera o GIL
em `mj_step`, então as threads rodam a física em paralelo. A política controla os 42
servos das patas (7 por pata) como desvio da postura canônica "parada de patins"; cabeça
e abdômen ficam parados.

Uma tentativa é um episódio inteiro: começa parada de patins e termina na queda ou no fim
do tempo. A semente de cada tentativa sai de hash(semente da execução, número da
tentativa), então qualquer tentativa pode ser refeita. Recompensa, queda e comandos seguem
a seção 3 de docs/plano.md.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import mujoco
import numpy as np

from mosca.body.fly import LEG_SEGMENTS, LEGS
from mosca.body.ik import LEG_JOINTS, LegIK
from mosca.body.poses import hold_pose_ctrl, set_ctrl
from mosca.body.skates import build_skater
from mosca.body.stance import skating_stance

BODY_PARTS = ("thorax", "head", "rostrum", "haustellum", "labrum", "antenna", "wing", "abdomen", "haltere")


@dataclass(frozen=True)
class RewardConfig:
    """Pesos da recompensa por passo.

    Parada, a mosca só ganha os termos de postura (w_up + w_yaw); andando na velocidade
    pedida e deslizando, ganha também w_vel + w_roll. Nas execuções A a D, o rolamento pagava
    integral com a mosca parada (patim e corpo a 0 cm/s) e a velocidade era uma gaussiana
    estreita, sem sinal longe do alvo: ficar parada rendia 2/3 do máximo, e a política média
    não saía do lugar.
    """

    w_vel: float = 1.0
    vel_shape: str = "tent"  # "tent": 1 − |v − v_pedida| / v_pedida, com inclinação desde v = 0; "gauss"
    sigma_vel: float = 1.0  # cm/s, só para "gauss"; tolerância = sigma_vel + sigma_vel_rel·v_pedida
    sigma_vel_rel: float = 0.0
    w_yaw: float = 0.2
    sigma_yaw: float = 1.0  # rad/s
    w_up: float = 0.1
    w_roll: float = 0.5  # bônus de rolamento (patins apoiados andando junto com o corpo)
    sigma_roll: float = 0.5  # cm/s
    roll_gated: bool = True  # o bônus de rolamento escala com v/v_pedida (parada não ganha nada)
    w_slip: float = 0.2  # por cm/s de derrapagem lateral média dos patins apoiados
    w_cot: float = 0.0  # custo de transporte; liga depois do primeiro movimento
    w_rate: float = 0.01  # mudança brusca de ação
    w_leg_floor: float = 0.1  # por segmento de pata encostado no chão
    w_contact: float = 0.0  # fração dos 6 patins apoiados (desestimula levantar os patins para dar passos)
    ema_tau: float = 0.2  # s, média móvel da velocidade usada na recompensa


@dataclass(frozen=True)
class EnvConfig:
    n_envs: int = 64
    n_threads: int = 16
    control_dt: float = 0.01  # s
    episode_seconds: float = 5.0
    action_scale: float = 0.5  # rad por unidade de ação, em torno da postura canônica
    action_clip: float = 1.0  # a ação é cortada em ±action_clip (depois vêm os limites dos servos)
    init_joint_noise: float = 0.03  # rad
    min_load: float = 0.02  # fração do peso para um patim contar como apoiado
    min_height: float = 0.07  # cm
    max_tilt_deg: float = 60.0
    seed: int = 0
    reward: RewardConfig = field(default_factory=RewardConfig)


def attempt_seed(run_seed: int, attempt: int) -> int:
    digest = hashlib.sha256(f"{run_seed}:{attempt}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


class SkateVecEnv:
    def __init__(self, cfg: EnvConfig = EnvConfig(), model: mujoco.MjModel | None = None):
        self.cfg = cfg
        self.model = model if model is not None else build_skater()
        m = self.model
        self.n = cfg.n_envs
        self.substeps = round(cfg.control_dt / m.opt.timestep)
        self.max_steps = round(cfg.episode_seconds / cfg.control_dt)
        self.thorax = m.body("thorax").id
        self.weight = m.body_subtreemass[self.thorax] * float(np.linalg.norm(m.opt.gravity))

        # Postura canônica, assentada por 0,5 s segurando a pose: estado de partida de toda tentativa.
        stance = skating_stance(m, LegIK(m))
        rest = mujoco.MjData(m)
        rest.qpos[:] = stance.qpos
        set_ctrl(m, rest, hold_pose_ctrl(m, rest))
        for _ in range(round(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, rest)
        rest.qpos[:2] = 0.0
        self.rest = (rest.qpos.copy(), rest.qvel.copy(), rest.act.copy(), rest.ctrl.copy())

        names = [f"{j}_{leg}" for leg in LEGS for j in LEG_JOINTS]
        self.leg_act = np.array([m.actuator(n).id for n in names])
        self.leg_qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in names])
        self.leg_dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in names])
        self.leg_actadr = m.actuator_actadr[self.leg_act]
        self.leg_lo, self.leg_hi = m.actuator_ctrlrange[self.leg_act].T
        self.leg_jlo, self.leg_jhi = m.jnt_range[[m.joint(n).id for n in names]].T
        self.stance_ctrl = self.rest[3][self.leg_act].copy()
        self.act_dim = len(names)

        def sensor_slice(name):
            s = m.sensor(name)
            return slice(s.adr[0], s.adr[0] + s.dim[0])

        self.gyro = sensor_slice("gyro")
        self.velocimeter = sensor_slice("velocimeter")
        self.touch = np.array([m.sensor_adr[m.sensor(f"touch_skate_{leg}").id] for leg in LEGS])
        self.skates = np.array([m.body(f"skate_{leg}").id for leg in LEGS])

        # Categoria de cada geom para os contatos com o chão: 1 lâmina, 2 pata, 3 corpo (queda).
        self.floor = m.geom("floor").id
        self.geom_kind = np.zeros(m.ngeom, dtype=np.int8)
        for g in range(m.ngeom):
            name = m.geom(g).name
            body = m.body(m.geom_bodyid[g]).name
            if name.startswith("skate_runner"):
                self.geom_kind[g] = 1
            elif any(body.startswith(f"{seg}_") for seg in LEG_SEGMENTS) or body.startswith("skate_"):
                self.geom_kind[g] = 2
            elif any(body.startswith(p) for p in BODY_PARTS):
                self.geom_kind[g] = 3

        self.obs_dim = 4 * self.act_dim + 3 + 3 + 3 + len(LEGS) + 2 + 1
        self.priv_dim = 1 + 2 * len(LEGS) + len(LEGS) + 2 + 1
        self.datas = [mujoco.MjData(m) for _ in range(self.n)]
        self.pool = ThreadPoolExecutor(max_workers=cfg.n_threads)
        self._vel = np.zeros(6)

        # Gancho opcional chamado a cada subpasso de física (só com n_envs = 1, para renderizar a
        # re-simulação em câmera lenta); os subpassos um a um dão o mesmo resultado bit a bit.
        self.substep_callback = None
        self.v_cmd = np.zeros(self.n)
        self.yaw_cmd = np.zeros(self.n)
        self.ema_v = np.zeros(self.n)
        self.prev_action = np.zeros((self.n, self.act_dim))
        self.steps = np.zeros(self.n, dtype=int)
        self.alive = np.zeros(self.n, dtype=bool)
        self.stats = [dict() for _ in range(self.n)]

    # ------------------------------------------------------------------ reset

    def reset(self, attempts: np.ndarray, v_cmd: np.ndarray, push: np.ndarray | None = None):
        """Recomeça todos os ambientes; `push` é a velocidade inicial (cm/s), uma ajuda do currículo."""
        qpos, qvel, act, ctrl = self.rest
        push = np.zeros(self.n) if push is None else push
        for i, attempt in enumerate(attempts):
            rng = np.random.default_rng(attempt_seed(self.cfg.seed, int(attempt)))
            d = self.datas[i]
            mujoco.mj_resetData(self.model, d)
            d.qpos[:], d.qvel[:], d.act[:], d.ctrl[:] = qpos, qvel, act, ctrl
            noise = rng.normal(0.0, self.cfg.init_joint_noise, self.act_dim)
            d.qpos[self.leg_qadr] = np.clip(qpos[self.leg_qadr] + noise, self.leg_jlo, self.leg_jhi)
            d.qvel[0] = push[i]
            mujoco.mj_forward(self.model, d)
            self.stats[i] = {"attempt": int(attempt), "v_cmd": float(v_cmd[i]), "push": float(push[i]),
                             "distance": 0.0, "rolled": 0.0, "moved": 0.0, "glide_steps": 0,
                             "return": 0.0, "fell": False, "leg_floor": 0, "work": 0.0, "grounded": 0.0}
        self.v_cmd[:] = v_cmd
        self.yaw_cmd[:] = 0.0
        self.ema_v[:] = push
        self.prev_action[:] = 0.0
        self.steps[:] = 0
        self.alive[:] = True
        obs, priv = np.zeros((self.n, self.obs_dim)), np.zeros((self.n, self.priv_dim))
        for i in range(self.n):
            obs[i], priv[i], *_ = self._measure(i, np.zeros(self.act_dim), update=False)
        return obs, priv

    # ------------------------------------------------------------------- step

    def step(self, actions: np.ndarray):
        actions = np.clip(actions, -self.cfg.action_clip, self.cfg.action_clip)
        ctrl = np.clip(self.stance_ctrl + self.cfg.action_scale * actions, self.leg_lo, self.leg_hi)
        alive = np.flatnonzero(self.alive)

        def physics(i):
            d = self.datas[i]
            d.ctrl[self.leg_act] = ctrl[i]
            mujoco.mj_step(self.model, d, nstep=self.substeps)

        if self.substep_callback is not None:
            assert self.n == 1, "o gancho de subpasso é só para re-simulação com um ambiente"
            d = self.datas[0]
            d.ctrl[self.leg_act] = ctrl[0]
            for sub in range(self.substeps):
                mujoco.mj_step(self.model, d)
                self.substep_callback(self, sub)
        else:
            list(self.pool.map(physics, alive))

        obs, priv = np.zeros((self.n, self.obs_dim)), np.zeros((self.n, self.priv_dim))
        reward = np.zeros(self.n)
        terminated = np.zeros(self.n, dtype=bool)
        truncated = np.zeros(self.n, dtype=bool)
        for i in alive:
            obs[i], priv[i], reward[i], terminated[i] = self._measure(i, actions[i])
            self.prev_action[i] = actions[i]
            self.steps[i] += 1
            self.stats[i]["return"] += reward[i]
            if terminated[i]:
                self.stats[i]["fell"] = True
            elif self.steps[i] >= self.max_steps:
                truncated[i] = True
        self.alive[alive] &= ~(terminated[alive] | truncated[alive])
        return obs, priv, reward, terminated, truncated

    # ---------------------------------------------------------------- medidas

    def _measure(self, i: int, action: np.ndarray, update: bool = True):
        """Observação, recompensa e queda depois de aplicar `action`; `update=False` não mexe no estado."""
        m, d, cfg, rw = self.model, self.datas[i], self.cfg, self.cfg.reward
        rot = d.xmat[self.thorax].reshape(3, 3)
        gyro = d.sensordata[self.gyro]
        vel_local = d.sensordata[self.velocimeter]
        vel = rot @ vel_local
        heading = np.array([rot[0, 0], rot[1, 0], 0.0])
        heading /= max(np.linalg.norm(heading), 1e-9)
        lateral = np.array([-heading[1], heading[0], 0.0])
        v_fwd = float(vel @ heading)
        yaw_rate = float((rot @ gyro)[2])
        up_z = float(rot[2, 2])
        if update:
            self.ema_v[i] += (cfg.control_dt / rw.ema_tau) * (v_fwd - self.ema_v[i])

        loads = d.sensordata[self.touch] / self.weight
        along = np.zeros(len(LEGS))
        side = np.zeros(len(LEGS))
        yaw_sin = np.zeros(len(LEGS))
        for k, body in enumerate(self.skates):
            # XBODY: referencial do próprio patim (x ao longo dele). Com mjOBJ_BODY o MuJoCo usa o
            # referencial de inércia, cujos eixos saem reordenados (no patim, o x de inércia é o "para
            # cima"); as execuções A a E mediram assim o rolamento e o deslize, errado.
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, body, self._vel, 1)
            along[k], side[k] = self._vel[3], self._vel[4]
            x = d.xmat[body].reshape(3, 3)[:, 0]
            yaw_sin[k] = x @ lateral
        grounded = loads > cfg.min_load

        geoms = d.contact.geom[: d.ncon]
        with_floor = (geoms == self.floor).any(axis=1)
        others = np.where(geoms[with_floor, 0] == self.floor, geoms[with_floor, 1], geoms[with_floor, 0])
        kinds = self.geom_kind[others]
        body_hit = bool((kinds == 3).any())
        leg_floor = int((kinds == 2).sum())

        terminated = body_hit or d.qpos[2] < cfg.min_height or up_z < np.cos(np.radians(cfg.max_tilt_deg))
        terminated = terminated or not np.isfinite(d.qacc).all()

        # Recompensa
        if rw.vel_shape == "tent":
            r_vel = max(0.0, 1.0 - abs(self.ema_v[i] - self.v_cmd[i]) / max(self.v_cmd[i], 0.5))
        else:
            sigma_vel = rw.sigma_vel + rw.sigma_vel_rel * self.v_cmd[i]
            r_vel = np.exp(-(((self.ema_v[i] - self.v_cmd[i]) / sigma_vel) ** 2))
        r_yaw = np.exp(-(((yaw_rate - self.yaw_cmd[i]) / rw.sigma_yaw) ** 2))
        r_up = max(up_z, 0.0)
        if grounded.any() and self.v_cmd[i] >= 0.5:
            r_roll = float(np.mean(np.exp(-(((along[grounded] - v_fwd) / rw.sigma_roll) ** 2))))
            if rw.roll_gated:
                r_roll *= min(max(v_fwd, 0.0) / self.v_cmd[i], 1.0)
            slip = float(np.mean(np.abs(side[grounded])))
        else:
            r_roll, slip = 0.0, float(np.mean(np.abs(side[grounded]))) if grounded.any() else 0.0
        power = np.abs(d.actuator_force[self.leg_act] * d.actuator_velocity[self.leg_act]).sum()
        cot = power / (self.weight * max(abs(v_fwd), 0.5))
        rate = float(np.mean((action - self.prev_action[i]) ** 2))
        contact = float(grounded.mean())
        reward = (rw.w_vel * r_vel + rw.w_yaw * r_yaw + rw.w_up * r_up + rw.w_roll * r_roll + rw.w_contact * contact
                  - rw.w_slip * slip - rw.w_cot * cot - rw.w_rate * rate - rw.w_leg_floor * leg_floor)

        # Estatísticas da tentativa
        if update:
            s = self.stats[i]
            s["distance"] += v_fwd * cfg.control_dt
            s["leg_floor"] += leg_floor
            s["work"] += power * cfg.control_dt
            s["grounded"] += contact
            if grounded.any():
                speed = abs(v_fwd)
                s["rolled"] += float(np.minimum(np.abs(along[grounded]), 1.2 * speed).sum())
                s["moved"] += speed * int(grounded.sum())
                if speed > 0.5 and (np.abs(along[grounded]) > 0.7 * speed).all():
                    s["glide_steps"] += 1

        qpos = d.qpos[self.leg_qadr] - self.rest[0][self.leg_qadr]
        qvel = d.qvel[self.leg_dadr]
        act = d.act[self.leg_actadr] - self.stance_ctrl
        gravity = rot.T @ np.array([0.0, 0.0, -1.0])
        obs = np.concatenate([qpos, 0.1 * qvel, act, action, gravity, gyro, vel_local,
                              loads, [self.v_cmd[i], self.yaw_cmd[i]], [self.ema_v[i]]])
        priv = np.concatenate([[d.qpos[2]], along, side, yaw_sin, vel[:2], [leg_floor]])
        return obs, priv, float(reward), bool(terminated)

    # ------------------------------------------------------------- resultado

    def episode_stats(self) -> list[dict]:
        out = []
        for i in range(self.n):
            s = dict(self.stats[i])
            steps = max(int(self.steps[i]), 1)
            s["steps"] = int(self.steps[i])
            s["seconds"] = steps * self.cfg.control_dt
            s["speed"] = s["distance"] / s["seconds"]
            s["rolling"] = s["rolled"] / s["moved"] if s["moved"] > 0 else 0.0
            s["glide_frac"] = s["glide_steps"] / steps
            s["grounded_frac"] = s["grounded"] / steps
            s["cot"] = s["work"] / (self.weight * abs(s["distance"])) if abs(s["distance"]) > 1e-3 else float("nan")
            out.append(s)
        return out

    def close(self) -> None:
        self.pool.shutdown()
