"""A professora do M4: a política de caminhada treinada do flybody, sem TensorFlow.

A política (Vaxenburg et al., Nature 2025; figshare da Janelia, GPL-3.0+) foi treinada por
reforço para imitar moscas reais andando. É uma MLP do Acme: Linear(741→512) → LayerNorm →
tanh → 3 × [Linear(512) → ELU] → média da ação (59). Os pesos saem direto do checkpoint do
SavedModel (mosca.body.tf_checkpoint); a ordem dos tensores segue a do Sonnet (viés, pesos
por camada), conferida rodando a política no ambiente original do flybody: com este
mapeamento a mosca anda 2,01 cm/s seguindo uma referência de 2 cm/s.

A observação (741) é a do ambiente `walk_imitation` do flybody, com as chaves em ordem
alfabética (como o Acme concatena): sentidos vestibulares, propriocepção, posição das
extremidades, força e toque nos tarsos e a trajetória de referência dos próximos 64
passos de controle (2 ms) no referencial da mosca. `TeacherObservation` monta esse vetor a
partir do estado cru do MuJoCo, em qualquer modelo com os mesmos nomes; juntas que não
existem no modelo (no nosso, os halteres são rígidos) entram com ângulo e velocidade 0.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from mosca.body.tf_checkpoint import read_checkpoint
from mosca.paths import ASSETS

POLICY_PREFIX = ASSETS / "flybody_policies" / "walking" / "variables" / "variables"
CONTROL_DT = 2e-3  # a política foi treinada com controle a 2 ms
FUTURE_STEPS = 64

# Juntas observadas pela política, na ordem do modelo do flybody (modo de andar, sem asas).
FLYBODY_JOINTS = (
    "head_abduct", "head_twist", "head", "abdomen_abduct", "abdomen", "abdomen_abduct_2", "abdomen_2",
    "abdomen_abduct_3", "abdomen_3", "abdomen_abduct_4", "abdomen_4", "abdomen_abduct_5", "abdomen_5",
    "abdomen_abduct_6", "abdomen_6", "abdomen_abduct_7", "abdomen_7", "haltere_left", "haltere_right",
    "coxa_abduct_T1_left", "coxa_twist_T1_left", "coxa_T1_left", "femur_twist_T1_left", "femur_T1_left",
    "tibia_T1_left", "tarsus_T1_left", "tarsus2_T1_left", "tarsus3_T1_left", "tarsus4_T1_left",
    "tarsus5_T1_left", "coxa_abduct_T1_right", "coxa_twist_T1_right", "coxa_T1_right",
    "femur_twist_T1_right", "femur_T1_right", "tibia_T1_right", "tarsus_T1_right", "tarsus2_T1_right",
    "tarsus3_T1_right", "tarsus4_T1_right", "tarsus5_T1_right", "coxa_abduct_T2_left", "coxa_twist_T2_left",
    "coxa_T2_left", "femur_twist_T2_left", "femur_T2_left", "tibia_T2_left", "tarsus_T2_left",
    "tarsus2_T2_left", "tarsus3_T2_left", "tarsus4_T2_left", "tarsus5_T2_left", "coxa_abduct_T2_right",
    "coxa_twist_T2_right", "coxa_T2_right", "femur_twist_T2_right", "femur_T2_right", "tibia_T2_right",
    "tarsus_T2_right", "tarsus2_T2_right", "tarsus3_T2_right", "tarsus4_T2_right", "tarsus5_T2_right",
    "coxa_abduct_T3_left", "coxa_twist_T3_left", "coxa_T3_left", "femur_twist_T3_left", "femur_T3_left",
    "tibia_T3_left", "tarsus_T3_left", "tarsus2_T3_left", "tarsus3_T3_left", "tarsus4_T3_left",
    "tarsus5_T3_left", "coxa_abduct_T3_right", "coxa_twist_T3_right", "coxa_T3_right",
    "femur_twist_T3_right", "femur_T3_right", "tibia_T3_right", "tarsus_T3_right", "tarsus2_T3_right",
    "tarsus3_T3_right", "tarsus4_T3_right", "tarsus5_T3_right",
)
LEG_SIDES = ("T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right")
LEG_ACTUATORS = ("coxa_abduct", "coxa_twist", "coxa", "femur_twist", "femur", "tibia", "tarsus", "tarsus2")
# Ordem das 59 ações da política: o flybody agrupa adesão, cabeça, abdômen e patas e só então
# remapeia para os atuadores (FruitFly.apply_action). No modelo, a ordem é cabeça, abdômen,
# patas e adesão: mandar a ação direto põe cada comando no atuador errado.
ACTION_ORDER = (tuple(f"adhere_claw_{leg}" for leg in LEG_SIDES) + ("head_abduct", "head_twist", "head")
                + ("abdomen_abduct", "abdomen") + tuple(f"{a}_{leg}" for leg in LEG_SIDES for a in LEG_ACTUATORS))
APPENDAGE_SITES = tuple(f"claw_{leg}" for leg in LEG_SIDES) + ("head",)


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))


def action_to_ctrl_index(model: mujoco.MjModel, prefix: str = "") -> np.ndarray:
    """Índice do atuador de cada ação da política (ctrl[idx] = ação)."""
    return np.array([model.actuator(prefix + name).id for name in ACTION_ORDER])


class WalkingTeacher:
    """A MLP da política de caminhada; `__call__` devolve a ação média (lote, 59) na ordem da
    política (ACTION_ORDER), sem corte."""

    def __init__(self, prefix: Path = POLICY_PREFIX):
        t = {int(k.split("/")[1]): v.astype(np.float64) for k, v in read_checkpoint(prefix).items()}
        self.w1, self.b1, self.ln_offset, self.ln_scale = t[1], t[0], t[2], t[3]
        self.hidden = [(t[5], t[4]), (t[7], t[6]), (t[9], t[8])]
        self.w_loc, self.b_loc = t[11], t[10]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(obs)
        if self._torch is not None:
            return self._forward_torch(x)
        h = x @ self.w1 + self.b1
        h = (h - h.mean(-1, keepdims=True)) / np.sqrt(h.var(-1, keepdims=True) + 1e-5) * self.ln_scale + self.ln_offset
        h = np.tanh(h)
        for w, b in self.hidden:
            h = _elu(h @ w + b)
        return h @ self.w_loc + self.b_loc

    _torch = None

    def to(self, device) -> "WalkingTeacher":
        """Passa a rodar em torch (float32) no dispositivo dado; na CPU do Spark, o numpy em
        float64 levava ~300 ms por passo com 64 moscas."""
        import torch

        def t(x):
            return torch.as_tensor(x, dtype=torch.float32, device=device)

        self._torch = {"device": device, "w1": t(self.w1), "b1": t(self.b1), "ln_s": t(self.ln_scale), "ln_o": t(self.ln_offset),
                       "hidden": [(t(w), t(b)) for w, b in self.hidden], "w_loc": t(self.w_loc), "b_loc": t(self.b_loc)}
        return self

    def _forward_torch(self, x: np.ndarray) -> np.ndarray:
        import torch

        p = self._torch
        with torch.no_grad():
            h = torch.as_tensor(x, dtype=torch.float32, device=p["device"]) @ p["w1"] + p["b1"]
            h = torch.nn.functional.layer_norm(h, (h.shape[-1],), eps=1e-5) * p["ln_s"] + p["ln_o"]
            h = torch.tanh(h)
            for w, b in p["hidden"]:
                h = torch.nn.functional.elu(h @ w + b)
            return (h @ p["w_loc"] + p["b_loc"]).cpu().numpy().astype(np.float64)


def _quat_conj_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """q1⁻¹ ⊗ q2 para q1 (4,) unitário e q2 (K, 4): q2 visto no referencial local de q1."""
    w1, x1, y1, z1 = q1[0], -q1[1], -q1[2], -q1[3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=1)


class TeacherObservation:
    """Monta a observação de 741 posições da política a partir de MjModel/MjData."""

    def __init__(self, model: mujoco.MjModel, prefix: str = ""):
        def jid(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)

        ids = [jid(n) for n in FLYBODY_JOINTS]
        self.present = np.array([i >= 0 for i in ids])
        self.qadr = np.array([model.jnt_qposadr[i] for i in ids if i >= 0])
        self.dadr = np.array([model.jnt_dofadr[i] for i in ids if i >= 0])

        def sensor(name):
            s = model.sensor(prefix + name)
            return np.arange(s.adr[0], s.adr[0] + s.dim[0])

        self.acc, self.gyro, self.vel = sensor("accelerometer"), sensor("gyro"), sensor("velocimeter")
        self.force = np.concatenate([sensor(f"force_tarsus_{leg}") for leg in LEG_SIDES])
        self.touch = np.concatenate([sensor(f"touch_claw_{leg}") for leg in LEG_SIDES])
        self.sites = np.array([model.site(prefix + s).id for s in APPENDAGE_SITES])
        self.thorax = model.body(prefix + "thorax").id
        # Junta livre no tórax (nosso modelo) ou num ancestral dele (no dm_control, o corpo de encaixe).
        ancestors, b = [], self.thorax
        while b > 0:
            ancestors.append(b)
            b = model.body_parentid[b]
        free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                and model.jnt_bodyid[j] in ancestors]
        self.root_qadr = model.jnt_qposadr[free[0]]
        self.nu = model.nu

    def __call__(self, data: mujoco.MjData, ref_qpos: np.ndarray, step: int,
                 sensordata: np.ndarray | None = None) -> np.ndarray:
        """`ref_qpos` (passos, 7): trajetória de referência a 2 ms; `step`: passo de controle atual;
        `sensordata`: leituras dos sensores a usar (no flybody, a média dos subpassos do passo)."""
        sens = data.sensordata if sensordata is None else sensordata
        xmat = data.xmat[self.thorax].reshape(3, 3)
        torso = data.xpos[self.thorax]
        fly_pos = data.qpos[self.root_qadr : self.root_qadr + 3]
        fly_quat = data.qpos[self.root_qadr + 3 : self.root_qadr + 7]
        joints_pos = np.zeros(len(FLYBODY_JOINTS))
        joints_vel = np.zeros(len(FLYBODY_JOINTS))
        joints_pos[self.present] = data.qpos[self.qadr]
        joints_vel[self.present] = data.qvel[self.dadr]
        ref = ref_qpos[step : step + FUTURE_STEPS + 1]
        parts = {
            "accelerometer": sens[self.acc],
            "actuator_activation": data.act[: self.nu],
            "appendages_pos": ((data.site_xpos[self.sites] - torso) @ xmat).ravel(),
            "force": sens[self.force],
            "gyro": sens[self.gyro],
            "joints_pos": joints_pos,
            "joints_vel": joints_vel,
            "ref_displacement": ((ref[:, :3] - fly_pos) @ xmat).ravel(),
            "ref_root_quat": _quat_conj_mult(fly_quat / np.linalg.norm(fly_quat), ref[:, 3:7]).ravel(),
            "touch": sens[self.touch],
            "velocimeter": sens[self.vel],
            "world_zaxis": xmat[2].copy(),
        }
        return np.concatenate([parts[k] for k in sorted(parts)])


REF_HEIGHT = 0.1278  # altura do tórax nas referências do flybody (cm)


def straight_trajectory(n_steps: int, speed: float, yaw_speed: float = 0.0, init_pos=(0.0, 0.0, REF_HEIGHT),
                        heading: float = 0.0, dt: float = CONTROL_DT) -> np.ndarray:
    """Trajetória de referência (passos, 7) a velocidade constante, reta ou em curva (yaw_speed em rad/s),
    na mesma forma da `constant_speed_trajectory` do flybody."""
    qpos = np.zeros((n_steps, 7))
    yaw = heading + yaw_speed * dt * np.arange(n_steps)
    qpos[:, :2] = init_pos[:2]
    qpos[1:, :2] += np.cumsum(speed * dt * np.stack([np.cos(yaw[:-1]), np.sin(yaw[:-1])], axis=1), axis=0)
    qpos[:, 2] = init_pos[2]
    qpos[:, 3] = np.cos(yaw / 2)
    qpos[:, 6] = np.sin(yaw / 2)
    return qpos


def heading_of(quat: np.ndarray) -> float:
    """Rumo (rad) de um quatérnio (w, x, y, z): o ângulo do eixo x do corpo projetado no chão."""
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
