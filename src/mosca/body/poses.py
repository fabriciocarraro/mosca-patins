"""Poses de moscas reais, tiradas da amostra de caminhada do flybody (500 Hz)."""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import mujoco
import numpy as np

from mosca.paths import WALKING_SAMPLE


@dataclass(frozen=True)
class WalkingFrame:
    joint_names: tuple[str, ...]
    joint_qpos: np.ndarray  # (102,)
    root_qpos: np.ndarray  # (7,) posição (cm) + quaternion


def load_walking_frame(snippet: int = 0, frame: int = 0) -> WalkingFrame:
    with h5py.File(WALKING_SAMPLE, "r") as f:
        names = tuple(n.decode() if isinstance(n, bytes) else n for n in f["id2name/joints"][()])
        traj = f["trajectories"][sorted(f["trajectories"].keys())[snippet]]
        return WalkingFrame(names, traj["qpos"][frame].astype(float), traj["root_qpos"][frame].astype(float))


def walking_speeds(snippet: int) -> np.ndarray:
    """Velocidade horizontal do corpo (cm/s) em cada quadro de um trecho."""
    with h5py.File(WALKING_SAMPLE, "r") as f:
        traj = f["trajectories"][sorted(f["trajectories"].keys())[snippet]]
        return np.linalg.norm(traj["root_qvel"][:, :2], axis=1)


def apply_frame(model: mujoco.MjModel, data: mujoco.MjData, frame: WalkingFrame, center: bool = True) -> None:
    """Copia a pose para o modelo, ignorando juntas que não existem nele (ex.: asas rígidas)."""
    root = frame.root_qpos.copy()
    if center:
        root[:2] = 0.0
    free = model.joint("free")
    data.qpos[free.qposadr[0] : free.qposadr[0] + 7] = root
    for name, value in zip(frame.joint_names, frame.joint_qpos):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = value
    data.qvel[:] = 0.0


def floor_contact_geoms(model: mujoco.MjModel) -> list[int]:
    """Geoms que podem tocar o chão: por máscara de colisão ou por par de contato explícito."""
    floor = model.geom("floor").id
    ids = {
        g
        for g in range(model.ngeom)
        if g != floor
        and (model.geom_contype[g] & model.geom_conaffinity[floor] or model.geom_conaffinity[g] & model.geom_contype[floor])
    }
    for g1, g2 in zip(model.pair_geom1, model.pair_geom2):
        if floor in (g1, g2):
            ids.add(g2 if g1 == floor else g1)
    return sorted(ids)


def settle_on_floor(model: mujoco.MjModel, data: mujoco.MjData, gap: float = 0.0) -> float:
    """Sobe ou desce a mosca até o ponto mais baixo ficar `gap` acima do chão. Devolve o ajuste (cm)."""
    floor = model.geom("floor").id
    mujoco.mj_forward(model, data)
    fromto = np.zeros(6)
    lowest = min(mujoco.mj_geomDistance(model, data, floor, g, 1.0, fromto) for g in floor_contact_geoms(model))
    shift = gap - lowest
    data.qpos[model.jnt_qposadr[model.joint("free").id] + 2] += shift
    mujoco.mj_forward(model, data)
    return shift


def hold_pose_ctrl(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Controles de posição que seguram a pose atual (juntas e tendões); adesão desligada."""
    ctrl = np.zeros(model.nu)
    mujoco.mj_forward(model, data)
    for i in range(model.nu):
        target = model.actuator_trnid[i, 0]
        kind = model.actuator_trntype[i]
        if kind == mujoco.mjtTrn.mjTRN_JOINT:
            ctrl[i] = data.qpos[model.jnt_qposadr[target]]
        elif kind == mujoco.mjtTrn.mjTRN_TENDON:
            ctrl[i] = data.ten_length[target]
    lo, hi = model.actuator_ctrlrange.T
    return np.clip(ctrl, lo, hi)


def set_ctrl(model: mujoco.MjModel, data: mujoco.MjData, ctrl: np.ndarray, settle_filters: bool = True) -> None:
    """Define os controles; com `settle_filters`, os filtros dos atuadores já começam no alvo."""
    data.ctrl[:] = ctrl
    if settle_filters:
        for i in range(model.nu):
            adr = model.actuator_actadr[i]
            if adr >= 0:
                data.act[adr] = ctrl[i]
