import mujoco
import numpy as np
import pytest

from mosca.body.fly import build_fly
from mosca.body.poses import apply_frame, hold_pose_ctrl, load_walking_frame, set_ctrl


@pytest.fixture(scope="module")
def model():
    return build_fly()


def test_counts_match_flybody_walking(model):
    # 59 ações = as mesmas do modo de andar do flybody; 19 juntas passivas viraram rígidas.
    assert model.nu == 59
    assert model.nq == 109 - 19
    thorax = model.body("thorax").id
    assert model.body_subtreemass[thorax] == pytest.approx(0.985e-3, rel=0.01)


def _rollout(model, n_control=250):
    data = mujoco.MjData(model)
    apply_frame(model, data, load_walking_frame(0, 0))
    base = hold_pose_ctrl(model, data)
    set_ctrl(model, data, base)
    traj = []
    for k in range(n_control):
        data.ctrl[:] = np.clip(base + 0.1 * np.sin(0.05 * k + np.arange(model.nu)), *model.actuator_ctrlrange.T)
        mujoco.mj_step(model, data, nstep=10)
        traj.append(data.qpos.copy())
    return np.array(traj)


def test_replay_is_bit_identical(model):
    assert np.array_equal(_rollout(model), _rollout(model))
