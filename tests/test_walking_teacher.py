"""A política de caminhada do flybody, convertida sem TensorFlow, anda no nosso modelo."""

import numpy as np
import pytest

from mosca.walking.teacher import ACTION_ORDER, POLICY_PREFIX, action_to_ctrl_index, straight_trajectory

HAS_POLICY = POLICY_PREFIX.with_name("variables.index").exists()


def test_action_order_covers_every_actuator_once():
    from mosca.body.fly import PhysicsConfig, build_fly_spec

    m = build_fly_spec(PhysicsConfig()).compile()
    idx = action_to_ctrl_index(m)
    assert len(ACTION_ORDER) == m.nu == 59 and sorted(idx.tolist()) == list(range(m.nu))
    assert m.actuator(int(idx[0])).name == "adhere_claw_T1_left"  # a política começa pela adesão


def test_straight_trajectory_matches_flybody_convention():
    ref = straight_trajectory(501, 2.0)
    assert np.allclose(ref[500, :3], [2.0, 0.0, 0.1278]) and np.allclose(ref[:, 3:], [1, 0, 0, 0])
    turn = straight_trajectory(501, 2.0, yaw_speed=1.0)
    assert np.isclose(2 * np.arctan2(turn[500, 6], turn[500, 3]), 1.0 * 0.002 * 500)


@pytest.mark.skipif(not HAS_POLICY, reason="política do flybody não baixada (download_assets.py --walking-policy)")
@pytest.mark.parametrize("speed", [1.0, 3.0])
def test_teacher_walks_in_our_model(speed):
    from mosca.walking.env import WalkingEnv
    from mosca.walking.teacher import WalkingTeacher

    env, teacher = WalkingEnv(), WalkingTeacher()
    ref = straight_trajectory(700, speed)
    obs = env.reset(ref)
    assert obs.shape == (741,)
    start = env.data.xpos[env.thorax].copy()
    for _ in range(500):  # 1 s
        obs = env.step(teacher(obs)[0])
    pos = env.data.xpos[env.thorax]
    assert np.linalg.norm(pos[:2] - ref[500, :2]) < 0.05  # segue a referência a menos de meio milímetro
    assert pos[2] > 0.11 and np.linalg.norm(pos[:2] - start[:2]) > 0.9 * speed
