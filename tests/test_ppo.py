import numpy as np
import pytest

from mosca.rl.ppo import RunningNorm, compute_gae


def test_gae_terminal_episode_is_discounted_sum():
    rew = np.ones((3, 1))
    val = np.zeros((3, 1))
    valid = np.ones((3, 1), dtype=bool)
    adv, ret = compute_gae(rew, val, valid, terminal=np.array([True]), last_val=np.array([99.0]), gamma=0.5, lam=1.0)
    assert ret[:, 0] == pytest.approx([1.75, 1.5, 1.0])  # queda: nada depois do último passo


def test_gae_timeout_bootstraps_last_value():
    rew = np.ones((3, 1))
    val = np.zeros((3, 1))
    valid = np.ones((3, 1), dtype=bool)
    adv, ret = compute_gae(rew, val, valid, terminal=np.array([False]), last_val=np.array([4.0]), gamma=0.5, lam=1.0)
    assert ret[:, 0] == pytest.approx([2.25, 2.5, 3.0])


def test_gae_handles_episodes_of_different_lengths():
    rew = np.ones((3, 2))
    val = np.zeros((3, 2))
    valid = np.array([[True, True], [True, False], [True, False]])
    adv, ret = compute_gae(rew, val, valid, terminal=np.array([True, False]), last_val=np.array([0.0, 2.0]),
                           gamma=0.5, lam=1.0)
    assert ret[:, 0] == pytest.approx([1.75, 1.5, 1.0])
    assert ret[:, 1] == pytest.approx([2.0, 0.0, 0.0])  # só o primeiro passo existe


def test_running_norm_matches_batch_statistics():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, size=(10_000, 4))
    norm = RunningNorm(4)
    for chunk in np.array_split(data, 7):
        norm.update(chunk)
    assert norm.mean == pytest.approx(data.mean(axis=0), abs=1e-3)
    assert norm.var == pytest.approx(data.var(axis=0), rel=1e-3)
