import numpy as np
import pytest

from mosca.env.skate_env import EnvConfig, SkateVecEnv


@pytest.fixture(scope="module")
def env():
    e = SkateVecEnv(EnvConfig(n_envs=4, n_threads=2, seed=7))
    yield e
    e.close()


def test_shapes(env):
    obs, priv = env.reset(np.arange(4), np.full(4, 1.0))
    assert obs.shape == (4, env.obs_dim) and priv.shape == (4, env.priv_dim)
    assert np.isfinite(obs).all() and np.isfinite(priv).all()


def test_standing_still_does_not_fall(env):
    env.reset(np.arange(4), np.zeros(4))
    for _ in range(50):  # 0,5 s segurando a postura canônica
        obs, priv, reward, terminated, truncated = env.step(np.zeros((4, env.act_dim)))
        assert not terminated.any()
        assert np.isfinite(reward).all()
    assert all(d.qpos[2] > 0.1 for d in env.datas)


def test_same_attempt_same_actions_is_bit_identical(env):
    rng = np.random.default_rng(0)
    actions = rng.uniform(-0.3, 0.3, size=(30, env.act_dim))

    def rollout():
        env.reset(np.full(4, 123), np.full(4, 2.0))
        traj = []
        for a in actions:
            obs, *_ = env.step(np.tile(a, (4, 1)))
            traj.append(obs.copy())
        return np.array(traj)

    first, second = rollout(), rollout()
    assert np.array_equal(first, second)
    assert np.array_equal(first[:, 0], first[:, 3])  # ambientes diferentes, mesma tentativa


def test_different_attempts_start_differently(env):
    obs, _ = env.reset(np.array([1, 2, 3, 4]), np.zeros(4))
    assert not np.array_equal(obs[0], obs[1])


def test_falling_ends_the_attempt(env):
    env.reset(np.arange(4), np.zeros(4))
    d = env.datas[0]
    d.qpos[2] = 0.3  # no ar, de cabeça para baixo (180° em torno do eixo x)
    d.qpos[3:7] = [0.0, 1.0, 0.0, 0.0]
    _, _, _, terminated, _ = env.step(np.zeros((4, env.act_dim)))
    assert terminated[0] and not env.alive[0]
    assert env.episode_stats()[0]["fell"]
    assert not terminated[1:].any()


def test_skate_velocity_is_measured_along_the_skate(env):
    # Mosca empurrada a 1 cm/s para a frente: os patins, que apontam para a frente, andam ~1 cm/s
    # ao longo do próprio eixo e quase nada de lado (priv = altura, ao longo ×6, de lado ×6, ...).
    _, priv = env.reset(np.arange(4), np.full(4, 1.0), push=np.full(4, 1.0))
    along, side = priv[:, 1:7], priv[:, 7:13]
    assert (along > 0.8).all() and (along < 1.05).all()
    assert (np.abs(side) < 0.5).all()
