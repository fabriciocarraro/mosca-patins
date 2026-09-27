"""Uma tentativa gravada no meio de uma leva é re-simulada sozinha, bit a bit."""

import numpy as np

from mosca.capture import IterationCapture, Recorder, replay
from mosca.env.skate_env import EnvConfig, SkateVecEnv


def test_captured_attempt_replays_bit_identically(tmp_path):
    cfg = EnvConfig(n_envs=3, n_threads=3, episode_seconds=0.6)
    env = SkateVecEnv(cfg)
    attempts, v_cmd, push = np.array([7, 8, 9]), np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.5, 0.0])
    env.reset(attempts, v_cmd, push)
    rec = Recorder(env, iteration=0, attempts=attempts, v_cmd=v_cmd, push=push, checkpoint_every=20)
    rng = np.random.default_rng(0)
    while env.alive.any():
        actions = rng.normal(size=(3, env.act_dim)).astype(np.float32)
        rec.before_step(actions)
        env.step(actions)
    batch_stats = env.episode_stats()
    final_qpos = env.datas[1].qpos.copy()
    env.close()
    rec.finish().save(tmp_path / "it00000.npz")

    cap = IterationCapture.load(tmp_path / "it00000.npz")
    seen = {}

    def grab(e, t):
        seen["qpos"] = e.datas[0].qpos.copy()

    res = replay(cap, 1, cfg, on_step=grab)
    assert res["max_deviation"] == 0.0
    assert np.array_equal(seen["qpos"], final_qpos)
    assert res["return"] == batch_stats[1]["return"] and res["steps"] == batch_stats[1]["steps"]


def test_substep_replay_matches_too(tmp_path):
    cfg = EnvConfig(n_envs=2, n_threads=2, episode_seconds=0.3)
    env = SkateVecEnv(cfg)
    attempts, v_cmd, push = np.array([3, 4]), np.array([1.0, 2.0]), np.array([0.0, 0.0])
    env.reset(attempts, v_cmd, push)
    rec = Recorder(env, iteration=0, attempts=attempts, v_cmd=v_cmd, push=push, checkpoint_every=10)
    rng = np.random.default_rng(1)
    while env.alive.any():
        actions = rng.normal(size=(2, env.act_dim)).astype(np.float32)
        rec.before_step(actions)
        env.step(actions)
    final_qpos = env.datas[0].qpos.copy()
    env.close()
    cap = rec.finish()
    count = {"n": 0}
    seen = {}

    def each_substep(e, t, sub):
        count["n"] += 1
        seen["qpos"] = e.datas[0].qpos.copy()

    res = replay(cap, 0, cfg, on_substep=each_substep)
    assert res["max_deviation"] == 0.0 and np.array_equal(seen["qpos"], final_qpos)
    assert count["n"] == int(cap.steps[0]) * round(cfg.control_dt / 2e-4)
