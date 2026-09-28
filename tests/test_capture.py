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


def test_turn_commands_are_captured_and_replayed(tmp_path):
    from mosca.env.skate_env import yaw_schedule

    cfg = EnvConfig(n_envs=2, n_threads=2, episode_seconds=0.4)
    env = SkateVecEnv(cfg)
    attempts, v_cmd, push = np.array([5, 6]), np.array([1.0, 2.0]), np.array([0.0, 0.0])
    sched = yaw_schedule(np.random.default_rng(2), 2, env.max_steps, 1.5, switch_steps=10)
    assert np.abs(sched).max() <= 1.5 and len(np.unique(sched[:, 0])) > 1
    env.reset(attempts, v_cmd, push, yaw_cmd=sched[0])
    rec = Recorder(env, iteration=0, attempts=attempts, v_cmd=v_cmd, push=push, checkpoint_every=10)
    rng = np.random.default_rng(3)
    t = 0
    while env.alive.any():
        actions = rng.normal(size=(2, env.act_dim)).astype(np.float32)
        env.set_commands(yaw_cmd=sched[t])
        rec.before_step(actions)
        env.step(actions)
        t += 1
    stats = env.episode_stats()
    env.close()
    rec.finish().save(tmp_path / "it00000.npz")
    cap = IterationCapture.load(tmp_path / "it00000.npz")
    assert np.array_equal(cap.yaw_cmd, sched[: len(cap.actions)])
    res = replay(cap, 1, cfg)
    assert res["max_deviation"] == 0.0
    assert res["return"] == stats[1]["return"] and res["yaw_error"] == stats[1]["yaw_error"]


def test_no_turns_draw_nothing_from_the_generator():
    from mosca.env.skate_env import yaw_schedule

    a, b = np.random.default_rng(7), np.random.default_rng(7)
    assert not yaw_schedule(a, 3, 50, 0.0, 10).any()
    assert a.random() == b.random()
