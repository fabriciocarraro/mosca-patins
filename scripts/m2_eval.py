"""M2: avalia uma política nos 20 testes fixos e confere os critérios de pronto.

Testes: 20 tentativas com sementes fixas (as mesmas para qualquer política), a partir do
repouso, sem empurrão, com a ação média da política (sem ruído de exploração) e
velocidade pedida de 3 cm/s. Critério do M2: ≥80% dos testes com média ≥3 cm/s em 5 s,
≥25% do tempo deslizando e <10% de quedas.

Uso:
    python scripts/m2_eval.py runs/m2_mlp_a/latest.pt
    python scripts/m2_eval.py runs/m2_mlp_a/latest.pt --gif outputs/m2_teste0.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.env.skate_env import EnvConfig, RewardConfig, SkateVecEnv  # noqa: E402
from mosca.rl.ppo import ActorCritic, RunningNorm  # noqa: E402

TEST_ATTEMPT_BASE = 10**9  # números das tentativas de teste: nunca aparecem no treino


def load(path: str, n_envs: int, threads: int):
    ckpt = torch.load(path, weights_only=False)
    env_cfg = dict(ckpt["env_cfg"])
    env_cfg["reward"] = RewardConfig(**env_cfg["reward"])
    env_cfg.update(n_envs=n_envs, n_threads=threads)
    env = SkateVecEnv(EnvConfig(**env_cfg))
    ac = ActorCritic(env.obs_dim, env.priv_dim, env.act_dim)
    ac.load_state_dict(ckpt["ac"])
    norm = RunningNorm(env.obs_dim)
    norm.load_state_dict(ckpt["norm_obs"])
    return env, ac, norm, ckpt["state"]


def policy(ac: ActorCritic, norm: RunningNorm, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return ac.actor(torch.from_numpy(norm(obs).astype(np.float32))).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--tests", type=int, default=20)
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gif", default="", help="grava um GIF do teste 0 (câmera lenta 4×)")
    args = parser.parse_args()

    env, ac, norm, state = load(args.checkpoint, args.tests, args.threads)
    attempts = TEST_ATTEMPT_BASE + np.arange(args.tests)
    obs, _ = env.reset(attempts, np.full(args.tests, args.speed))
    while env.alive.any():
        obs, *_ = env.step(policy(ac, norm, obs))
    episodes = env.episode_stats()

    print(f"checkpoint: iteração {state['it']}, {state['attempts']} tentativas de treino, v_max {state['v_max']:.1f} cm/s")
    print(" teste  dur(s)  queda  vel(cm/s)  rolamento  desliza")
    for k, e in enumerate(episodes):
        print(f"  {k:4d}  {e['seconds']:6.2f}  {'sim' if e['fell'] else 'não':>5}  {e['speed']:9.2f}  {e['rolling']:9.2f}  {e['glide_frac']:7.2f}")
    fast = np.mean([not e["fell"] and e["seconds"] >= env.cfg.episode_seconds - 1e-9 and e["speed"] >= args.speed for e in episodes])
    glide = np.mean([e["glide_frac"] for e in episodes])
    falls = np.mean([e["fell"] for e in episodes])
    print(f"\nmédia ≥{args.speed:g} cm/s em {env.cfg.episode_seconds:g} s: {fast:.0%} dos testes (critério ≥80%)")
    print(f"tempo deslizando: {glide:.0%} (critério ≥25%); rolamento médio {np.mean([e['rolling'] for e in episodes]):.2f}")
    print(f"quedas: {falls:.0%} (critério <10%)")
    print("M2:", "OK" if fast >= 0.8 and glide >= 0.25 and falls < 0.1 else "ainda não")
    env.close()

    if args.gif:
        from PIL import Image

        env, ac, norm, _ = load(args.checkpoint, 1, 1)
        obs, _ = env.reset(attempts[:1], np.full(1, args.speed))
        renderer = mujoco.Renderer(env.model, height=480, width=854)
        cam = mujoco.MjvCamera()
        cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, env.thorax
        cam.distance, cam.elevation, cam.azimuth = 0.9, -22, 120
        frames = []
        while env.alive.any():
            obs, *_ = env.step(policy(ac, norm, obs))
            renderer.update_scene(env.datas[0], camera=cam)
            frames.append(Image.fromarray(renderer.render()))
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=40, loop=0)
        print(f"GIF: {args.gif} ({len(frames)} quadros a 100 Hz, reproduzido a 25 fps = 4× mais lento)")
        env.close()


if __name__ == "__main__":
    main()
