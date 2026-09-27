"""M2: uma MLP aprende a patinar por reforço (PPO).

Currículo: a velocidade pedida é sorteada em [0,3·v_max, v_max]; v_max sobe `--v-step`
quando 80% das tentativas da iteração terminam sem cair, até o fim do tempo e com
velocidade média perto da pedida. Ajuda opcional: parte das tentativas começa já andando
(empurrão inicial), com probabilidade que cai de 50% a zero em `--push-iters` iterações.
Tentativas a partir do repouso nunca somem, e cada tentativa registra se teve empurrão.

Exploração: ruído correlacionado no tempo (constante `--noise-tau`), porque ruído branco a
100 Hz faz as patas vibrarem, e a vibração sozinha empurra a mosca sobre os patins (a
política parecia andar só por causa do ruído). A cada `--eval-every` iterações, os testes
fixos rodam sem ruído, e o resultado entra em metrics.jsonl com o prefixo eval_.

Saídas em runs/<nome>/: config.json, metrics.jsonl (uma linha por iteração),
attempts.jsonl (uma linha por tentativa), checkpoints/ e latest.pt.

Uso:
    python scripts/m2_train.py --run m2_mlp_a --iters 1000 --envs 64 --threads 14 --torch-threads 2
    python scripts/m2_train.py --run m2_mlp_a --iters 2000 --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.env.skate_env import EnvConfig, RewardConfig, SkateVecEnv, attempt_seed  # noqa: E402
from mosca.paths import RUNS  # noqa: E402
from mosca.rl.ppo import ActorCritic, PPOConfig, RunningNorm, compute_gae, ppo_update  # noqa: E402

TEST_ATTEMPT_BASE = 10**9  # tentativas de teste: nunca aparecem no treino
REWARD_FLAGS = ("w_vel", "sigma_vel", "sigma_vel_rel", "w_yaw", "w_up", "w_roll", "sigma_roll", "w_slip",
                "w_cot", "w_rate", "w_leg_floor", "w_contact")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--threads", type=int, default=14, help="threads da física")
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control-dt", type=float, default=0.01)
    p.add_argument("--episode-seconds", type=float, default=5.0)
    p.add_argument("--v-start", type=float, default=1.0, help="cm/s")
    p.add_argument("--v-final", type=float, default=6.0, help="cm/s")
    p.add_argument("--v-step", type=float, default=0.5, help="cm/s")
    p.add_argument("--push-iters", type=int, default=200)
    p.add_argument("--init-std", type=float, default=0.25)
    p.add_argument("--noise-tau", type=float, default=0.05, help="s; 0 = ruído branco")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-speed", type=float, default=3.0)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--resume", action="store_true")
    defaults = RewardConfig()
    for name in REWARD_FLAGS:
        p.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    return p.parse_args()


def evaluate(env: SkateVecEnv, ac: ActorCritic, norm_obs: RunningNorm, speed: float) -> dict:
    """Testes fixos, sem ruído, a partir do repouso (os mesmos em toda avaliação)."""
    obs, _ = env.reset(TEST_ATTEMPT_BASE + np.arange(env.n), np.full(env.n, speed))
    while env.alive.any():
        with torch.no_grad():
            action = ac.actor(torch.from_numpy(norm_obs(obs).astype(np.float32))).numpy()
        obs, *_ = env.step(action)
    eps = env.episode_stats()
    return {f"eval_{k}": float(np.nanmean([e[k] for e in eps]))
            for k in ("speed", "rolling", "glide_frac", "fell", "seconds", "grounded_frac", "cot")}


def save(path: Path, ac, opt, norm_obs, norm_priv, state, env_cfg, args) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"ac": ac.state_dict(), "opt": opt.state_dict(), "norm_obs": norm_obs.state_dict(),
                "norm_priv": norm_priv.state_dict(), "state": state, "env_cfg": asdict(env_cfg),
                "args": vars(args)}, tmp)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    run_dir = RUNS / args.run
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    reward_cfg = RewardConfig(**{name: getattr(args, name) for name in REWARD_FLAGS})
    env_cfg = EnvConfig(n_envs=args.envs, n_threads=args.threads, control_dt=args.control_dt,
                        episode_seconds=args.episode_seconds, seed=args.seed, reward=reward_cfg)
    ppo_cfg = PPOConfig()
    env = SkateVecEnv(env_cfg)
    ac = ActorCritic(env.obs_dim, env.priv_dim, env.act_dim, init_std=args.init_std)
    beta = float(np.exp(-args.control_dt / args.noise_tau)) if args.noise_tau > 0 else 0.0
    opt = torch.optim.Adam(ac.parameters(), lr=ppo_cfg.lr)
    norm_obs, norm_priv = RunningNorm(env.obs_dim), RunningNorm(env.priv_dim)
    state = {"it": 0, "v_max": args.v_start, "lr": ppo_cfg.lr, "total_steps": 0, "attempts": 0}

    latest = run_dir / "latest.pt"
    if args.resume and latest.exists():
        ckpt = torch.load(latest, weights_only=False)
        ac.load_state_dict(ckpt["ac"])
        opt.load_state_dict(ckpt["opt"])
        norm_obs.load_state_dict(ckpt["norm_obs"])
        norm_priv.load_state_dict(ckpt["norm_priv"])
        state = ckpt["state"]
        print(f"retomando da iteração {state['it']}")
    else:
        (run_dir / "config.json").write_text(json.dumps(
            {"args": vars(args), "env": asdict(env_cfg), "ppo": asdict(ppo_cfg),
             "obs_dim": env.obs_dim, "priv_dim": env.priv_dim, "act_dim": env.act_dim}, indent=2), encoding="utf-8")

    n, horizon = env.n, env.max_steps
    print("  it   passos  retorno  dur(s) queda%  vel   rol  desliza v_max sucesso    lr      kl   s/it")
    for it in range(state["it"], args.iters):
        t0 = time.perf_counter()
        attempts = np.arange(it * n, (it + 1) * n)
        rng = np.random.default_rng(attempt_seed(args.seed, -1 - it))
        v_max = state["v_max"]
        v_cmd = rng.uniform(0.3 * v_max, v_max, n)
        p_push = max(0.0, 0.5 * (1 - it / args.push_iters)) if args.push_iters > 0 else 0.0
        push = np.where(rng.random(n) < p_push, v_cmd, 0.0)
        obs, priv = env.reset(attempts, v_cmd, push)
        gen = torch.Generator().manual_seed(attempt_seed(args.seed, -1 - it))

        buf = {k: np.zeros((horizon, n, d), np.float32) for k, d in
               (("obs", env.obs_dim), ("priv", env.priv_dim), ("act", env.act_dim), ("mean", env.act_dim))}
        for k in ("logp", "val", "rew"):
            buf[k] = np.zeros((horizon, n), np.float32)
        buf["valid"] = np.zeros((horizon, n), bool)
        raw_obs, raw_priv = [], []
        terminal = np.zeros(n, bool)
        last_obs, last_priv = np.zeros((n, env.obs_dim)), np.zeros((n, env.priv_dim))
        log_std = ac.log_std.detach().clone()
        noise = torch.randn((n, env.act_dim), generator=gen)

        steps = 0
        for t in range(horizon):
            valid = env.alive.copy()
            if not valid.any():
                break
            o = norm_obs(obs).astype(np.float32)
            pr = norm_priv(priv).astype(np.float32)
            with torch.no_grad():
                ot, pt = torch.from_numpy(o), torch.from_numpy(pr)
                dist = ac.dist(ot)
                if t > 0:  # ruído correlacionado: cada passo herda parte do anterior (variância 1 mantida)
                    noise = beta * noise + np.sqrt(1 - beta**2) * torch.randn((n, env.act_dim), generator=gen)
                action = dist.mean + dist.stddev * noise
                buf["logp"][t] = dist.log_prob(action).sum(-1).numpy()
                buf["val"][t] = ac.value(ot, pt).numpy()
            buf["obs"][t], buf["priv"][t] = o, pr
            buf["act"][t], buf["mean"][t] = action.numpy(), dist.mean.numpy()
            buf["valid"][t] = valid
            raw_obs.append(obs[valid])
            raw_priv.append(priv[valid])
            obs, priv, reward, term, trunc = env.step(action.numpy())
            buf["rew"][t] = reward * ppo_cfg.reward_scale
            terminal |= term
            last_obs[trunc], last_priv[trunc] = obs[trunc], priv[trunc]
            steps = t + 1

        with torch.no_grad():
            last_val = ac.value(torch.from_numpy(norm_obs(last_obs).astype(np.float32)),
                                torch.from_numpy(norm_priv(last_priv).astype(np.float32))).numpy()
        adv, ret = compute_gae(buf["rew"][:steps], buf["val"][:steps], buf["valid"][:steps], terminal, last_val,
                               ppo_cfg.gamma, ppo_cfg.lam)
        mask = buf["valid"][:steps]
        batch = {k: torch.from_numpy(buf[k][:steps][mask]) for k in ("obs", "priv", "act", "logp", "mean")}
        batch["adv"] = torch.from_numpy(adv[mask].astype(np.float32))
        batch["ret"] = torch.from_numpy(ret[mask].astype(np.float32))
        batch["log_std"] = log_std
        norm_obs.update(np.concatenate(raw_obs))
        norm_priv.update(np.concatenate(raw_priv))
        state["lr"], stats = ppo_update(ac, opt, batch, ppo_cfg, state["lr"], gen)

        episodes = env.episode_stats()
        ok = [e for e in episodes if not e["fell"] and e["steps"] >= horizon
              and abs(e["speed"] - e["v_cmd"]) < max(0.5, 0.25 * e["v_cmd"])]
        success = len(ok) / n
        if success >= 0.8:
            state["v_max"] = min(args.v_final, v_max + args.v_step)
        state["it"] = it + 1
        state["total_steps"] += int(mask.sum())
        state["attempts"] += n

        def mean(key):
            return float(np.mean([e[key] for e in episodes]))

        metrics = {"it": it, "steps": state["total_steps"], "attempts": state["attempts"], "v_max": v_max,
                   "p_push": p_push, "return": mean("return"), "seconds": mean("seconds"),
                   "fell": mean("fell"), "speed": mean("speed"), "rolling": mean("rolling"),
                   "glide_frac": mean("glide_frac"), "grounded_frac": mean("grounded_frac"), "success": success,
                   "lr": state["lr"], "std": float(ac.log_std.detach().exp().mean()),
                   "time": time.perf_counter() - t0, **stats}
        if args.eval_every and (it + 1) % args.eval_every == 0:
            ev = evaluate(env, ac, norm_obs, args.eval_speed)
            metrics.update(ev)
            print(f"      avaliação sem ruído a {args.eval_speed:g} cm/s: vel {ev['eval_speed']:.2f}, rolamento "
                  f"{ev['eval_rolling']:.2f}, desliza {ev['eval_glide_frac']:.2f}, quedas {ev['eval_fell']:.0%}, "
                  f"patins no chão {ev['eval_grounded_frac']:.2f}", flush=True)
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        with open(run_dir / "attempts.jsonl", "a", encoding="utf-8") as f:
            for e in episodes:
                f.write(json.dumps({"it": it, **e}) + "\n")
        print(f"{it:4d} {state['total_steps']:8d} {metrics['return']:8.1f} {metrics['seconds']:6.2f} "
              f"{100 * metrics['fell']:5.1f} {metrics['speed']:5.2f} {metrics['rolling']:5.2f} "
              f"{metrics['glide_frac']:6.2f} {v_max:5.1f} {success:6.2f} {state['lr']:.1e} {stats['kl']:.4f} "
              f"{metrics['time']:5.1f}", flush=True)

        save(latest, ac, opt, norm_obs, norm_priv, state, env_cfg, args)
        if (it + 1) % args.save_every == 0:
            save(run_dir / "checkpoints" / f"it{it + 1:05d}.pt", ac, opt, norm_obs, norm_priv, state, env_cfg, args)
    env.close()


if __name__ == "__main__":
    main()
