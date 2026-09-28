"""M4: professora lenta, uma MLP que anda com a latência de uma rede de neurônios lentos (PPO).

A política de caminhada do flybody só anda reagindo quase na hora: com as ações atrasadas
10 ms, ou filtradas com τ de 20 ms, ela cai em menos de 1 s (o comando dela é um liga-desliga
que depende do estado dos atuadores a cada 2 ms). O conectoma (neurônios com τ ~20 ms, várias
camadas entre os sentidos e os motores) responde em 20 a 40 ms e não tem como imitá-la.

Esta professora vê só o que o conectoma vê (os sentidos das patas e os comandos de velocidade
e giro), e as suas saídas passam por um atraso e um filtro de primeira ordem (`--delay-ms`,
`--tau-ms`) antes de chegar aos atuadores. Ela parte da destilação da política do flybody
(m4_distill.py --student mlp com a mesma latência, `--init-from`) e treina por PPO para andar
na velocidade e no giro pedidos sem cair. Depois, o conectoma aprende com ela.

Recompensa por passo (2 ms): velocidade para a frente (média móvel de 0,1 s) em tenda em torno
da pedida, giro perto do pedido, tórax em pé e custo pequeno de mudança brusca do comando. A
tentativa termina na queda (sem bônus por continuar viva).

Uso (no Spark):
    python scripts/m4_slow_teacher.py --run lenta_a --init-from runs/mlp_lag20/latest.pt --iters 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.env.skate_env import attempt_seed  # noqa: E402
from mosca.paths import RUNS  # noqa: E402
from mosca.rl.ppo import ActorCritic, PPOConfig, RunningNorm, compute_gae, ppo_update  # noqa: E402
from mosca.walking.student import N_FEATURES, OutputLatency, features  # noqa: E402
from mosca.walking.teacher import FUTURE_STEPS, straight_trajectory  # noqa: E402
from mosca.walking.vec_env import N_OUT, WalkingVecEnv  # noqa: E402

DT = 0.002
TEST_COMMANDS = [(v, y) for v in (1.0, 2.0, 3.0) for y in (0.0, 1.0, -1.0)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--init-from", default="", help="checkpoint da destilação (m4_distill --student mlp)")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episode-seconds", type=float, default=2.0)
    p.add_argument("--v-min", type=float, default=0.5)
    p.add_argument("--v-max", type=float, default=3.0)
    p.add_argument("--yaw-max", type=float, default=1.0)
    p.add_argument("--delay-ms", type=float, default=None, help="padrão: o do checkpoint de partida")
    p.add_argument("--tau-ms", type=float, default=None, help="padrão: o do checkpoint de partida")
    p.add_argument("--init-std", type=float, default=0.3)
    p.add_argument("--noise-tau", type=float, default=0.02, help="s; ruído de exploração correlacionado")
    p.add_argument("--action-clip", type=float, default=20.0, help="os alvos da professora do flybody vão a ~±10 (os atuadores cortam no próprio limite)")
    p.add_argument("--gamma", type=float, default=0.998)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--target-kl", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--reward-scale", type=float, default=0.02)
    p.add_argument("--w-yaw", type=float, default=0.3)
    p.add_argument("--w-up", type=float, default=0.2)
    p.add_argument("--w-rate", type=float, default=0.01)
    p.add_argument("--critic-warmup", type=int, default=5, help="iterações só do crítico, com a política parada")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-seconds", type=float, default=5.0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


class Walker:
    """Ambiente de caminhada com a latência da professora, a recompensa e o estado do crítico."""

    def __init__(self, n: int, threads: int, delay_steps: int, tau: float, args):
        self.env = WalkingVecEnv(n, n_threads=threads)
        self.lat = OutputLatency(n, N_OUT, delay_steps, tau, DT)
        self.args = args
        self.n = n
        self.ema_v, self.ret, self.v_sum = np.zeros(n), np.zeros(n), np.zeros(n)
        self.prev_u = np.zeros((n, N_OUT))
        self.priv_dim = self.privileged().shape[1]

    def reset(self, v_cmd: np.ndarray, yaw: np.ndarray, seconds: float, headings: np.ndarray) -> np.ndarray:
        steps = round(seconds / DT)
        refs = [straight_trajectory(steps + FUTURE_STEPS + 1, v, yaw_speed=y, heading=h) for v, y, h in zip(v_cmd, yaw, headings)]
        self.env.reset(refs, v_cmd, yaw)
        self.lat.reset(np.zeros(N_OUT))
        self.ema_v = np.zeros(self.n)
        self.prev_u = np.zeros((self.n, N_OUT))
        self.ret = np.zeros(self.n)
        self.v_sum = np.zeros(self.n)
        return self.obs()

    def obs(self) -> np.ndarray:
        return features(self.env.student_obs(), self.env.v_cmd).astype(np.float32)

    def privileged(self) -> np.ndarray:
        e = self.env
        body = []
        for d in e.datas:
            body.append(np.concatenate([[d.xpos[e.thorax][2]], d.xmat[e.thorax][6:9]]))
        body = np.array(body)
        return np.concatenate([body, e.sensor_mean[:, e.velocimeter], e.sensor_mean[:, e.gyro],
                               self.ema_v[:, None], self.lat.state()], axis=1).astype(np.float32)

    def step(self, u: np.ndarray):
        """Aplica o comando `u` (lote, 48) da política; devolve obs, priv, recompensa, queda, fim do tempo."""
        e, a = self.env, self.args
        alive = e.alive.copy()
        y = self.lat(u)
        e.step(e.ctrl_from_student(y), y)
        v_fwd = e.sensor_mean[:, e.velocimeter[0]]
        self.ema_v += (DT / 0.1) * (v_fwd - self.ema_v)
        tent = 1.0 - np.abs(self.ema_v - e.v_cmd) / np.maximum(e.v_cmd, 0.5)
        yaw_err = e.sensor_mean[:, e.gyro[2]] - e.yaw_cmd
        up = np.array([d.xmat[e.thorax][8] for d in e.datas])
        rate = np.mean((u - self.prev_u) ** 2, axis=1)
        reward = np.maximum(tent, -1.0) + a.w_yaw * np.exp(-(yaw_err / 0.5) ** 2) + a.w_up * up - a.w_rate * rate
        reward = np.where(alive, reward, 0.0)
        self.prev_u = u
        self.ret += reward
        self.v_sum += np.where(alive, v_fwd, 0.0)
        term = alive & e.fell
        trunc = alive & ~e.alive & ~e.fell
        return self.obs(), self.privileged(), reward, term, trunc

    def close(self) -> None:
        self.env.close()


def load_init(path: str, ac: ActorCritic) -> dict:
    """Pesos da MLP da destilação (MLPStudent) no ator: mesmas camadas; o viés de saída soma dec_bias."""
    src = torch.load(path, weights_only=False, map_location="cpu")
    pol, cfg = src["policy"], src["controller"]
    with torch.no_grad():
        for k in (0, 2, 4):
            ac.actor[k].weight.copy_(pol[f"mlp.{k}.weight"])
            ac.actor[k].bias.copy_(pol[f"mlp.{k}.bias"])
        ac.actor[4].bias.add_(pol["dec_bias"])
    return cfg


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    run_dir = RUNS / args.run
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    latest = run_dir / "latest.pt"

    ac = ActorCritic(N_FEATURES, 1, N_OUT, init_std=args.init_std)  # dimensão do crítico ajustada abaixo
    init_cfg = {}
    if args.init_from and not (args.resume and latest.exists()):
        init_cfg = load_init(args.init_from, ac)
    delay_ms = args.delay_ms if args.delay_ms is not None else 2.0 * init_cfg.get("lag_steps", 0)
    tau_ms = args.tau_ms if args.tau_ms is not None else 1000.0 * init_cfg.get("tau", 0.0)
    walker = Walker(args.envs, args.threads, round(delay_ms / 2), tau_ms / 1000, args)
    actor_state = ac.actor.state_dict()
    ac = ActorCritic(N_FEATURES, walker.priv_dim, N_OUT, init_std=args.init_std)
    if init_cfg:
        ac.actor.load_state_dict(actor_state)
    ppo_cfg = PPOConfig(gamma=args.gamma, lam=args.lam, target_kl=args.target_kl, lr=args.lr, reward_scale=args.reward_scale)
    opt = torch.optim.Adam(ac.parameters(), lr=ppo_cfg.lr)
    norm_priv = RunningNorm(walker.priv_dim)
    state = {"it": 0, "lr": ppo_cfg.lr, "total_steps": 0, "attempts": 0}
    if args.resume and latest.exists():
        ckpt = torch.load(latest, weights_only=False)
        ac.load_state_dict(ckpt["ac"])
        opt.load_state_dict(ckpt["opt"])
        norm_priv.load_state_dict(ckpt["norm_priv"])
        state = ckpt["state"]
        delay_ms, tau_ms = ckpt["latency"]["delay_ms"], ckpt["latency"]["tau_ms"]
        print(f"retomando da iteração {state['it']}")
    else:
        (run_dir / "config.json").write_text(json.dumps(
            {"args": vars(args), "latency": {"delay_ms": delay_ms, "tau_ms": tau_ms}, "init_controller": init_cfg,
             "priv_dim": walker.priv_dim}, indent=2, default=str), encoding="utf-8")
    beta = float(np.exp(-DT / args.noise_tau)) if args.noise_tau > 0 else 0.0

    def save(path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save({"ac": ac.state_dict(), "opt": opt.state_dict(), "norm_priv": norm_priv.state_dict(), "state": state,
                    "latency": {"delay_ms": delay_ms, "tau_ms": tau_ms}, "args": vars(args)}, tmp)
        tmp.replace(path)

    def evaluate() -> dict:
        n = walker.n
        cmds = [TEST_COMMANDS[j % len(TEST_COMMANDS)] for j in range(n)]
        v, y = np.array([c[0] for c in cmds]), np.array([c[1] for c in cmds])
        obs = walker.reset(v, y, args.eval_seconds, np.zeros(n))
        while walker.env.alive.any():
            with torch.no_grad():
                u = ac.actor(torch.from_numpy(obs)).numpy()
            obs, *_ = walker.step(np.clip(u, -args.action_clip, args.action_clip))
        e = walker.env
        ok = ~e.fell & (e.t >= round(args.eval_seconds / DT) - 1)
        speed = walker.v_sum / np.maximum(e.t, 1)
        return {"eval_success": float(ok.mean()), "eval_falls": float(e.fell.mean()),
                "eval_speed_ratio": float(np.mean(speed / v))}

    n = walker.n
    horizon = round(args.episode_seconds / DT)
    print(f"professora lenta: atraso {delay_ms:g} ms, filtro {tau_ms:g} ms; {sum(p.numel() for p in ac.actor.parameters()):,} "
          f"parâmetros no ator; crítico vê {walker.priv_dim} grandezas a mais")
    print("  it  retorno  dur(s) queda%  vel/pedida    lr  kl_fim ép  s/it")
    for it in range(state["it"], args.iters):
        t0 = time.perf_counter()
        rng = np.random.default_rng(attempt_seed(args.seed, -1 - it))
        v_cmd = rng.uniform(args.v_min, args.v_max, n)
        yaw = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(-args.yaw_max, args.yaw_max, n))
        obs = walker.reset(v_cmd, yaw, args.episode_seconds, rng.uniform(-np.pi, np.pi, n))
        priv = walker.privileged()
        gen = torch.Generator().manual_seed(attempt_seed(args.seed, -1 - it))
        buf = {k: np.zeros((horizon, n, d), np.float32) for k, d in
               (("obs", N_FEATURES), ("priv", walker.priv_dim), ("act", N_OUT), ("mean", N_OUT))}
        for k in ("logp", "val", "rew"):
            buf[k] = np.zeros((horizon, n), np.float32)
        buf["valid"] = np.zeros((horizon, n), bool)
        raw_priv, terminal = [], np.zeros(n, bool)
        last_obs, last_priv = np.zeros((n, N_FEATURES), np.float32), np.zeros((n, walker.priv_dim), np.float32)
        log_std = ac.log_std.detach().clone()
        noise = torch.randn((n, N_OUT), generator=gen)
        steps = 0
        for t in range(horizon):
            valid = walker.env.alive.copy()
            if not valid.any():
                break
            pr = norm_priv(priv).astype(np.float32)
            with torch.no_grad():
                ot, pt = torch.from_numpy(obs), torch.from_numpy(pr)
                dist = ac.dist(ot)
                if t > 0:
                    noise = beta * noise + np.sqrt(1 - beta**2) * torch.randn((n, N_OUT), generator=gen)
                action = dist.mean + dist.stddev * noise
                buf["logp"][t] = dist.log_prob(action).sum(-1).numpy()
                buf["val"][t] = ac.value(ot, pt).numpy()
            buf["obs"][t], buf["priv"][t] = obs, pr
            buf["act"][t], buf["mean"][t] = action.numpy(), dist.mean.numpy()
            buf["valid"][t] = valid
            raw_priv.append(priv[valid])
            obs, priv, reward, term, trunc = walker.step(np.clip(action.numpy(), -args.action_clip, args.action_clip))
            buf["rew"][t] = reward * ppo_cfg.reward_scale
            terminal |= term
            last_obs[trunc], last_priv[trunc] = obs[trunc], priv[trunc]
            steps = t + 1
        with torch.no_grad():
            last_val = ac.value(torch.from_numpy(last_obs), torch.from_numpy(norm_priv(last_priv).astype(np.float32))).numpy()
        adv, ret = compute_gae(buf["rew"][:steps], buf["val"][:steps], buf["valid"][:steps], terminal, last_val,
                               ppo_cfg.gamma, ppo_cfg.lam)
        mask = buf["valid"][:steps]
        batch = {k: torch.from_numpy(buf[k][:steps][mask]) for k in ("obs", "priv", "act", "logp", "mean")}
        batch["adv"] = torch.from_numpy(adv[mask].astype(np.float32))
        batch["ret"] = torch.from_numpy(ret[mask].astype(np.float32))
        batch["log_std"] = log_std
        norm_priv.update(np.concatenate(raw_priv))
        warmup = it < args.critic_warmup
        if warmup:  # só o crítico: a política fica como saiu da destilação
            frozen = [p for p in list(ac.actor.parameters()) + [ac.log_std]]
            for p in frozen:
                p.requires_grad_(False)
        state["lr"], stats = ppo_update(ac, opt, batch, ppo_cfg, state["lr"], gen)
        if warmup:
            for p in frozen:
                p.requires_grad_(True)
            state["lr"] = ppo_cfg.lr
        e = walker.env
        speed = walker.v_sum / np.maximum(e.t, 1)
        state["it"] = it + 1
        state["total_steps"] += int(mask.sum())
        state["attempts"] += n
        metrics = {"it": it, "return": float(walker.ret.mean()), "seconds": float((e.t * DT).mean()),
                   "fell": float(e.fell.mean()), "speed_ratio": float(np.mean(speed / v_cmd)), "lr": state["lr"],
                   "std": float(ac.log_std.detach().exp().mean()), "warmup": warmup, "time": time.perf_counter() - t0, **stats}
        if args.eval_every and (it + 1) % args.eval_every == 0:
            ev = evaluate()
            metrics.update(ev)
            print(f"      teste sem ruído ({args.eval_seconds:g} s): {ev['eval_success']:.0%} sem cair até o fim, "
                  f"velocidade/pedida {ev['eval_speed_ratio']:.2f}", flush=True)
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        print(f"{it:4d} {metrics['return']:8.1f} {metrics['seconds']:6.2f} {100 * metrics['fell']:5.1f} "
              f"{metrics['speed_ratio']:11.2f} {state['lr']:.1e} {stats['kl_final']:.4f} {stats['epochs']:2d} "
              f"{metrics['time']:5.1f}{' (crítico)' if warmup else ''}", flush=True)
        save(latest)
        if (it + 1) % 25 == 0:
            save(run_dir / "checkpoints" / f"it{it + 1:05d}.pt")
    walker.close()


if __name__ == "__main__":
    main()
