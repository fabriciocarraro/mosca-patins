"""O conectoma aprende a patinar por PPO (direto, ou partindo do conectoma que anda, `--init-from`).

Mesmo ambiente, recompensa, currículo e exploração da MLP (scripts/m2_train.py). O ator é o
controlador de conectoma (mosca.brain.controller): sem camadas escondidas, só a fiação do
MaleCNS entre os proprioceptores e o comando, de um lado, e os neurônios motores, do outro.
O estado da rede passa de um passo de controle para o outro; na atualização, as tentativas
são cortadas em trechos de `--chunk` passos (mosca.rl.recurrent_ppo). O crítico é uma MLP
que vê o estado completo do simulador, como na MLP.

A rede roda na GPU (`--device cuda`), com a memória limitada por `--max-gpu-mem-gb`; a
física continua na CPU, em threads.

Partindo do conectoma que anda (M4, `--init-from runs/NOME/best.pt`): a rede, os codificadores
(patas e halteres), o tônus e os ganhos de comando vêm do checkpoint; no decodificador, cada
ligação neurônio motor → junta que existe nos dois corpos herda o ganho (o tendão longo, que
na caminhada comanda a adesão, volta a abaixar o tarso); o viés é recalibrado na postura de
patinação. Use a mesma configuração do controlador da caminhada (τ, ganhos por ligação,
halteres) e subpassos de RK4 menores que τ (com τ de 5 ms, `--substeps 5`: 2 ms por subpasso).

Com `--capture`, cada iteração grava em runs/<nome>/capture/ as ações de todas as tentativas
(re-simuláveis bit a bit por scripts/replay_attempt.py) e a versão da política usada na coleta.

Uso (no Spark):
    python scripts/connectome_train.py --run conectoma_a --iters 300 --device cuda
    python scripts/connectome_train.py --run conectoma_a --iters 600 --device cuda --resume
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
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.brain.controller import ConnectomePolicy, ControllerConfig  # noqa: E402
from mosca.capture import Recorder, library_versions  # noqa: E402
from mosca.brain.graph import Connectome  # noqa: E402
from mosca.env.skate_env import EnvConfig, RewardConfig, SkateVecEnv, attempt_seed  # noqa: E402
from mosca.paths import MALECNS_DIR, RUNS  # noqa: E402
from mosca.rl.ppo import RunningNorm, compute_gae  # noqa: E402
from mosca.rl.recurrent_ppo import Critic, RecurrentPPOConfig, Rollout, recurrent_ppo_update  # noqa: E402

TEST_ATTEMPT_BASE = 10**9
REWARD_FLAGS = ("w_vel", "w_yaw", "sigma_yaw", "w_up", "w_roll", "sigma_roll", "w_slip", "w_rate", "w_leg_floor")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-gpu-mem-gb", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--graph", default=str(MALECNS_DIR / "controller_graph_min5.npz"))
    p.add_argument("--episode-seconds", type=float, default=5.0)
    p.add_argument("--v-start", type=float, default=1.0)
    p.add_argument("--v-final", type=float, default=4.0)
    p.add_argument("--v-step", type=float, default=0.5)
    p.add_argument("--push-iters", type=int, default=200)
    p.add_argument("--action-scale", type=float, default=0.15)
    p.add_argument("--action-clip", type=float, default=3.0)
    p.add_argument("--noise-tau", type=float, default=0.05)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--critic-lr", type=float, default=3e-4, help="taxa fixa do crítico (a do ator se ajusta pela KL)")
    p.add_argument("--enc-lr-mult", type=float, default=10.0, help="multiplicador da taxa do codificador")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--minibatch-chunks", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-speed", type=float, default=3.0)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--capture", action="store_true", help="grava as tentativas para re-simulação (M5/M6)")
    p.add_argument("--init-from", default="", help="checkpoint do conectoma que anda (m4_distill.py)")
    p.add_argument("--substeps", type=int, default=2, help="subpassos de RK4 por passo de controle (10 ms)")
    p.add_argument("--all-synapse-gains", action="store_true", help="ganho treinável por ligação (degrau 2)")
    p.add_argument("--haltere-input", action="store_true", help="giroscópio do tórax nos aferentes dos halteres")
    p.add_argument("--haltere-scale", type=float, default=2.0)
    p.add_argument("--haltere-offset", type=float, default=None)
    p.add_argument("--turn-cells", default="DNa01,DNa02")
    p.add_argument("--net-dtype", choices=("float32", "float64"), default="float32",
                   help="precisão da rede (float64: a atualização re-executa a mesma política da coleta)")
    p.add_argument("--syn-lr-mult", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=3e-4, help="taxa-base inicial do ator (ajustada pela KL)")
    p.add_argument("--mults", default="", help='multiplicadores por grupo, "rede=0.85,decodificador=1.8,..." '
                                              "(medidos pelo m6_sensitivity.py); sobrepõem os padrões")
    p.add_argument("--cmd-lr-mult", type=float, default=10.0)
    cc = ControllerConfig()
    for name in ("size_ref", "walk_gain", "turn_gain", "enc_std", "prop_offset", "motor_tone", "dec_gain", "init_std",
                 "tau_scale"):
        p.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(cc, name))
    rc = RewardConfig(w_yaw=0.3, sigma_yaw=0.5)  # giro mais punido que na MLP (que virava ~25°/s)
    for name in REWARD_FLAGS:
        p.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(rc, name))
    return p.parse_args()


def to_dev(x, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def evaluate(env, pol, speed: float) -> dict:
    """Testes fixos, sem ruído, a partir do repouso."""
    obs, _ = env.reset(TEST_ATTEMPT_BASE + np.arange(env.n), np.full(env.n, speed))
    r = pol.initial_state(env.n)
    v_cmd = to_dev(np.full(env.n, speed), pol.device)
    with torch.no_grad():
        while env.alive.any():
            r, mean = pol(r, to_dev(obs, pol.device), v_cmd)
            obs, *_ = env.step(mean.cpu().numpy())
    eps = env.episode_stats()
    return {f"eval_{k}": float(np.nanmean([e[k] for e in eps]))
            for k in ("speed", "rolling", "glide_frac", "fell", "seconds", "grounded_frac", "cot")}


def load_walking(pol: ConnectomePolicy, path: str, device) -> int:
    """Copia do conectoma que anda o que vale nos dois corpos (ver a documentação do módulo)."""
    src = torch.load(path, weights_only=False, map_location=device)
    weights = dict(src["policy"])
    if src.get("ema"):
        weights.update({k: v.to(device) for k, v in src["ema"].items()})
    own = pol.state_dict()
    fixed = {"net.tau0", "net.a0", "net.theta0", "net.r_max"}
    skip = fixed | {k for k in own if k.startswith("dec_") or k == "log_std"}
    copied = {k: v for k, v in weights.items() if k in own and own[k].shape == v.shape and k not in skip}
    own.update(copied)
    pol.load_state_dict(own)
    # Decodificador: ganho de cada ligação (motor, saída) presente nos dois corpos.
    walk_pairs = {(int(m), int(o)): i for i, (m, o) in enumerate(zip(weights["dec_motor"].tolist(), weights["dec_out"].tolist()))}
    n_dec = 0
    with torch.no_grad():
        for i, (m, o) in enumerate(zip(pol.dec_motor.tolist(), pol.dec_out.tolist())):
            j = walk_pairs.get((int(m), int(o)))
            if j is not None and bool(pol.dec_free[i]) == bool(weights["dec_free"][j]):
                pol.dec_raw[i] = weights["dec_raw"][j]
                n_dec += 1
    return len(copied) + n_dec


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", device.index or 0)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_mem_gb * 2**30 / total), device)
    run_dir = RUNS / args.run
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "capture").mkdir(exist_ok=True)

    reward_cfg = RewardConfig(**{name: getattr(args, name) for name in REWARD_FLAGS})
    env_cfg = EnvConfig(n_envs=args.envs, n_threads=args.threads, episode_seconds=args.episode_seconds,
                        action_scale=args.action_scale, action_clip=args.action_clip, seed=args.seed, reward=reward_cfg)
    ctrl_cfg = ControllerConfig(size_ref=args.size_ref, walk_gain=args.walk_gain, turn_gain=args.turn_gain,
                                enc_std=args.enc_std, prop_offset=args.prop_offset, motor_tone=args.motor_tone,
                                dec_gain=args.dec_gain, init_std=args.init_std, control_dt=env_cfg.control_dt,
                                substeps=args.substeps, seed=args.seed, tau_scale=args.tau_scale,
                                all_synapse_gains=args.all_synapse_gains, haltere_input=args.haltere_input,
                                haltere_scale=args.haltere_scale, haltere_offset=args.haltere_offset,
                                turn_cells=tuple(args.turn_cells.split(",")), net_dtype=args.net_dtype)
    ppo_cfg = RecurrentPPOConfig(target_kl=args.target_kl, epochs=args.epochs, chunk=args.chunk, lr=args.lr,
                                 minibatch_chunks=args.minibatch_chunks)
    env = SkateVecEnv(env_cfg)
    graph = Connectome.load(Path(args.graph))
    pol = ConnectomePolicy(graph, ctrl_cfg, device=device)
    critic = Critic(env.obs_dim, env.priv_dim).to(device)
    # Taxas relativas por grupo, medidas pela sensibilidade da ação média a cada grupo: o
    # codificador (proprioceptores abaixo do limiar) é ~40 vezes menos sensível que a rede.
    mults = {"rede": 1.0, "codificador": args.enc_lr_mult, "tonus": 3.0, "decodificador": 1.0,
             "comando": args.cmd_lr_mult, "exploracao": 1.0, "sinapses": args.syn_lr_mult}
    for item in filter(None, args.mults.split(",")):
        name, value = item.split("=")
        mults[name] = float(value)
    groups = [{"params": ps, "lr": ppo_cfg.lr * mults[name], "mult": mults[name], "name": name}
              for name, ps in pol.param_groups().items()]
    assert sum(p.numel() for g in groups for p in g["params"]) == sum(p.numel() for p in pol.parameters())
    opt = torch.optim.Adam(groups + [{"params": critic.parameters(), "lr": args.critic_lr, "name": "critic"}])
    norm_obs, norm_priv = RunningNorm(env.obs_dim), RunningNorm(env.priv_dim)
    state = {"it": 0, "v_max": args.v_start, "lr": ppo_cfg.lr, "total_steps": 0, "attempts": 0}
    beta = float(np.exp(-env_cfg.control_dt / args.noise_tau)) if args.noise_tau > 0 else 0.0

    latest = run_dir / "latest.pt"
    if args.resume and latest.exists():
        ckpt = torch.load(latest, weights_only=False, map_location=device)
        pol.load_state_dict(ckpt["policy"])
        critic.load_state_dict(ckpt["critic"])
        opt.load_state_dict(ckpt["opt"])
        norm_obs.load_state_dict(ckpt["norm_obs"])
        norm_priv.load_state_dict(ckpt["norm_priv"])
        state = ckpt["state"]
        print(f"retomando da iteração {state['it']}")
    else:
        if args.init_from:
            print(f"partindo de {args.init_from}: {load_walking(pol, args.init_from, device)} tensores/ganhos copiados")
        # Saída de repouso = postura canônica: calibra com a mosca parada, no meio da faixa de comandos.
        obs_rest, _ = env.reset(np.arange(env.n) + 10**8, np.full(env.n, args.v_start))
        pol.calibrate_rest(to_dev(obs_rest[:8], device), 0.65 * args.v_start)
        (run_dir / "config.json").write_text(json.dumps(
            {"args": vars(args), "env": asdict(env_cfg), "controller": asdict(ctrl_cfg), "ppo": asdict(ppo_cfg),
             "neurons": graph.n, "trainable": sum(p.numel() for p in pol.parameters()),
             "versions": library_versions()}, indent=2), encoding="utf-8")

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save({"policy": pol.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(),
                    "norm_obs": norm_obs.state_dict(), "norm_priv": norm_priv.state_dict(), "state": state,
                    "env_cfg": asdict(env_cfg), "controller": asdict(ctrl_cfg), "args": vars(args)}, tmp)
        tmp.replace(path)

    n, horizon, chunk = env.n, env.max_steps, ppo_cfg.chunk
    print(f"conectoma: {graph.n:,} neurônios, {sum(p.numel() for p in pol.parameters()):,} parâmetros treináveis; "
          f"dispositivo {device}")
    print("  it   passos  retorno  dur(s) queda%  vel   rol  desliza v_max sucesso    lr  kl_fim ép coleta atualiza")
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
        recorder = Recorder(env, it, attempts, v_cmd, push) if args.capture else None
        if args.capture:  # versão da política desta coleta (antes da atualização)
            torch.save(pol.state_dict(), run_dir / "capture" / f"policy_it{it:05d}.pt")

        T, A = horizon, env.act_dim
        buf = {"obs": torch.zeros(T, n, env.obs_dim, device=device),
               "obs_norm": torch.zeros(T, n, env.obs_dim, device=device),
               "priv_norm": torch.zeros(T, n, env.priv_dim, device=device),
               "act": torch.zeros(T, n, A, device=device), "mean": torch.zeros(T, n, A, device=device),
               "logp": torch.zeros(T, n, device=device)}
        states = torch.zeros((T + chunk - 1) // chunk, graph.n, n, device=device, dtype=pol.initial_state(1).dtype)
        rew = np.zeros((T, n), np.float32)
        val = np.zeros((T, n), np.float32)
        valid = np.zeros((T, n), bool)
        raw_obs, raw_priv = [], []
        terminal = np.zeros(n, bool)
        last_obs, last_priv = np.zeros((n, env.obs_dim)), np.zeros((n, env.priv_dim))
        log_std = pol.log_std.detach().clone()
        vc = to_dev(v_cmd, device)
        r = pol.initial_state(n)
        noise = torch.randn((n, A), generator=gen).to(device)
        steps = 0
        activity = []  # (fração de neurônios ativos, taxa média dos motores) a cada 50 passos
        with torch.no_grad():
            for t in range(T):
                alive = env.alive.copy()
                if not alive.any():
                    break
                if t % chunk == 0:
                    states[t // chunk] = r
                o, on, pn = to_dev(obs, device), to_dev(norm_obs(obs), device), to_dev(norm_priv(priv), device)
                r, mean = pol(r, o, vc)
                if t % 50 == 0:
                    live = torch.as_tensor(alive, device=device)
                    activity.append(((r[:, live] > 0.01).float().mean().item(), r[pol.motors][:, live].mean().item()))
                if t > 0:  # ruído correlacionado, variância 1 mantida
                    noise = beta * noise + np.sqrt(1 - beta**2) * torch.randn((n, A), generator=gen).to(device)
                std = log_std.exp()
                action = mean + std * noise
                buf["logp"][t] = Normal(mean, std.expand_as(mean)).log_prob(action).sum(-1)
                val[t] = critic(on, pn).cpu().numpy()
                buf["obs"][t], buf["obs_norm"][t], buf["priv_norm"][t] = o, on, pn
                buf["act"][t], buf["mean"][t] = action, mean
                valid[t] = alive
                raw_obs.append(obs[alive])
                raw_priv.append(priv[alive])
                action_np = action.cpu().numpy()
                if recorder is not None:
                    recorder.before_step(action_np)
                obs, priv, reward, term, trunc = env.step(action_np)
                rew[t] = reward * ppo_cfg.reward_scale
                terminal |= term
                last_obs[trunc], last_priv[trunc] = obs[trunc], priv[trunc]
                steps = t + 1
            last_val = critic(to_dev(norm_obs(last_obs), device), to_dev(norm_priv(last_priv), device)).cpu().numpy()
        t_collect = time.perf_counter() - t0
        if recorder is not None:
            recorder.finish().save(run_dir / "capture" / f"it{it:05d}.npz")

        adv, ret = compute_gae(rew[:steps], val[:steps], valid[:steps], terminal, last_val, ppo_cfg.gamma, ppo_cfg.lam)
        ro = Rollout(obs=buf["obs"][:steps], obs_norm=buf["obs_norm"][:steps], priv_norm=buf["priv_norm"][:steps],
                     act=buf["act"][:steps], logp=buf["logp"][:steps], mean=buf["mean"][:steps],
                     valid=to_dev(valid[:steps], device, torch.bool), v_cmd=vc, states=states,
                     adv=to_dev(adv, device), ret=to_dev(ret, device), log_std=log_std)
        norm_obs.update(np.concatenate(raw_obs))
        norm_priv.update(np.concatenate(raw_priv))
        t1 = time.perf_counter()
        state["lr"], stats = recurrent_ppo_update(pol, critic, opt, ro, ppo_cfg, state["lr"], gen)
        t_update = time.perf_counter() - t1
        del ro, buf, states

        episodes = env.episode_stats()
        ok = [e for e in episodes if not e["fell"] and e["steps"] >= horizon
              and abs(e["speed"] - e["v_cmd"]) < max(0.5, 0.25 * e["v_cmd"])]
        success = len(ok) / n
        if success >= 0.8:
            state["v_max"] = min(args.v_final, v_max + args.v_step)
        state["it"] = it + 1
        state["total_steps"] += int(valid[:steps].sum())
        state["attempts"] += n

        def mean_of(key):
            return float(np.mean([e[key] for e in episodes]))

        metrics = {"it": it, "steps": state["total_steps"], "attempts": state["attempts"], "v_max": v_max,
                   "p_push": p_push, "return": mean_of("return"), "seconds": mean_of("seconds"), "fell": mean_of("fell"),
                   "speed": mean_of("speed"), "rolling": mean_of("rolling"), "glide_frac": mean_of("glide_frac"),
                   "grounded_frac": mean_of("grounded_frac"), "success": success, "lr": state["lr"],
                   "std": float(pol.log_std.detach().exp().mean()), "walk_gain": pol.log_walk_gain.exp().item(),
                   "time_collect": t_collect, "time_update": t_update,
                   "active_frac": float(np.mean([a for a, _ in activity])), "motor_rate": float(np.mean([m for _, m in activity])),
                   **stats}
        if args.eval_every and (it + 1) % args.eval_every == 0:
            eval_speed = min(args.eval_speed, state["v_max"])
            ev = evaluate(env, pol, eval_speed)
            metrics.update(ev, eval_v_cmd=eval_speed)
            print(f"      avaliação sem ruído a {eval_speed:g} cm/s: vel {ev['eval_speed']:.2f}, rolamento "
                  f"{ev['eval_rolling']:.2f}, desliza {ev['eval_glide_frac']:.2f}, quedas {ev['eval_fell']:.0%}", flush=True)
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        with open(run_dir / "attempts.jsonl", "a", encoding="utf-8") as f:
            for e in episodes:
                f.write(json.dumps({"it": it, **e}) + "\n")
        print(f"{it:4d} {state['total_steps']:8d} {metrics['return']:8.1f} {metrics['seconds']:6.2f} "
              f"{100 * metrics['fell']:5.1f} {metrics['speed']:5.2f} {metrics['rolling']:5.2f} "
              f"{metrics['glide_frac']:6.2f} {v_max:5.1f} {success:6.2f} {state['lr']:.1e} {stats['kl_final']:.4f} "
              f"{stats['epochs']:2d} {t_collect:6.1f} {t_update:8.1f}", flush=True)
        save(latest)
        if (it + 1) % args.save_every == 0:
            save(run_dir / "checkpoints" / f"it{it + 1:05d}.pt")
    env.close()


if __name__ == "__main__":
    main()
