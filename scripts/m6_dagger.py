"""M6: o conectoma aprende o gesto de patinar imitando uma MLP patinadora (DAgger), como no M4 com a caminhada.

Professora: uma MLP do m2_train (ação média, com a normalização de observação dela) que patina com os seis patins no
chão, empurra com as patas do meio, desliza, faz curvas e para (mlp_c1b, regras de docs/m6_regras_novas.md). Aluno: o
controlador de conectoma dos patins (o mesmo do m6_es), partindo do conectoma que anda (M4), com ganho por ligação
(sinal fixo) e decodificador anatômico (motor → junta do seu músculo, sinal fixo).

A cada iteração:
1. Coleta: uma fração β das moscas é conduzida pela professora e o resto pelo aluno (β cai de 1 a `--beta-min` em
   `--beta-iters` iterações); nas do aluno, a cada passo a ação executada é a da professora com probabilidade `--mix`
   (caindo a `--mix-min`). Em todas, a professora diz o que faria naquele estado: esse é o alvo do aluno.
2. Treino: trechos de `--chunk` passos sorteados das últimas `--buffer-iters` coletas, cada um recomeçando do estado da
   rede gravado na coleta, com `--burn-in` passos de aquecimento sem gradiente. O erro é o quadrático normalizado por
   saída, depois do filtro dos atuadores (`--filtered-loss`). Com `--mirror`, metade dos trechos vai espelhada (patas
   trocadas em ângulo absoluto, o que é lateral invertido), para o movimento sair simétrico, como no plano e no M4.
3. Teste do aluno sozinho, a cada `--eval-every` iterações, com a média móvel dos parâmetros (`--ema`): os testes do M7
   (critério do M2, curvas de ±0,5 rad/s, parada), mais a fração de patins no chão e o deslize com os seis. O melhor
   teste fica em best.pt.

Uso (no Spark):
    python scripts/m6_dagger.py --run ensina_a --teacher runs/mlp_c1b/checkpoints/it00400.pt \
        --init-from runs/anda_r5I/best_it6.pt --mirror --filtered-loss --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m7_eval import TEST_ATTEMPT_BASE, commands, summarize  # noqa: E402
from mosca.brain.controller import ConnectomePolicy, ControllerConfig, load_walking  # noqa: E402
from mosca.brain.graph import Connectome  # noqa: E402
from mosca.capture import library_versions  # noqa: E402
from mosca.env.skate_env import EnvConfig, RewardConfig, SkateVecEnv, attempt_seed, yaw_schedule  # noqa: E402
from mosca.paths import MALECNS_DIR, RUNS  # noqa: E402
from mosca.rl.ppo import ActorCritic, RunningNorm  # noqa: E402

LEG_SWAP = np.array([1, 0, 3, 2, 5, 4])  # T1_left <-> T1_right, ...


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--teacher", required=True, help="checkpoint da MLP patinadora (m2_train.py)")
    p.add_argument("--init-from", required=True, help="conectoma que anda (m4_distill.py)")
    p.add_argument("--graph", default=str(MALECNS_DIR / "controller_graph_min5.npz"))
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-gpu-mem-gb", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--net-dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--episode-seconds", type=float, default=5.0)
    p.add_argument("--v-min", type=float, default=1.0)
    p.add_argument("--v-max", type=float, default=4.0)
    p.add_argument("--p-stand", type=float, default=0.2)
    p.add_argument("--yaw-max", type=float, default=1.0)
    p.add_argument("--yaw-switch", type=float, default=1.5, help="duração média de cada trecho de giro (s)")
    p.add_argument("--beta-iters", type=int, default=20)
    p.add_argument("--beta-min", type=float, default=0.3)
    p.add_argument("--mix", type=float, default=0.3)
    p.add_argument("--mix-min", type=float, default=0.1)
    p.add_argument("--mix-iters", type=int, default=30)
    p.add_argument("--chunk", type=int, default=16, help="passos de controle (10 ms) por trecho de treino")
    p.add_argument("--burn-in", type=int, default=6)
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--minibatch-chunks", type=int, default=16)
    p.add_argument("--buffer-iters", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dec-lr-mult", type=float, default=10.0)
    p.add_argument("--syn-lr-mult", type=float, default=0.3)
    p.add_argument("--cmd-lr-mult", type=float, default=0.0)
    p.add_argument("--filtered-loss", action="store_true")
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--ema", type=float, default=0.995)
    p.add_argument("--eval-every", type=int, default=3)
    p.add_argument("--save-every", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--tau-scale", type=float, default=0.25)
    p.add_argument("--substeps", type=int, default=5)
    p.add_argument("--turn-gain", type=float, default=1500.0)
    p.add_argument("--turn-cells", default="DNa02")
    p.add_argument("--haltere-scale", type=float, default=0.5)
    p.add_argument("--haltere-offset", type=float, default=0.0)
    p.add_argument("--enc-std", type=float, default=2.0)
    p.add_argument("--dec-gain", type=float, default=0.01)
    return p.parse_args()


def to_dev(x, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


@contextmanager
def swapped(module: torch.nn.Module, params: dict[str, torch.Tensor] | None):
    """Troca temporariamente os parâmetros do módulo pelos de `params` (a média móvel)."""
    if params is None:
        yield
        return
    backup = {k: p.detach().clone() for k, p in module.named_parameters()}
    with torch.no_grad():
        for k, p in module.named_parameters():
            p.copy_(params[k])
    try:
        yield
    finally:
        with torch.no_grad():
            for k, p in module.named_parameters():
                p.copy_(backup[k])


class MLPTeacher:
    """A MLP patinadora: alvo = a ação média dela, cortada nos limites do ambiente."""

    def __init__(self, path: str, env: SkateVecEnv):
        ck = torch.load(path, weights_only=False, map_location="cpu")
        cfg = ck["env_cfg"]
        assert abs(cfg["control_dt"] - env.cfg.control_dt) < 1e-12 and abs(cfg["action_scale"] - env.cfg.action_scale) < 1e-12
        self.ac = ActorCritic(env.obs_dim, env.priv_dim, env.act_dim)
        self.ac.load_state_dict(ck["ac"])
        self.norm = RunningNorm(env.obs_dim)
        self.norm.load_state_dict(ck["norm_obs"])
        self.clip = env.cfg.action_clip

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            a = self.ac.actor(torch.from_numpy(self.norm(obs).astype(np.float32))).numpy()
        return np.clip(a, -self.clip, self.clip).astype(np.float32)


class Mirror:
    """Espelho esquerda-direita da observação (lote, 186) e da ação (lote, 42) dos patins: troca as patas em ângulo
    absoluto (nas patas direitas, os eixos são o espelho dos das esquerdas) e inverte o que é lateral."""

    def __init__(self, env: SkateVecEnv):
        rest = env.rest[0][env.leg_qadr]
        self.q_shift = self.swap(rest[None])[0] - rest
        self.c_shift = self.swap(env.stance_ctrl[None])[0] - env.stance_ctrl
        self.scale = env.cfg.action_scale

    @staticmethod
    def swap(x: np.ndarray) -> np.ndarray:
        return x.reshape(len(x), 6, -1)[:, LEG_SWAP].reshape(len(x), -1)

    def obs(self, o: np.ndarray) -> np.ndarray:
        m = o.copy()
        m[:, 0:42] = self.swap(o[:, 0:42]) + self.q_shift
        m[:, 42:84] = self.swap(o[:, 42:84])
        m[:, 84:126] = self.swap(o[:, 84:126]) + self.c_shift
        m[:, 126:168] = self.swap(o[:, 126:168]) + self.c_shift / self.scale
        m[:, 169] *= -1  # gravidade em y
        m[:, [171, 173]] *= -1  # giroscópio: rolagem e guinada
        m[:, 175] *= -1  # velocidade em y
        m[:, 177:183] = o[:, 177:183][:, LEG_SWAP]  # cargas dos patins
        m[:, 184] *= -1  # giro pedido
        return m

    def act(self, a: np.ndarray) -> np.ndarray:
        return self.swap(a) + self.c_shift / self.scale


def actuator_alpha(env: SkateVecEnv) -> np.ndarray:
    """Fração do caminho até o comando que o filtro dos atuadores percorre num passo de controle."""
    m = env.model
    tau = m.actuator_dynprm[env.leg_act, 0]
    h = m.opt.timestep
    return (1.0 - (1.0 - h / tau) ** env.substeps).astype(np.float32)


def rollout(env, pol, teacher, v_cmd, yaw, driver_teacher, chunk, device, attempts, mix=0.0, rng=None):
    """Uma leva de tentativas; devolve observações, alvos, máscara e os estados da rede no começo de cada trecho."""
    n = env.n
    obs, _ = env.reset(attempts, v_cmd, yaw_cmd=yaw[0])
    T = env.max_steps
    obs_buf = np.zeros((T, n, env.obs_dim), np.float32)
    tgt_buf = np.zeros((T, n, env.act_dim), np.float32)
    valid = np.zeros((T, n), bool)
    states = torch.zeros(((T + chunk - 1) // chunk, pol.n, n), device=device, dtype=pol.initial_state(1).dtype)
    r = pol.initial_state(n)
    vc = to_dev(v_cmd, device)
    steps = 0
    with torch.no_grad():
        for t in range(T):
            if not env.alive.any():
                break
            if t % chunk == 0:
                states[t // chunk] = r
            r, out = pol(r, to_dev(obs, device), vc)
            target = teacher(obs)
            use_teacher = driver_teacher | ((rng.random(n) < mix) if mix > 0 else False)
            action = np.where(use_teacher[:, None], target, out.cpu().numpy())
            obs_buf[t], tgt_buf[t], valid[t] = obs, target, env.alive
            env.set_commands(yaw_cmd=yaw[min(t, len(yaw) - 1)])
            obs, *_ = env.step(action)
            steps = t + 1
    eps = env.episode_stats()
    return {"obs": obs_buf[:steps], "tgt": tgt_buf[:steps], "valid": valid[:steps], "states": states,
            "v_cmd": np.asarray(v_cmd, np.float32), "episodes": eps}


def evaluate(env_eval: SkateVecEnv, pol, device) -> dict:
    """Os testes do M7 com o aluno sozinho (sem professora, sem ruído), a partir do repouso."""
    attempts, speed, yaw = commands()
    obs, _ = env_eval.reset(attempts, speed, yaw_cmd=yaw)
    v = to_dev(speed, device)
    r = pol.initial_state(len(attempts))
    with torch.no_grad():
        while env_eval.alive.any():
            r, out = pol(r, to_dev(obs, device), v)
            obs, *_ = env_eval.step(out.cpu().numpy())
    return summarize(env_eval.episode_stats(), env_eval.cfg.episode_seconds)


def score(s: dict) -> float:
    """Pontuação do teste para escolher o best.pt: o critério do M2 em velocidade, patins no chão, deslize com os seis,
    curvas para os dois lados e parada, menos as quedas."""
    return (s["m2_fast"] + s["m2_grounded"] + s["m2_glide6"] + 0.5 * min(max(s["turn_range"], 0.0), 1.0)
            + 0.5 * s["stand_ok"] - s["m2_falls"] - s["turn_falls"])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", device.index or 0)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_mem_gb * 2**30 / total), device)
    run_dir = RUNS / args.run
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    tcfg = torch.load(args.teacher, weights_only=False, map_location="cpu")["env_cfg"]  # o ambiente da professora
    env_cfg = EnvConfig(n_envs=args.envs, n_threads=args.threads, episode_seconds=args.episode_seconds,
                        control_dt=tcfg["control_dt"], action_scale=tcfg["action_scale"],
                        action_clip=tcfg["action_clip"], seed=args.seed, reward=RewardConfig())
    env = SkateVecEnv(env_cfg)
    env_eval = SkateVecEnv(EnvConfig(**{**env_cfg.__dict__, "n_envs": 60}))
    teacher = MLPTeacher(args.teacher, env)
    mirror = Mirror(env)
    graph = Connectome.load(Path(args.graph))
    cfg = ControllerConfig(control_dt=env_cfg.control_dt, substeps=args.substeps, tau_scale=args.tau_scale,
                           all_synapse_gains=True, haltere_input=True, haltere_scale=args.haltere_scale,
                           haltere_offset=args.haltere_offset, turn_cells=tuple(args.turn_cells.split(",")),
                           turn_gain=args.turn_gain, enc_std=args.enc_std, dec_gain=args.dec_gain, seed=args.seed,
                           net_dtype=args.net_dtype)
    pol = ConnectomePolicy(graph, cfg, device=device)
    mults = {"rede": 1.0, "codificador": 10.0, "tonus": 3.0, "decodificador": args.dec_lr_mult,
             "comando": args.cmd_lr_mult, "exploracao": 0.0, "sinapses": args.syn_lr_mult}
    groups = [{"params": ps, "lr": args.lr * mults.get(name, 1.0), "name": name}
              for name, ps in pol.param_groups().items() if mults.get(name, 1.0) > 0]
    opt = torch.optim.Adam(groups)
    alpha_np = actuator_alpha(env)
    alpha = to_dev(alpha_np, device)
    state = {"it": 0, "best_score": -1e9}
    ema, tgt_std, tgt_std_f = None, None, None
    latest = run_dir / "latest.pt"
    if args.resume and latest.exists():
        ck = torch.load(latest, weights_only=False, map_location=device)
        pol.load_state_dict(ck["policy"])
        opt.load_state_dict(ck["opt"])
        state, tgt_std, tgt_std_f = ck["state"], ck["tgt_std"], ck["tgt_std_f"]
        ema = {k: v.to(device) for k, v in ck["ema"].items()} if ck.get("ema") else None
        print(f"retomando da iteração {state['it']}")
    else:
        copied = load_walking(pol, args.init_from, device)
        print(f"partindo de {args.init_from}: {copied} tensores/ganhos copiados")
        (run_dir / "config.json").write_text(json.dumps(
            {"args": vars(args), "controller": asdict(cfg), "env": {**asdict(env_cfg), "reward": asdict(env_cfg.reward)},
             "neurons": graph.n, "actuator_alpha": alpha_np.round(4).tolist(), "versions": library_versions()},
            indent=2), encoding="utf-8")
    print(f"conectoma: {graph.n:,} neurônios, {sum(p.numel() for p in pol.parameters()):,} parâmetros; {device}; "
          f"filtro dos atuadores por passo {alpha_np.mean():.2f}")
    print("  it   beta   mix  erro (filtrado)  aluno: vel/pedida quedas no-chão  coleta treino")

    buffer: list[dict] = []
    n = env.n
    for it in range(state["it"], args.iters):
        t0 = time.perf_counter()
        rng = np.random.default_rng(attempt_seed(args.seed, it))
        v_cmd = rng.uniform(args.v_min, args.v_max, n)
        stand = rng.random(n) < args.p_stand
        v_cmd[stand] = 0.0
        yaw = yaw_schedule(rng, n, env.max_steps, args.yaw_max, args.yaw_switch / env_cfg.control_dt)
        yaw[:, stand] = 0.0
        beta = max(args.beta_min, 1.0 - it / max(args.beta_iters, 1))
        driver_teacher = np.zeros(n, bool)
        driver_teacher[rng.permutation(n)[: round(beta * n)]] = True
        mix = max(args.mix_min, args.mix * (1.0 - it / max(args.mix_iters, 1))) if args.mix > 0 else 0.0
        attempts = np.arange(it * n, (it + 1) * n)
        ro = rollout(env, pol, teacher, v_cmd, yaw, driver_teacher, args.chunk, device, attempts, mix=mix, rng=rng)
        t_collect = time.perf_counter() - t0
        buffer = (buffer + [ro])[-args.buffer_iters:]
        if tgt_std is None:  # escala de cada saída, fixada na primeira coleta (só a professora)
            tgt_std = np.maximum(ro["tgt"][ro["valid"]].std(axis=0), 0.05).astype(np.float32)
            filt = np.empty_like(ro["tgt"])
            filt[0] = ro["tgt"][0]
            for t in range(1, len(filt)):
                filt[t] = filt[t - 1] + alpha_np * (ro["tgt"][t] - filt[t - 1])
            tgt_std_f = np.maximum(filt[ro["valid"]].std(axis=0), 0.05).astype(np.float32)
        std_t, std_f = to_dev(tgt_std, device), to_dev(tgt_std_f, device)

        t1 = time.perf_counter()
        losses = []
        segs = [(b, k, i) for b, r_ in enumerate(buffer) for k in range(r_["states"].shape[0]) for i in range(n)
                if k * args.chunk < len(r_["valid"]) and r_["valid"][k * args.chunk, i]]
        for _ in range(args.updates):
            pick = [segs[j] for j in rng.choice(len(segs), size=min(args.minibatch_chunks, len(segs)), replace=False)]
            r = torch.stack([buffer[b]["states"][k][:, i] for b, k, i in pick], dim=1)
            flip = rng.random(len(pick)) < 0.5 if args.mirror else np.zeros(len(pick), bool)
            loss_sum, count = 0.0, 0.0
            out_f = tgt_f = None
            for t in range(args.chunk):
                idx = [min(k * args.chunk + t, len(buffer[b]["obs"]) - 1) for b, k, i in pick]
                obs_t = np.stack([buffer[b]["obs"][j, i] for (b, k, i), j in zip(pick, idx)])
                tgt_np = np.stack([buffer[b]["tgt"][j, i] for (b, k, i), j in zip(pick, idx)])
                if flip.any():  # espelhados: o estado inicial da rede é o do trecho original (o aquecimento o renova)
                    obs_t[flip], tgt_np[flip] = mirror.obs(obs_t[flip]), mirror.act(tgt_np[flip])
                v_t = to_dev([buffer[b]["v_cmd"][i] for b, k, i in pick], device)
                tgt_t = to_dev(tgt_np, device)
                with torch.set_grad_enabled(t >= args.burn_in):
                    r, out = pol(r, to_dev(obs_t, device), v_t)
                    out_f = out if out_f is None else out_f + alpha * (out - out_f)
                tgt_f = tgt_t if tgt_f is None else tgt_f + alpha * (tgt_t - tgt_f)
                if t < args.burn_in:
                    continue
                mask = np.array([k * args.chunk + t < len(buffer[b]["valid"]) and buffer[b]["valid"][k * args.chunk + t, i]
                                 for b, k, i in pick], dtype=np.float32)
                m_t = to_dev(mask, device)
                err = ((out_f - tgt_f) / std_f) if args.filtered_loss else ((out - tgt_t) / std_t)
                loss_sum = loss_sum + (err.pow(2).mean(dim=1) * m_t).sum()
                count += mask.sum()
            loss = loss_sum / max(count, 1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
            if args.ema > 0:
                with torch.no_grad():
                    if ema is None:
                        ema = {k: p.detach().clone() for k, p in pol.named_parameters()}
                    for k, p in pol.named_parameters():
                        ema[k].lerp_(p.detach(), 1.0 - args.ema)
            losses.append(loss.item())
        t_train = time.perf_counter() - t1

        student = ~driver_teacher
        eps = ro["episodes"]
        moving = student & (v_cmd > 0)
        ratio = float(np.mean([eps[i]["speed"] / v_cmd[i] for i in np.flatnonzero(moving)])) if moving.any() else float("nan")
        falls = float(np.mean([eps[i]["fell"] for i in np.flatnonzero(student)])) if student.any() else float("nan")
        grounded = float(np.mean([eps[i]["grounded_frac"] for i in np.flatnonzero(student)])) if student.any() else float("nan")
        metrics = {"it": it, "beta": beta, "mix": mix, "loss": float(np.mean(losses)), "student_speed_ratio": ratio,
                   "student_falls": falls, "student_grounded": grounded, "time_collect": t_collect, "time_train": t_train}
        if args.eval_every and (it + 1) % args.eval_every == 0:
            with swapped(pol, ema):
                ev = evaluate(env_eval, pol, device)
            metrics.update({f"eval_{k}": v for k, v in ev.items()})
            s = score(ev)
            best = s > state["best_score"]
            print(f"      teste do aluno sozinho: M2 {'OK' if ev['m2_ok'] else 'não'} ({ev['m2_fast']:.0%} ≥3 cm/s, "
                  f"{ev['m2_speed']:.2f} cm/s, quedas {ev['m2_falls']:.0%}), patins no chão {ev['m2_grounded']:.0%}, "
                  f"desliza com os seis {ev['m2_glide6']:.0%}, curvas {ev['turn_left']:+.2f}/{ev['turn_right']:+.2f} rad/s, "
                  f"parada {ev['stand_ok']:.0%}{' (melhor até aqui)' if best else ''}", flush=True)
            if best:
                state["best_score"], state["best_it"] = s, it + 1
                policy = pol.state_dict()
                if ema is not None:
                    policy.update({k: v.detach().clone() for k, v in ema.items()})
                torch.save({"policy": policy, "state": dict(state), "controller": asdict(cfg),
                            "env_cfg": {**asdict(env_cfg), "reward": asdict(env_cfg.reward)}, "args": {**vars(args),
                            "graph": args.graph}, "eval": ev}, run_dir / "best.pt")
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        print(f"{it:4d} {beta:6.2f} {mix:5.2f} {metrics['loss']:13.3f} {ratio:20.2f} {falls:6.2f} {grounded:8.2f} "
              f"{t_collect:7.1f} {t_train:6.1f}", flush=True)
        state["it"] = it + 1
        ck = {"policy": pol.state_dict(), "opt": opt.state_dict(), "state": state, "tgt_std": tgt_std,
              "tgt_std_f": tgt_std_f, "controller": asdict(cfg), "args": vars(args), "ema": ema,
              "env_cfg": {**asdict(env_cfg), "reward": asdict(env_cfg.reward)}}
        torch.save(ck, latest.with_suffix(".tmp"))
        latest.with_suffix(".tmp").replace(latest)
        if args.save_every and (it + 1) % args.save_every == 0:
            policy = pol.state_dict()
            if ema is not None:
                policy.update({k: v.detach().clone() for k, v in ema.items()})
            torch.save({"policy": policy, "state": dict(state), "controller": asdict(cfg),
                        "env_cfg": {**asdict(env_cfg), "reward": asdict(env_cfg.reward)}, "args": vars(args)},
                       run_dir / "checkpoints" / f"it{it + 1:05d}.pt")
    env.close()
    env_eval.close()


if __name__ == "__main__":
    main()
