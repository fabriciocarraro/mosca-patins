"""M6: o conectoma aprende a patinar por estratégias evolutivas (OpenAI-ES), sem gradiente pela rede.

Partindo do conectoma que anda (`--init-from`), cada geração roda uma leva de tentativas em que cada
ambiente é uma variação da mosca: fatores por tipo celular, tônus dos motores, vieses do codificador e
decodificador anatômico perturbados (+σε e −σε, pares antitéticos); os ganhos por ligação ficam os do
conectoma que anda. As variações que patinam melhor (retorno da tentativa, a mesma recompensa do PPO)
puxam a média da geração seguinte. A ação é a média da política, sem ruído: a exploração é nos parâmetros.
Currículo de velocidade como no PPO (a faixa pedida abre com 80% de acerto), julgado pela avaliação da
média (a população perturbada acerta pouco).

`--probe`: só o diagnóstico — sorteia uma geração em cada escala de `--probe-scales` (múltiplos dos σ)
e mostra se alguma variação sai do ponto fixo (velocidade) em comparação com a média, sem atualizar nada.

Uso (no Spark):
    python scripts/m6_es.py --run evolui_a --init-from runs/anda_r5I/best_it6.pt --probe --device cuda
    python scripts/m6_es.py --run evolui_a --init-from runs/anda_r5I/best_it6.pt --generations 500 --device cuda
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
from mosca.brain.controller import ConnectomePolicy, ControllerConfig, load_walking  # noqa: E402
from mosca.brain.graph import Connectome  # noqa: E402
from mosca.capture import Recorder, library_versions  # noqa: E402
from mosca.env.skate_env import (EnvConfig, RewardConfig, SkateVecEnv, attempt_seed, command_success,  # noqa: E402
                                 yaw_schedule)
from mosca.paths import MALECNS_DIR, RUNS  # noqa: E402
from mosca.rl.es import PopulationES  # noqa: E402

TEST_ATTEMPT_BASE = 10**9
REWARD_FLAGS = ("w_vel", "w_yaw", "sigma_yaw", "w_up", "w_roll", "sigma_roll", "w_slip", "w_rate", "w_leg_floor",
                "w_glide", "w_sym", "w_cot", "w_contact")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--init-from", default="", help="conectoma que anda (m4_distill.py)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--generations", type=int, default=500)
    p.add_argument("--envs", type=int, default=64, help="tamanho da população (par)")
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-gpu-mem-gb", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--graph", default=str(MALECNS_DIR / "controller_graph_min5.npz"))
    p.add_argument("--net-dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--deterministic", action="store_true",
                   help="controlador determinístico (exige --net-dtype float64): a re-simulação do cérebro reproduz a coleta")
    p.add_argument("--capture", action="store_true",
                   help="grava cada tentativa (ações, estados) e a média dos parâmetros de cada geração (M5/M6)")
    p.add_argument("--episode-seconds", type=float, default=5.0)
    p.add_argument("--control-dt", type=float, default=0.01)
    p.add_argument("--action-scale", type=float, default=0.15)
    p.add_argument("--action-clip", type=float, default=3.0)
    p.add_argument("--v-start", type=float, default=1.0)
    p.add_argument("--v-final", type=float, default=4.0)
    p.add_argument("--v-step", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.3, help="passo relativo: a média anda ~lr·σ por geração")
    p.add_argument("--lr-warm", type=float, default=0.0,
                   help="passo relativo nas primeiras --lr-warm-gens gerações (sair do ponto fixo depressa)")
    p.add_argument("--lr-warm-gens", type=int, default=0)
    p.add_argument("--sigma-type", type=float, default=0.05, help="σ dos log-fatores por tipo celular (a, θ, τ)")
    p.add_argument("--sigma-tone", type=float, default=1.0, help="σ do tônus dos motores (corrente)")
    p.add_argument("--sigma-enc", type=float, default=0.5, help="σ dos vieses do codificador e dos halteres (corrente)")
    p.add_argument("--sigma-dec-raw", type=float, default=0.1, help="σ dos log-ganhos do decodificador")
    p.add_argument("--sigma-dec-bias", type=float, default=0.05, help="σ do viés do decodificador (ação)")
    p.add_argument("--sigma-walk", type=float, default=0.2, help="σ do log-ganho do comando de velocidade (DNg100)")
    p.add_argument("--sigma-turn", type=float, default=0.2, help="σ do log-ganho do comando de giro (DNa02)")
    p.add_argument("--turn-side-gains", action="store_true", help="ganho de giro separado para o lado direito")
    p.add_argument("--neuron-theta", action="store_true",
                   help="fator de limiar por neurônio também evoluído (ajustes diferentes em cada lado)")
    p.add_argument("--sigma-neuron", type=float, default=0.05, help="σ do log-fator de limiar por neurônio")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-speed", type=float, default=4.0, help="teto da velocidade da avaliação (padrão: a do currículo)")
    p.add_argument("--phase2-gen", type=int, default=0,
                   help="geração em que entram os pedidos de parada e as curvas (antes, só retas: o deslize aparece "
                        "primeiro; depois, cedo o bastante para não perder a via direita do DNa02)")
    p.add_argument("--push-gens", type=int, default=0,
                   help="empurrão inicial do currículo (como no m2_train): até --push-max dos pares começa já andando, "
                        "fração que cai a zero em tantas gerações (0 = sem empurrão); os testes partem do repouso")
    p.add_argument("--push-max", type=float, default=0.5)
    p.add_argument("--push-from", type=int, default=0, help="geração em que o empurrão começa (para ramificações)")
    p.add_argument("--p-stand", type=float, default=0.0,
                   help="fração dos pares que pede velocidade 0 (DNg100 sem corrente): calar o DNg100 passa a parar a mosca")
    p.add_argument("--yaw-final", type=float, default=0.0,
                   help="giro máximo pedido no fim do currículo (rad/s, só pelo DNa02); abre depois de --v-final")
    p.add_argument("--yaw-step", type=float, default=0.25)
    p.add_argument("--yaw-start", type=float, default=0.0, help="giro máximo no começo (rad/s)")
    p.add_argument("--yaw-switch", type=float, default=1.5, help="duração média de cada trecho de giro (s)")
    p.add_argument("--yaw-tol", type=float, default=0.5, help="erro médio de giro filtrado para a tentativa contar (rad/s)")
    p.add_argument("--eval-yaw", type=float, default=1.0, help="giro das curvas da avaliação (rad/s)")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--count-lifted", action="store_true",
                   help="rolamento pela média dos seis patins (no ar vale 0) e deslize só com os seis no chão")
    p.add_argument("--yaw-gated", action="store_true",
                   help="a recompensa de giro só paga na proporção em que a mosca anda na velocidade pedida")
    p.add_argument("--yaw-filtered", action="store_true",
                   help="recompensa de giro pelo giro médio de 200 ms (o instantâneo é dominado pelo balanço)")
    p.add_argument("--probe", action="store_true", help="só o diagnóstico de sensibilidade")
    p.add_argument("--probe-scales", default="0.5,1,2,4")
    p.add_argument("--probe-speed", type=float, default=1.5)
    # Controlador: os mesmos do conectoma que anda (M4) e do PPO de patins.
    p.add_argument("--substeps", type=int, default=5)
    p.add_argument("--tau-scale", type=float, default=0.25)
    p.add_argument("--turn-gain", type=float, default=1500.0)
    p.add_argument("--enc-std", type=float, default=2.0)
    p.add_argument("--dec-gain", type=float, default=0.01)
    p.add_argument("--haltere-scale", type=float, default=0.5)
    p.add_argument("--haltere-offset", type=float, default=0.0)
    p.add_argument("--turn-cells", default="DNa02")
    rc = RewardConfig(w_yaw=0.3, sigma_yaw=0.5)
    for name in REWARD_FLAGS:
        p.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(rc, name))
    return p.parse_args()


def run_episodes(env, pol, attempts, v_cmd, offsets=None, record: int | None = None, yaw: np.ndarray | None = None,
                 push: np.ndarray | None = None):
    """Uma leva de tentativas com a ação média; `offsets`: uma variação da mosca por ambiente; `yaw`:
    giro pedido a cada passo (passos, lote); `push`: velocidade inicial de cada tentativa (empurrão do currículo).
    Com `record` (nº da geração), devolve também a captura da leva."""
    pol.set_population(offsets)
    push = np.zeros(env.n) if push is None else push
    obs, _ = env.reset(attempts, v_cmd, push, yaw_cmd=None if yaw is None else yaw[0])
    recorder = None if record is None else Recorder(env, record, attempts, v_cmd, push)
    r = pol.initial_state(env.n)
    v = torch.as_tensor(v_cmd, dtype=torch.float32, device=pol.device)
    t = 0
    with torch.no_grad():
        while env.alive.any():
            r, mean = pol(r, torch.as_tensor(obs, dtype=torch.float32, device=pol.device), v)
            action = mean.cpu().numpy()
            if yaw is not None:
                env.set_commands(yaw_cmd=yaw[min(t, len(yaw) - 1)])
            t += 1
            if recorder is not None:
                recorder.before_step(action)
            obs, *_ = env.step(action)
    pol.set_population(None)
    episodes = env.episode_stats()
    return (episodes, recorder.finish() if recorder is not None else None) if record is not None else episodes


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
    reward_cfg = RewardConfig(**{name: getattr(args, name) for name in REWARD_FLAGS}, yaw_filtered=args.yaw_filtered,
                              yaw_gated=args.yaw_gated, count_lifted=args.count_lifted)
    env_cfg = EnvConfig(n_envs=args.envs, n_threads=args.threads, episode_seconds=args.episode_seconds,
                        control_dt=args.control_dt, action_scale=args.action_scale, action_clip=args.action_clip,
                        seed=args.seed, reward=reward_cfg)
    ctrl_cfg = ControllerConfig(control_dt=args.control_dt, substeps=args.substeps, tau_scale=args.tau_scale,
                                all_synapse_gains=True, haltere_input=True, haltere_scale=args.haltere_scale,
                                haltere_offset=args.haltere_offset, turn_cells=tuple(args.turn_cells.split(",")),
                                turn_gain=args.turn_gain, enc_std=args.enc_std, dec_gain=args.dec_gain,
                                seed=args.seed, net_dtype=args.net_dtype, deterministic=args.deterministic,
                                turn_side_gains=args.turn_side_gains, neuron_theta=args.neuron_theta)
    env = SkateVecEnv(env_cfg)
    pol = ConnectomePolicy(Connectome.load(Path(args.graph)), ctrl_cfg, device=device)
    for q in pol.parameters():
        q.requires_grad_(False)
    params = pol.es_parameters()
    for q in params.values():
        q.requires_grad_(True)
    sig = {"log_a": args.sigma_type, "log_theta": args.sigma_type, "log_tau": args.sigma_type,
           "motor_bias": args.sigma_tone, "dec_raw": args.sigma_dec_raw, "dec_bias": args.sigma_dec_bias,
           "halt_b": args.sigma_enc, "log_walk_gain": args.sigma_walk, "log_turn_gain": args.sigma_turn,
           "log_turn_gain_r": args.sigma_turn, "log_theta_neuron": args.sigma_neuron,
           **{f"enc_b_{k}": args.sigma_enc for k in range(6)}}
    es = PopulationES(params, {k: sig[k] for k in params}, args.lr, args.seed)
    state = {"gen": 0, "v_max": args.v_start, "yaw_max": args.yaw_start, "attempts": 0, "steps": 0}
    latest = run_dir / "latest.pt"
    if args.resume and latest.exists():
        ck = torch.load(latest, weights_only=False, map_location=device)
        missing, _ = pol.load_state_dict(ck["policy"], strict=False)
        if "log_turn_gain_r" in missing:  # checkpoint de antes do ganho por lado: começa igual ao da esquerda
            with torch.no_grad():
                pol.log_turn_gain_r.copy_(pol.log_turn_gain)
        unknown = set(missing) - {"log_turn_gain_r", "net.log_theta_neuron"}  # o fator por neurônio começa em 0
        if unknown:
            raise RuntimeError(f"parâmetros faltando no checkpoint: {sorted(unknown)}")
        try:
            es.opt.load_state_dict(ck["opt"])
        except ValueError:  # o conjunto de parâmetros variados mudou: o Adam recomeça
            print("estado do otimizador incompatível com os parâmetros atuais: o Adam recomeça")
        for group, name in zip(es.opt.param_groups, es.params):  # a taxa da linha de comando vale na retomada
            group["lr"] = args.lr * es.sigma[name]
        state = ck["state"]
        state.setdefault("yaw_max", 0.0)
        print(f"retomando da geração {state['gen']}")
        # Toda retomada fica registrada com os argumentos (mudanças de receita no meio de um treino gravado).
        with open(run_dir / "resumes.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"gen": state["gen"], "args": vars(args)}) + "\n")
    else:
        if args.init_from:
            print(f"partindo de {args.init_from}: {load_walking(pol, args.init_from, device)} tensores/ganhos copiados")
        obs_rest, _ = env.reset(np.arange(env.n) + 10**8, np.full(env.n, args.v_start))
        with torch.no_grad():
            pol.calibrate_rest(torch.as_tensor(obs_rest[:8], dtype=torch.float32, device=device), 0.65 * args.v_start)
    if not args.probe and not (run_dir / "config.json").exists():
        (run_dir / "config.json").write_text(json.dumps(
            {"args": vars(args), "env": asdict(env_cfg), "controller": asdict(ctrl_cfg), "neurons": pol.n, "es_dim": es.dim,
             "sigma": es.sigma, "versions": library_versions()}, indent=2), encoding="utf-8")
    n = env.n
    print(f"estratégias evolutivas: população {n}, {es.dim:,} parâmetros variados "
          f"({', '.join(f'{k} {v.numel()}' for k, v in params.items())})")

    if args.probe:
        v = np.full(n, args.probe_speed)
        base = run_episodes(env, pol, TEST_ATTEMPT_BASE + np.arange(n), v)
        print(f"média (sem variação), {args.probe_speed:g} cm/s: velocidade {np.mean([e['speed'] for e in base]):.2f} "
              f"(máx {max(e['speed'] for e in base):.2f}), retorno {np.mean([e['return'] for e in base]):.1f}, "
              f"quedas {np.mean([e['fell'] for e in base]):.0%}")
        for scale in map(float, args.probe_scales.split(",")):
            eps = es.sample(10**6 + int(100 * scale), n)
            off = {k: scale * t for k, t in es.offsets(eps, device).items()}
            eps_ = run_episodes(env, pol, TEST_ATTEMPT_BASE + np.arange(n), v, off)
            sp = np.array([e["speed"] for e in eps_])
            print(f"σ×{scale:g}: velocidade média {sp.mean():.2f}, melhores {np.sort(sp)[-5:][::-1].round(2)}, "
                  f"acima de 0,3 cm/s: {(sp > 0.3).mean():.0%}; retorno {np.mean([e['return'] for e in eps_]):.1f} "
                  f"(máx {max(e['return'] for e in eps_):.1f}); quedas {np.mean([e['fell'] for e in eps_]):.0%}", flush=True)
        env.close()
        return

    print(" ger  tentativas  ret_méd  ret_máx  vel_méd  vel_máx queda% v_max sucesso  |grad|  tempo")
    for gen in range(state["gen"], args.generations):
        lr_now = args.lr_warm if gen < args.lr_warm_gens and args.lr_warm > 0 else args.lr
        for group, name in zip(es.opt.param_groups, es.params):
            group["lr"] = lr_now * es.sigma[name]
        t0 = time.perf_counter()
        attempts = np.arange(gen * n, (gen + 1) * n)
        rng = np.random.default_rng(attempt_seed(args.seed, -1 - gen))
        v_max = state["v_max"]
        phase2 = gen >= args.phase2_gen
        p_stand = args.p_stand if phase2 else 0.0
        v_pair = rng.uniform(0.3 * v_max, v_max, n // 2)
        if p_stand > 0:  # pares mandados ficar parados
            v_pair[rng.random(n // 2) < p_stand] = 0.0
        v_cmd = np.concatenate([v_pair, v_pair])  # os dois membros de cada par antitético com o mesmo pedido
        yaw_max = state["yaw_max"] if phase2 else 0.0
        yaw_pair = yaw_schedule(rng, n // 2, env.max_steps, yaw_max, args.yaw_switch / env_cfg.control_dt)
        yaw = np.concatenate([yaw_pair, yaw_pair], axis=1) if yaw_max > 0 else None
        k = gen - args.push_from
        p_push = args.push_max * max(0.0, 1.0 - k / args.push_gens) if args.push_gens > 0 and k >= 0 else 0.0
        push = None
        if p_push > 0:  # empurrão inicial (ajuda do currículo): o par começa já andando na velocidade pedida
            push_pair = np.where(rng.random(n // 2) < p_push, v_pair, 0.0)
            push = np.concatenate([push_pair, push_pair])
        eps = es.sample(gen, n)
        if args.capture:  # a média desta geração; a variação de cada tentativa sai da semente (es.sample)
            (run_dir / "capture").mkdir(exist_ok=True)
            torch.save(pol.state_dict(), run_dir / "capture" / f"policy_it{gen:05d}.pt")
            episodes, cap = run_episodes(env, pol, attempts, v_cmd, es.offsets(eps, device), record=gen, yaw=yaw,
                                         push=push)
            cap.save(run_dir / "capture" / f"it{gen:05d}.npz")
        else:
            episodes = run_episodes(env, pol, attempts, v_cmd, es.offsets(eps, device), yaw=yaw, push=push)
        ret = np.array([e["return"] for e in episodes])
        grad_norm = es.step(eps, ret)
        success = command_success(episodes, env.max_steps)  # da população perturbada (só registro)
        state["gen"] = gen + 1
        state["attempts"] += n
        state["steps"] += int(sum(e["steps"] for e in episodes))
        sp = np.array([e["speed"] for e in episodes])
        metrics = {"gen": gen, "attempts": state["attempts"], "v_max": v_max, "yaw_max": yaw_max,
                   "return_mean": float(ret.mean()),
                   "return_max": float(ret.max()), "speed_mean": float(sp.mean()), "speed_max": float(sp.max()),
                   "fell": float(np.mean([e["fell"] for e in episodes])), "success": success, "grad_norm": grad_norm,
                   "p_push": p_push, "time": time.perf_counter() - t0}
        if args.eval_every and (gen + 1) % args.eval_every == 0:
            speed = min(args.eval_speed, state["v_max"])
            eval_yaw = min(args.eval_yaw, yaw_max)
            level = state["v_max"] + yaw_max  # nível do currículo nesta avaliação
            # Com --p-stand, um quarto dos testes pede para ficar parada (conta no acerto do currículo).
            stand = (np.arange(n) % 4 == 3) if p_stand > 0 else np.zeros(n, bool)
            yaw_tests = np.tile(np.where(stand, 0.0, np.array([0.0, eval_yaw, -eval_yaw])[np.arange(n) % 3]),
                                (env.max_steps, 1))
            ev = run_episodes(env, pol, TEST_ATTEMPT_BASE + np.arange(n), np.where(stand, 0.0, speed),
                              yaw=yaw_tests if eval_yaw > 0 else None)
            stand_ok = float(np.mean([not e["fell"] and abs(e["speed"]) < 0.5 for e, s in zip(ev, stand) if s])) \
                if stand.any() else float("nan")
            ev_move = [e for e, s in zip(ev, stand) if not s]
            # Currículo pela média (a mosca sem perturbação): a velocidade pedida sobe com 80% de acerto.
            eval_success = command_success(ev, env.max_steps, args.yaw_tol if args.yaw_final > 0 and phase2 else None)
            if eval_success >= 0.8:  # primeiro a faixa de velocidade, depois a de giro
                if state["v_max"] < args.v_final and speed >= state["v_max"]:
                    state["v_max"] = min(args.v_final, state["v_max"] + args.v_step)
                elif state["v_max"] >= args.v_final and args.yaw_final > 0 and phase2:
                    state["yaw_max"] = min(args.yaw_final, state["yaw_max"] + args.yaw_step)
            metrics.update(eval_success=eval_success, eval_yaw_cmd=eval_yaw, eval_stand_ok=stand_ok,
                           eval_yaw_error=float(np.mean([e["yaw_error"] for e in ev_move])))
            metrics.update(eval_v_cmd=speed, eval_speed=float(np.mean([e["speed"] for e in ev_move])),
                           eval_glide=float(np.mean([e["glide_frac"] for e in ev_move])),
                           eval_glide6=float(np.mean([e.get("glide6_frac", np.nan) for e in ev_move])),
                           eval_fell=float(np.mean([e["fell"] for e in ev])))
            score = level + eval_success  # o nível do currículo manda; no mesmo nível, o acerto
            best = score > state.get("best_score", -1.0)
            print(f"      avaliação da média a {speed:g} cm/s: vel {metrics['eval_speed']:.2f}, desliza "
                  f"{metrics['eval_glide']:.2f} (com os seis no chão {metrics['eval_glide6']:.2f}), quedas {metrics['eval_fell']:.0%}, acerto {eval_success:.0%}"
                  + (f", curvas ±{eval_yaw:g} rad/s, erro de giro {metrics['eval_yaw_error']:.2f}" if eval_yaw > 0 else "")
                  + (f", fica parada quando pedido em {stand_ok:.0%}" if stand.any() else "")
                  + (" (melhor até agora)" if best else ""),
                  flush=True)
            if best:
                state["best_score"], state["best_gen"] = score, gen + 1
                torch.save({"policy": pol.state_dict(), "state": state, "controller": asdict(ctrl_cfg),
                            "env_cfg": asdict(env_cfg), "args": vars(args)}, run_dir / "best.pt")
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        with open(run_dir / "attempts.jsonl", "a", encoding="utf-8") as f:
            for e in episodes:
                f.write(json.dumps({"gen": gen, **e}) + "\n")
        print(f"{gen:4d} {state['attempts']:11d} {ret.mean():8.1f} {ret.max():8.1f} {sp.mean():8.2f} {sp.max():8.2f} "
              f"{100 * metrics['fell']:6.1f} {v_max:5.1f} {success:7.2f} {grad_norm:7.3f} {metrics['time']:6.1f}", flush=True)
        tmp = latest.with_suffix(".tmp")
        torch.save({"policy": pol.state_dict(), "opt": es.opt.state_dict(), "state": state, "args": vars(args),
                    "controller": asdict(ctrl_cfg), "env_cfg": asdict(env_cfg)}, tmp)
        tmp.replace(latest)
        if (gen + 1) % args.save_every == 0:
            torch.save({"policy": pol.state_dict(), "state": state, "controller": asdict(ctrl_cfg),
                        "env_cfg": asdict(env_cfg), "args": vars(args)}, run_dir / "checkpoints" / f"gen{gen + 1:05d}.pt")
    env.close()


if __name__ == "__main__":
    main()
