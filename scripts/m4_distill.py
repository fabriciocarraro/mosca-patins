"""M4: o conectoma aprende a andar imitando a política de caminhada do flybody (DAgger).

A professora (mosca.walking.teacher) segue uma trajetória de referência; o aluno (o
controlador de conectoma, corpo "walk") só recebe a velocidade e o giro pedidos e os
sentidos das próprias patas. A cada iteração:
1. Uma fração β das moscas é conduzida pela professora e o resto pelo aluno (β cai de 1 a
   `--beta-min` em `--beta-iters` iterações). Nas do aluno, a cada passo a ação executada é a
   da professora com probabilidade `--mix` (caindo até `--mix-min`): o aluno vê os próprios
   erros, e as intervenções evitam a queda imediata. Em todas, a professora diz, a cada 2 ms,
   o que faria naquele estado; esse é o alvo do aluno (a ação dela já cortada nos limites).
2. O aluno treina por retropropagação em trechos de `--chunk` passos, sorteados das últimas
   `--buffer-iters` iterações: cada trecho recomeça do estado da rede gravado na coleta, com
   `--burn-in` passos sem gradiente para atualizar o estado, e o erro quadrático normalizado
   (por saída) nos passos restantes. Com `--filtered-loss`, o erro é o das ações depois do
   filtro dos atuadores (τ de 10 ms): a professora comanda em liga-desliga, e 73% da variância
   dos seus comandos é um chacoalhar que os atuadores apagam e que neurônios lentos não seguem.

`--ref-leak-tau` faz a referência da professora acompanhar a mosca (ver WalkingVecEnv): sem
isso, um aluno que fica para trás recebe da professora ordens de "correr para alcançar", que
dependem de um atraso que ele não tem como saber. `--student mlp` troca o conectoma por uma
MLP com as mesmas entradas (controle: separa os problemas da receita dos da fiação).

O desempenho do aluno sozinho oscila entre iterações (anda num teste, fica parado no seguinte)
sem que o erro mostre. Com `--ema`, os testes usam a média móvel exponencial dos parâmetros, e
o melhor teste (fração sem cair × velocidade/pedida, no máximo 1) fica salvo em best.pt.

`--teacher` troca a professora: "flybody" (a política publicada, que só anda reagindo em menos
de 10 ms) ou o checkpoint de uma professora lenta (m4_slow_teacher.py), que anda com a latência
de uma rede de neurônios lentos; o alvo é o comando dela depois do atraso e do filtro.

Critério do M4 (docs/plano.md): ≥90% de caminhadas de 5 s bem-sucedidas a 1–3 cm/s, e o
DNa02 faz virar.

Uso (no Spark):
    python scripts/m4_distill.py --run anda_a --iters 60 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.brain.controller import ConnectomePolicy, ControllerConfig  # noqa: E402
from mosca.brain.graph import Connectome  # noqa: E402
from mosca.capture import library_versions  # noqa: E402
from mosca.env.skate_env import attempt_seed  # noqa: E402
from mosca.paths import MALECNS_DIR, RUNS  # noqa: E402
from mosca.walking.student import MLPStudent, MLPStudentConfig, SlowTeacher  # noqa: E402
from mosca.walking.teacher import FUTURE_STEPS, WalkingTeacher, straight_trajectory  # noqa: E402
from mosca.walking.vec_env import WalkingVecEnv  # noqa: E402

TEST_COMMANDS = [(v, y) for v in (1.0, 2.0, 3.0) for y in (0.0, 1.0, -1.0)]


class PuglieseNetKeys:
    @staticmethod
    def edge_ids_for_rows(pol, graph) -> torch.Tensor:
        """Índices, na lista de todas as ligações (ordem CSR), das ligações que chegam aos motores das
        patas, na mesma ordem em que o modo "só motores" guarda os seus ganhos (ordem COO por linha)."""
        from mosca.body.fly import LEGS as _LEGS
        from mosca.brain.pugliese import signed_matrix
        w = (0.03 * signed_matrix(graph)).tocsr()
        w.sort_indices()
        rows = np.zeros(w.shape[0], dtype=bool)
        rows[np.concatenate([graph.groups[f"motor_{leg}"] for leg in _LEGS])] = True
        edge_rows = np.repeat(np.arange(w.shape[0]), np.diff(w.indptr))
        coo = w.tocoo()
        order_csr = np.flatnonzero(rows[edge_rows])
        order_coo = np.flatnonzero(rows[coo.row])
        # COO de uma CSR ordenada percorre as ligações na mesma ordem da CSR
        assert np.array_equal(coo.row[order_coo], edge_rows[order_csr])
        return torch.as_tensor(order_csr, device=pol.device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-gpu-mem-gb", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--graph", default=str(MALECNS_DIR / "controller_graph_min5.npz"),
                   help="grafo do controlador (o embaralhado, para o controle do M7)")
    p.add_argument("--episode-seconds", type=float, default=2.0)
    p.add_argument("--v-min", type=float, default=0.5)
    p.add_argument("--v-max", type=float, default=3.0)
    p.add_argument("--yaw-max", type=float, default=1.0)
    p.add_argument("--p-stand", type=float, default=0.0,
                   help="fração das tentativas com a mosca mandada ficar parada (v = 0, DNg100 calado): sem elas, "
                        "o aluno pode andar sem depender do DNg100")
    p.add_argument("--beta-iters", type=int, default=15)
    p.add_argument("--beta-min", type=float, default=0.1)
    p.add_argument("--mix", type=float, default=0.0, help="probabilidade inicial de a professora intervir a cada passo")
    p.add_argument("--mix-min", type=float, default=0.0)
    p.add_argument("--mix-iters", type=int, default=30)
    p.add_argument("--init-from", default="", help="checkpoint para partir (parâmetros com o mesmo nome e forma; "
                                                  "ganhos dos motores viram ganhos das mesmas ligações)")
    p.add_argument("--chunk", type=int, default=48)
    p.add_argument("--burn-in", type=int, default=16)
    p.add_argument("--minibatch-chunks", type=int, default=16)
    p.add_argument("--updates", type=int, default=60, help="passos do otimizador por iteração")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dec-lr-mult", type=float, default=10.0,
                   help="multiplicador da taxa do decodificador (os alvos têm médias de várias unidades)")
    p.add_argument("--buffer-iters", type=int, default=4)
    p.add_argument("--motor-synapse-gains", action="store_true",
                   help="degrau 2 da escada: ganho treinável por ligação nas entradas dos motores das patas")
    p.add_argument("--all-synapse-gains", action="store_true",
                   help="degrau 2 completo: ganho treinável por ligação em toda a rede (sinal fixo)")
    p.add_argument("--syn-lr-mult", type=float, default=10.0)
    p.add_argument("--filtered-loss", action="store_true",
                   help="erro das ações depois do filtro dos atuadores (o que chega às juntas)")
    p.add_argument("--ref-leak-tau", type=float, default=0.0,
                   help="constante de tempo (s) com que a referência da professora é puxada para a mosca; 0 = fixa")
    p.add_argument("--ref-heading-tau", type=float, default=None,
                   help="constante de tempo do rumo da referência (padrão: a de --ref-leak-tau)")
    p.add_argument("--reset-turn-gain", action="store_true",
                   help="com --init-from, volta o ganho de giro ao de --turn-gain (no anda_r5L ele caiu de 100 "
                        "para 20 e os DNa01/DNa02, de limiar 90 a 200, pararam de disparar)")
    p.add_argument("--cmd-lr-mult", type=float, default=10.0, help="taxa dos ganhos de comando (0 = fixos)")
    p.add_argument("--student", choices=("conectoma", "mlp"), default="conectoma")
    p.add_argument("--teacher", default="flybody", help='"flybody" ou o checkpoint de uma professora lenta')
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--mlp-layers", type=int, default=2)
    p.add_argument("--student-lag-ms", type=float, default=0.0, help="MLP: atraso das entradas")
    p.add_argument("--student-tau-ms", type=float, default=0.0, help="MLP: filtro na saída (neurônios lentos)")
    p.add_argument("--ema", type=float, default=0.0,
                   help="decaimento por atualização da média móvel dos parâmetros usada nos testes (0 = sem)")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-seconds", type=float, default=5.0)
    p.add_argument("--resume", action="store_true")
    cc = ControllerConfig()
    for name in ("size_ref", "walk_gain", "turn_gain", "enc_std", "prop_offset", "motor_tone", "dec_gain", "tau_scale"):
        p.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(cc, name))
    return p.parse_args()


def to_dev(x, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


@contextmanager
def swapped(module: torch.nn.Module, params: dict[str, torch.Tensor]):
    """Troca temporariamente os parâmetros do módulo pelos de `params` (a média móvel)."""
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


def actuator_filter(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Comandos (passos, lote, saídas) depois do filtro dos atuadores, a partir do primeiro."""
    f = np.empty_like(x)
    f[0] = x[0]
    for t in range(1, len(x)):
        f[t] = f[t - 1] + alpha * (x[t] - f[t - 1])
    return f


class FlybodyTeacher:
    """A política publicada do flybody: alvo = a ação dela cortada nos limites, no formato do aluno."""

    def __init__(self, device):
        self.policy = WalkingTeacher().to(device)

    def reset(self) -> None:
        pass

    def __call__(self, env, obs):
        a = self.policy(env.teacher_obs())
        return env.student_target(a), env.teacher_ctrl(a)


class SlowTeacherAdapter:
    def __init__(self, path, n_envs):
        self.teacher = SlowTeacher(path, n_envs)

    def reset(self) -> None:
        self.teacher.reset()

    def __call__(self, env, obs):
        y = self.teacher(obs, env.v_cmd)
        return y.astype(np.float32), env.ctrl_from_student(y)


def rollout(env, pol, teacher, refs, v_cmd, yaw, driver_teacher, chunk, device, mix=0.0, rng=None):
    """Uma leva de tentativas; devolve observações, alvos, máscara e os estados da rede por trecho."""
    env.reset(refs, v_cmd, yaw)
    teacher.reset()
    T = min(len(r) for r in refs) - FUTURE_STEPS - 1
    n = env.n
    obs_buf = np.zeros((T, n, 186), np.float32)
    tgt_buf = np.zeros((T, n, pol.n_out), np.float32)
    valid = np.zeros((T, n), bool)
    states = torch.zeros(((T + chunk - 1) // chunk, pol.n, n), device=device)
    r = pol.initial_state(n)
    vc = to_dev(v_cmd, device)
    start = env.positions()
    steps = 0
    with torch.no_grad():
        for t in range(T):
            if not env.alive.any():
                break
            if t % chunk == 0:
                states[t // chunk] = r
            obs = env.student_obs()
            r, out = pol(r, to_dev(obs, device), vc)
            out_np = out.cpu().numpy()
            target, teacher_ctrl = teacher(env, obs)
            use_teacher = driver_teacher | ((rng.random(n) < mix) if mix > 0 else False)
            ctrl = np.where(use_teacher[:, None], teacher_ctrl, env.ctrl_from_student(out_np))
            obs_buf[t], tgt_buf[t], valid[t] = obs, target, env.alive
            env.step(ctrl, out_np)
            steps = t + 1
    end = env.positions()
    return {"obs": obs_buf[:steps], "tgt": tgt_buf[:steps], "valid": valid[:steps], "states": states,
            "v_cmd": np.asarray(v_cmd, np.float32), "fell": env.fell.copy(), "t": env.t.copy(),
            "dist": np.linalg.norm(end[:, :2] - start[:, :2], axis=1)}


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

    env = WalkingVecEnv(args.envs, n_threads=args.threads, ref_leak_tau=args.ref_leak_tau,
                        ref_heading_tau=args.ref_heading_tau)
    teacher = FlybodyTeacher(device) if args.teacher == "flybody" else SlowTeacherAdapter(args.teacher, args.envs)
    if args.student == "mlp":
        graph = None
        cfg = MLPStudentConfig(hidden=args.mlp_hidden, layers=args.mlp_layers, lag_steps=round(args.student_lag_ms / 2),
                               tau=args.student_tau_ms / 1000, seed=args.seed)
        pol = MLPStudent(cfg, device=device)
    else:
        graph = Connectome.load(Path(args.graph))
        cfg = ControllerConfig(body="walk", control_dt=0.002, substeps=1, size_ref=args.size_ref, walk_gain=args.walk_gain,
                               turn_gain=args.turn_gain, enc_std=args.enc_std, prop_offset=args.prop_offset,
                               motor_tone=args.motor_tone, dec_gain=args.dec_gain, seed=args.seed, tau_scale=args.tau_scale,
                               motor_synapse_gains=args.motor_synapse_gains, all_synapse_gains=args.all_synapse_gains)
        pol = ConnectomePolicy(graph, cfg, device=device)
    mults = {"rede": 1.0, "codificador": 10.0, "tonus": 3.0, "decodificador": args.dec_lr_mult, "comando": args.cmd_lr_mult,
             "exploracao": 0.0, "sinapses": args.syn_lr_mult, "mlp": 1.0}
    alpha_np = env.act_alpha.astype(np.float32)
    alpha = to_dev(alpha_np, device)
    opt = torch.optim.Adam([{"params": ps, "lr": args.lr * mults[name], "name": name}
                            for name, ps in pol.param_groups().items()])
    state = {"it": 0}
    ema = None
    tgt_std = tgt_std_f = None
    latest = run_dir / "latest.pt"
    if args.init_from and not (args.resume and latest.exists()):
        src = torch.load(args.init_from, weights_only=False, map_location=device)
        own = pol.state_dict()
        fixed = {"net.tau0", "net.a0", "net.theta0", "net.r_max"}  # sorteados a partir da configuração
        copied = {k: v for k, v in src["policy"].items() if k in own and own[k].shape == v.shape and k not in fixed}
        own.update(copied)
        pol.load_state_dict(own)
        tgt_std, tgt_std_f = src.get("tgt_std"), src.get("tgt_std_f")
        if "net.log_gain" in src["policy"] and pol.net.edge_gains:  # ganhos dos motores -> mesmas ligações
            old = PuglieseNetKeys.edge_ids_for_rows(pol, graph)
            with torch.no_grad():
                pol.net.log_edge_gain[old] = src["policy"]["net.log_gain"].to(device)
        if args.reset_turn_gain:
            with torch.no_grad():
                pol.log_turn_gain.fill_(float(np.log(args.turn_gain)))
        print(f"partindo de {args.init_from}: {len(copied)} tensores copiados", flush=True)
    if args.resume and latest.exists():
        ckpt = torch.load(latest, weights_only=False, map_location=device)
        pol.load_state_dict(ckpt["policy"])
        opt.load_state_dict(ckpt["opt"])
        state, tgt_std, tgt_std_f = ckpt["state"], ckpt["tgt_std"], ckpt.get("tgt_std_f")
        if ckpt.get("ema"):
            ema = {k: v.to(device) for k, v in ckpt["ema"].items()}
        print(f"retomando da iteração {state['it']}")
    else:
        (run_dir / "config.json").write_text(json.dumps(
            {"args": vars(args), "controller": asdict(cfg), "neurons": graph.n if graph else 0,
             "trainable": sum(p.numel() for p in pol.parameters()), "versions": library_versions()}, indent=2),
            encoding="utf-8")

    buffer: list[dict] = []
    ref_steps = round(args.episode_seconds / 0.002) + FUTURE_STEPS + 1
    name = f"conectoma (corpo walk): {graph.n:,} neurônios" if graph else f"MLP {cfg.layers}×{cfg.hidden}"
    print(f"{name}, {sum(p.numel() for p in pol.parameters()):,} parâmetros; {device}")
    print("  it   beta  erro_norm (início→fim)  filtrado  quedas(aluno)  vel_aluno/pedida  coleta  treino")
    for it in range(state["it"], args.iters):
        t0 = time.perf_counter()
        rng = np.random.default_rng(attempt_seed(args.seed, it))
        n = env.n
        v_cmd = rng.uniform(args.v_min, args.v_max, n)
        yaw = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(-args.yaw_max, args.yaw_max, n))
        stand = rng.random(n) < args.p_stand
        v_cmd[stand], yaw[stand] = 0.0, 0.0
        refs = [straight_trajectory(ref_steps, v, yaw_speed=y, heading=rng.uniform(-np.pi, np.pi)) for v, y in zip(v_cmd, yaw)]
        beta = max(args.beta_min, 1.0 - it / max(args.beta_iters, 1))
        driver_teacher = np.zeros(n, bool)
        driver_teacher[rng.permutation(n)[: round(beta * n)]] = True
        mix = max(args.mix_min, args.mix * (1.0 - it / max(args.mix_iters, 1))) if args.mix > 0 else 0.0
        ro = rollout(env, pol, teacher, refs, v_cmd, yaw, driver_teacher, args.chunk, device, mix=mix, rng=rng)
        t_collect = time.perf_counter() - t0
        buffer = (buffer + [ro])[-args.buffer_iters :]
        if tgt_std is None:  # escala de cada saída, fixada na primeira coleta (só a professora)
            tgt_std = np.maximum(ro["tgt"][ro["valid"]].std(axis=0), 0.05).astype(np.float32)
            with torch.no_grad():  # o viés do decodificador começa na média dos alvos
                pol.dec_bias.copy_(to_dev(ro["tgt"][ro["valid"]].mean(axis=0), device))
        if tgt_std_f is None:
            filt = actuator_filter(ro["tgt"], alpha_np)
            tgt_std_f = np.maximum(filt[ro["valid"]].std(axis=0), 0.05).astype(np.float32)
        std_t, std_f = to_dev(tgt_std, device), to_dev(tgt_std_f, device)

        # Treino: trechos sorteados do buffer.
        t1 = time.perf_counter()
        losses, losses_f = [], []
        segs = [(b, k, i) for b, r_ in enumerate(buffer) for k in range(r_["states"].shape[0]) for i in range(n)
                if k * args.chunk < len(r_["valid"]) and r_["valid"][k * args.chunk, i]]
        for u in range(args.updates):
            pick = [segs[j] for j in rng.choice(len(segs), size=min(args.minibatch_chunks, len(segs)), replace=False)]
            r = torch.stack([buffer[b]["states"][k][:, i] for b, k, i in pick], dim=1)
            loss_sum, loss_f_sum, count = 0.0, 0.0, 0.0
            out_f = tgt_f = None  # comandos do aluno e da professora depois do filtro dos atuadores
            for t in range(args.chunk):
                obs_t = np.stack([buffer[b]["obs"][min(k * args.chunk + t, len(buffer[b]["obs"]) - 1), i] for b, k, i in pick])
                v_t = to_dev([buffer[b]["v_cmd"][i] for b, k, i in pick], device)
                tgt_t = to_dev(np.stack([buffer[b]["tgt"][min(k * args.chunk + t, len(buffer[b]["tgt"]) - 1), i]
                                         for b, k, i in pick]), device)
                with torch.set_grad_enabled(t >= args.burn_in):
                    r, out = pol(r, to_dev(obs_t, device), v_t)
                    out_f = out if out_f is None else out_f + alpha * (out - out_f)
                tgt_f = tgt_t if tgt_f is None else tgt_f + alpha * (tgt_t - tgt_f)
                if t < args.burn_in:
                    continue
                mask = np.array([k * args.chunk + t < len(buffer[b]["valid"]) and buffer[b]["valid"][k * args.chunk + t, i]
                                 for b, k, i in pick], dtype=np.float32)
                m_t = to_dev(mask, device)
                loss_sum = loss_sum + (((out - tgt_t) / std_t).pow(2).mean(dim=1) * m_t).sum()
                loss_f_sum = loss_f_sum + (((out_f - tgt_f) / std_f).pow(2).mean(dim=1) * m_t).sum()
                count += mask.sum()
            loss, loss_f = loss_sum / max(count, 1.0), loss_f_sum / max(count, 1.0)
            opt.zero_grad()
            (loss_f if args.filtered_loss else loss).backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
            if args.ema > 0:
                with torch.no_grad():
                    if ema is None:
                        ema = {k: p.detach().clone() for k, p in pol.named_parameters()}
                    for k, p in pol.named_parameters():
                        ema[k].lerp_(p.detach(), 1.0 - args.ema)
            losses.append(loss.item())
            losses_f.append(loss_f.item())
        t_train = time.perf_counter() - t1

        student = ~driver_teacher
        speed = ro["dist"] / np.maximum(ro["t"] * 0.002, 1e-6)
        moving = student & (v_cmd > 0)
        ratio = float(np.mean(speed[moving] / v_cmd[moving])) if moving.any() else float("nan")
        falls = float(np.mean(ro["fell"][student])) if student.any() else float("nan")
        main_losses = losses_f if args.filtered_loss else losses
        metrics = {"it": it, "beta": beta, "mix": mix, "loss": float(np.mean(losses)), "loss_filt": float(np.mean(losses_f)),
                   "loss_first": float(np.mean(main_losses[:5])), "loss_last": float(np.mean(main_losses[-5:])),
                   "student_falls": falls,
                   "student_speed_ratio": ratio, "time_collect": t_collect, "time_train": t_train}
        if args.eval_every and (it + 1) % args.eval_every == 0:
            cmds = [TEST_COMMANDS[j % len(TEST_COMMANDS)] for j in range(n)]
            ev_refs = [straight_trajectory(round(args.eval_seconds / 0.002) + FUTURE_STEPS + 1, v, yaw_speed=y) for v, y in cmds]
            with swapped(pol, ema) if ema is not None else nullcontext():
                ev = rollout(env, pol, teacher, ev_refs, np.array([c[0] for c in cmds]), np.array([c[1] for c in cmds]),
                             np.zeros(n, bool), args.chunk, device)
            ok = (~ev["fell"]) & (ev["t"] >= round(args.eval_seconds / 0.002) - 1)
            ev_speed = ev["dist"] / np.maximum(ev["t"] * 0.002, 1e-6)
            metrics.update(eval_success=float(ok.mean()), eval_falls=float(ev["fell"].mean()),
                           eval_speed_ratio=float(np.mean(ev_speed / np.array([c[0] for c in cmds]))))
            score = float(ok.mean()) * min(1.0, metrics["eval_speed_ratio"])
            best = score > state.get("best_score", -1.0)
            print(f"      teste sem professora ({args.eval_seconds:g} s): {ok.mean():.0%} sem cair até o fim, "
                  f"velocidade/pedida {metrics['eval_speed_ratio']:.2f}{' (melhor até aqui)' if best else ''}", flush=True)
            if best:
                state["best_score"], state["best_it"] = score, it + 1
                policy = pol.state_dict()
                if ema is not None:
                    policy.update({k: v.detach().clone() for k, v in ema.items()})
                torch.save({"policy": policy, "state": dict(state), "tgt_std": tgt_std, "tgt_std_f": tgt_std_f,
                            "controller": asdict(cfg), "args": vars(args), "eval": {k: metrics[k] for k in metrics
                                                                                    if k.startswith("eval_")}},
                           run_dir / "best.pt")
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        print(f"{it:4d} {beta:6.2f} {metrics['loss']:10.3f} ({metrics['loss_first']:.2f}→{metrics['loss_last']:.2f}) "
              f"{metrics['loss_filt']:9.3f} {falls:14.2f} {ratio:17.2f} {t_collect:7.1f} {t_train:7.1f}", flush=True)
        state["it"] = it + 1
        ckpt = {"policy": pol.state_dict(), "opt": opt.state_dict(), "state": state, "tgt_std": tgt_std,
                "tgt_std_f": tgt_std_f, "controller": asdict(cfg), "args": vars(args), "ema": ema}
        torch.save(ckpt, latest.with_suffix(".tmp"))
        latest.with_suffix(".tmp").replace(latest)
        if (it + 1) % 10 == 0:
            torch.save(ckpt, run_dir / "checkpoints" / f"it{it + 1:05d}.pt")
    env.close()


if __name__ == "__main__":
    main()
