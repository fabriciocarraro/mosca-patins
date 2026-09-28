"""M4: confere o critério de pronto do conectoma que anda (docs/plano.md).

Critério: ≥90% de caminhadas de 5 s bem-sucedidas a 1–3 cm/s, e o DNa02 faz virar. Cada
tentativa parte da mosca parada, sem professora e sem ruído. Bem-sucedida = não cai e a
velocidade média para a frente (velocímetro do tórax, de 0,5 s a 5 s) fica a menos de
max(0,5 cm/s, 25%) da pedida (a mesma tolerância do M2) e, sem giro pedido, o giro médio fica
abaixo de 0,5 rad/s (andar em círculos não conta como reta).

Testes:
- reta: 1, 2 e 3 cm/s, rumo inicial sorteado;
- curva: 2 cm/s com giro pedido de ±1 rad/s (entra nos DNa01/DNa02 do lado da curva);
- DNa02 sozinho: sem giro pedido, corrente só nos DNa02 de um lado (a do giro de 1 rad/s). O efeito é a
  diferença de giro contra a mesma caminhada sem estímulo (mesmos rumos, pareada): um desvio próprio
  da marcha não conta a favor nem contra;
- DNg100 calado: 2 cm/s pedidos, mas a taxa dos dois DNg100 é zerada a cada passo.

Uso:
    python scripts/m4_eval.py --ckpt runs/anda_r5L/latest.pt --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.brain.controller import ConnectomePolicy, ControllerConfig  # noqa: E402
from mosca.brain.graph import Connectome  # noqa: E402
from mosca.paths import MALECNS_DIR  # noqa: E402
from mosca.walking.evaluate import run_student as run  # noqa: E402
from mosca.walking.vec_env import WalkingVecEnv  # noqa: E402

DT = 0.002


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--graph", default="", help="padrão: o do treino (args do checkpoint)")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", default="", help="JSON com os resultados")
    p.add_argument("--raw", action="store_true", help="usa os parâmetros crus mesmo se o checkpoint tiver a média móvel")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ck = torch.load(args.ckpt, weights_only=False, map_location=device)
    ctrl = dict(ck["controller"])
    if "turn_cells" in ctrl:
        ctrl["turn_cells"] = tuple(ctrl["turn_cells"])
    cfg = ControllerConfig(**ctrl)
    graph_path = Path(args.graph or ck["args"].get("graph") or MALECNS_DIR / "controller_graph_min5.npz")
    if not graph_path.exists():  # caminho gravado em outra máquina: o mesmo arquivo na pasta local
        graph_path = MALECNS_DIR / graph_path.name
    graph = Connectome.load(graph_path)
    pol = ConnectomePolicy(graph, cfg, device=device)
    policy = dict(ck["policy"])
    if ck.get("ema") and not args.raw:  # checkpoint de iteração com a média móvel dos parâmetros
        policy.update({k: v.to(device) for k, v in ck["ema"].items()})
    pol.load_state_dict(policy)
    env = WalkingVecEnv(args.envs, n_threads=args.threads)
    rng = np.random.default_rng(args.seed)
    n = args.envs
    heads = rng.uniform(-np.pi, np.pi, n)
    results = {"ckpt": args.ckpt, "graph": graph_path.name}

    v = np.array([(1.0, 2.0, 3.0)[k % 3] for k in range(n)])
    res = run(env, pol, v, np.zeros(n), heads, args.seconds, device)
    ok = res["completed"] & (np.abs(res["speed"] - v) < np.maximum(0.5, 0.25 * v)) & (np.abs(res["yaw_rate"]) < 0.5)
    straight = {}
    for speed in (1.0, 2.0, 3.0):
        sel = v == speed
        straight[str(speed)] = {"success": float(ok[sel].mean()), "falls": float(res["fell"][sel].mean()),
                                "speed_mean": float(res["speed"][sel].mean()),
                                "yaw_mean": float(res["yaw_rate"][sel].mean()), "yaw_sd": float(res["yaw_rate"][sel].std()),
                                "straight_frac": float((np.abs(res["yaw_rate"][sel]) < 0.5).mean())}
    results["straight"] = straight
    results["straight_success"] = float(ok.mean())
    results["straight_yaw_rate"] = float(res["yaw_rate"][res["completed"]].mean()) if res["completed"].any() else 0.0
    print(f"reta: {ok.mean():.0%} bem-sucedidas (critério: 90%); giro médio sem giro pedido "
          f"{results['straight_yaw_rate']:+.2f} rad/s")
    for speed, s in straight.items():
        print(f"   {speed} cm/s: {s['success']:.0%} bem-sucedidas, quedas {s['falls']:.0%}, velocidade média {s['speed_mean']:.2f} cm/s, "
              f"giro {s['yaw_mean']:+.2f} ± {s['yaw_sd']:.2f} rad/s ({s['straight_frac']:.0%} abaixo de 0,5)")

    yaw = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    res = run(env, pol, np.full(n, 2.0), yaw, heads, args.seconds, device)
    right_way = np.sign(res["yaw_rate"]) == np.sign(yaw)
    results["turn"] = {"right_direction": float(right_way[res["completed"]].mean()) if res["completed"].any() else 0.0,
                       "falls": float(res["fell"].mean()),
                       "yaw_rate_left": float(res["yaw_rate"][yaw > 0].mean()),
                       "yaw_rate_right": float(res["yaw_rate"][yaw < 0].mean())}
    print(f"curva pedida (±1 rad/s a 2 cm/s): giro médio {results['turn']['yaw_rate_left']:+.2f} / "
          f"{results['turn']['yaw_rate_right']:+.2f} rad/s; lado certo em {results['turn']['right_direction']:.0%}; "
          f"quedas {results['turn']['falls']:.0%}")

    side = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    extra = torch.zeros(pol.n, n, device=device)
    dna02_l = torch.as_tensor(graph.groups["DNa02_L"], device=device)
    dna02_r = torch.as_tensor(graph.groups["DNa02_R"], device=device)
    turn = pol.log_turn_gain.exp().item()
    extra[dna02_l[:, None], torch.as_tensor(np.flatnonzero(side > 0), device=device)[None, :]] = turn
    extra[dna02_r[:, None], torch.as_tensor(np.flatnonzero(side < 0), device=device)[None, :]] = turn
    base = run(env, pol, np.full(n, 2.0), np.zeros(n), heads, args.seconds, device)
    res = run(env, pol, np.full(n, 2.0), np.zeros(n), heads, args.seconds, device, extra_current=extra)
    delta = res["yaw_rate"] - base["yaw_rate"]
    ok_pair = res["completed"] & base["completed"]
    right_way = np.sign(delta) == np.sign(side)
    results["dna02"] = {"right_direction": float(right_way[ok_pair].mean()) if ok_pair.any() else 0.0,
                        "falls": float(res["fell"].mean()), "baseline_yaw": float(base["yaw_rate"].mean()),
                        "yaw_rate_left": float(res["yaw_rate"][side > 0].mean()),
                        "yaw_rate_right": float(res["yaw_rate"][side < 0].mean()),
                        "delta_left": float(delta[side > 0].mean()), "delta_right": float(delta[side < 0].mean())}
    print(f"DNa02 de um lado (sem giro pedido): giro {results['dna02']['yaw_rate_left']:+.2f} (esquerdo) / "
          f"{results['dna02']['yaw_rate_right']:+.2f} (direito) rad/s contra {results['dna02']['baseline_yaw']:+.2f} sem estímulo; "
          f"efeito {results['dna02']['delta_left']:+.2f} / {results['dna02']['delta_right']:+.2f} rad/s; "
          f"lado certo em {results['dna02']['right_direction']:.0%}")

    silence = torch.as_tensor(np.concatenate([graph.groups["DNg100_L"], graph.groups["DNg100_R"]]), device=device)
    res = run(env, pol, np.full(n, 2.0), np.zeros(n), heads, args.seconds, device, silence=silence)
    results["dng100_silenced"] = {"speed_mean": float(res["speed"].mean()), "falls": float(res["fell"].mean())}
    print(f"DNg100 calado (2 cm/s pedidos): velocidade média {res['speed'].mean():.2f} cm/s, quedas {res['fell'].mean():.0%}")

    ok_m4 = results["straight_success"] >= 0.9 and results["dna02"]["right_direction"] >= 0.9
    results["m4_criterion"] = bool(ok_m4)
    print(f"critério do M4: {'cumprido' if ok_m4 else 'ainda não'}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
