"""Testes do conectoma que anda sozinho (critério do M4), usados pelo m4_eval.py e pela destilação.

Cada tentativa parte da mosca parada, sem professora e sem ruído. A velocidade é a média do
velocímetro do tórax (para a frente) e o giro, a do giroscópio (guinada), de 0,5 s até o fim.
Uma reta é bem-sucedida se a mosca não cai, a velocidade fica a menos de max(0,5 cm/s, 25%) da
pedida (a tolerância do M2) e o giro médio fica abaixo de 0,5 rad/s.
"""

from __future__ import annotations

import numpy as np
import torch

from mosca.walking.teacher import FUTURE_STEPS, straight_trajectory

DT = 0.002
WARM = 0.5  # s descartados no começo (a mosca sai do repouso)


def run_student(env, pol, v_cmd, yaw_cmd, headings, seconds, device, silence=None, extra_current=None) -> dict:
    """Uma leva de tentativas do conectoma sozinho; devolve quedas, conclusão e médias de velocidade e giro.
    `silence`: neurônios cuja taxa é zerada a cada passo; `extra_current` (N, lote): corrente somada."""
    n = env.n
    steps = round(seconds / DT)
    refs = [straight_trajectory(steps + FUTURE_STEPS + 1, v, yaw_speed=y, heading=h) for v, y, h in zip(v_cmd, yaw_cmd, headings)]
    env.reset(refs, np.asarray(v_cmd, float), np.asarray(yaw_cmd, float))
    r = pol.initial_state(n)
    vc = torch.as_tensor(np.asarray(v_cmd), dtype=torch.float32, device=device)
    warm = round(WARM / DT)
    v_sum, w_sum, count = np.zeros(n), np.zeros(n), np.zeros(n)
    with torch.no_grad():
        for t in range(steps):
            if not env.alive.any():
                break
            obs = torch.as_tensor(env.student_obs(), dtype=torch.float32, device=device)
            current = pol.currents(obs, vc)
            if extra_current is not None:
                current = current + extra_current
            r = pol.step_net(r, current)
            if silence is not None:
                r[silence] = 0.0
            out = pol.decode(r).cpu().numpy()
            alive = env.alive.copy()
            env.step(env.ctrl_from_student(out), out)
            if t >= warm:
                m = alive & env.alive
                v_sum += np.where(m, env.sensor_mean[:, env.velocimeter[0]], 0.0)
                w_sum += np.where(m, env.sensor_mean[:, env.gyro[2]], 0.0)
                count += m
    completed = ~env.fell & (env.t >= steps - 1)
    return {"fell": env.fell.copy(), "completed": completed, "speed": v_sum / np.maximum(count, 1),
            "yaw_rate": w_sum / np.maximum(count, 1)}


def straight_ok(res: dict, v: np.ndarray) -> np.ndarray:
    return res["completed"] & (np.abs(res["speed"] - v) < np.maximum(0.5, 0.25 * v)) & (np.abs(res["yaw_rate"]) < 0.5)


def side_current(pol, groups: dict, cell: str, sides: np.ndarray, device) -> torch.Tensor:
    """Corrente (N, lote) nos neurônios `cell` do lado de cada tentativa (+1 esquerdo, −1 direito, 0 nenhum),
    igual à que o comando de giro de 1 rad/s injeta."""
    extra = torch.zeros(pol.n, len(sides), device=device)
    turn = pol.log_turn_gain.exp().item()
    for side, code in ((1, "L"), (-1, "R")):
        cols = np.flatnonzero(sides == side)
        if len(cols):
            rows = torch.as_tensor(groups[f"{cell}_{code}"], device=device)
            extra[rows[:, None], torch.as_tensor(cols, device=device)[None, :]] = turn
    return extra


def quick_m4(env, pol, groups: dict, device, seconds: float = 5.0, seed: int = 0) -> dict:
    """Critério do M4 numa leva só: um quarto das moscas em reta a 1, 2 e 3 cm/s cada, e o resto a 2 cm/s
    com o DNa02 de um lado estimulado, com os mesmos rumos de moscas da reta a 2 cm/s (pareadas)."""
    n = env.n
    k = n // 4
    s = n - 3 * k
    heads = np.random.default_rng(seed).uniform(-np.pi, np.pi, n)
    v = np.concatenate([np.full(k, 1.0), np.full(k, 2.0), np.full(k, 3.0), np.full(s, 2.0)])
    pair = k + np.arange(s) % k  # moscas da reta a 2 cm/s com o mesmo rumo
    heads[3 * k :] = heads[pair]
    sides = np.zeros(n)
    sides[3 * k :] = np.where(np.arange(s) % 2 == 0, 1.0, -1.0)
    res = run_student(env, pol, v, np.zeros(n), heads, seconds, device,
                      extra_current=side_current(pol, groups, "DNa02", sides, device))
    ok = straight_ok(res, v)[: 3 * k]
    delta = res["yaw_rate"][3 * k :] - res["yaw_rate"][pair]
    both = res["completed"][3 * k :] & res["completed"][pair]
    right = (np.sign(delta) == sides[3 * k :]) & both
    out = {"straight_success": float(ok.mean()), "dna02_right": float(right.mean()),
           "falls": float(res["fell"].mean()),
           "dna02_left_effect": float(delta[sides[3 * k :] > 0].mean()),
           "dna02_right_effect": float(delta[sides[3 * k :] < 0].mean())}
    for j, speed in enumerate((1.0, 2.0, 3.0)):
        sl = slice(j * k, (j + 1) * k)
        out[f"straight_{speed:g}"] = float(ok[sl].mean())
        out[f"speed_{speed:g}"] = float(res["speed"][sl].mean())
        out[f"yaw_{speed:g}"] = float(res["yaw_rate"][sl].mean())
    out["score"] = out["straight_success"] * out["dna02_right"]
    return out
