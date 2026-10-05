"""M8: análise da marcha num teste fixo: o que cada patim faz e de onde vem o empurrão.

Re-simula um teste fixo (as sementes dos testes do M2, sem variação nem ruído) a partir de um checkpoint do conectoma
(m6_es / connectome_train) ou da MLP (m2_train) e mede, a cada subpasso de física (0,2 ms):
- contato de cada patim com o chão (lâminas) e de outras partes do corpo com o chão;
- ângulo de cada patim em relação ao rumo do corpo (positivo = ponta para fora, nos dois lados);
- velocidade de cada patim ao longo da lâmina e de lado (escorregão);
- força do chão em cada patim, e a parte dela na direção do rumo do corpo (empurrão; negativa = freio).

Saídas em outputs/: <nome>_marcha.json (resumo por patim, de 1 s ao fim) e <nome>_marcha.png (diagrama de contato,
ângulos e empurrão por patim numa janela, parcela do empurrão de cada patim e rastro dos patins visto de cima).

Uso:
    python scripts/m8_gait.py runs/final_s0/checkpoints/gen02000.pt --name final_s0 --speed 3.5
    python scripts/m8_gait.py runs/m7p_mlp_s1/checkpoints/it01000.pt --name mlp_s1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mosca.body.fly import LEGS  # noqa: E402

TEST_ATTEMPT_BASE = 10**9
SHORT = {"T1_left": "frente E", "T1_right": "frente D", "T2_left": "meio E", "T2_right": "meio D",
         "T3_left": "trás E", "T3_right": "trás D"}


class Recorder:
    """Grava, a cada subpasso, contatos, ângulos, velocidades e forças dos patins."""

    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.floor = model.geom("floor").id
        self.thorax = model.body("thorax").id
        self.skate = [model.body(f"skate_{leg}").id for leg in LEGS]
        self.geom_leg = {}
        for k, leg in enumerate(LEGS):
            for tag in ("lf", "lb", "rf", "rb"):
                self.geom_leg[model.geom(f"skate_runner_{tag}_{leg}").id] = k
        self.rows = []
        self.vel = np.zeros(6)
        self.f6 = np.zeros(6)

    def __call__(self, env, sub=None) -> None:
        m, d = self.m, env.datas[0]
        hx = d.xmat[self.thorax].reshape(3, 3)[:, 0]
        heading = np.arctan2(hx[1], hx[0])
        fwd = np.array([np.cos(heading), np.sin(heading), 0.0])
        up = d.xmat[self.thorax].reshape(3, 3)[:, 2]
        row = {"pos": d.xpos[self.thorax].copy(), "heading": heading, "up_z": up[2],
               "contact": np.zeros(6, bool), "fz": np.zeros(6), "f_fwd": np.zeros(6), "f_lat": np.zeros(6),
               "angle": np.zeros(6), "v_along": np.zeros(6), "v_side": np.zeros(6), "skate_pos": np.zeros((6, 3)),
               "other_floor": 0}
        for k, b in enumerate(self.skate):
            ax = d.xmat[b].reshape(3, 3)[:, 0]
            yaw = np.arctan2(ax[1], ax[0])
            rel = (yaw - heading + np.pi) % (2 * np.pi) - np.pi
            row["angle"][k] = rel if LEGS[k].endswith("left") else -rel  # ponta para fora positiva
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, self.vel, 0)
            axis_h = np.array([ax[0], ax[1], 0.0])
            axis_h /= max(np.linalg.norm(axis_h), 1e-12)
            side_h = np.array([-axis_h[1], axis_h[0], 0.0])
            row["v_along"][k] = self.vel[3:] @ axis_h
            row["v_side"][k] = self.vel[3:] @ side_h
            row["skate_pos"][k] = d.xpos[b]
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if self.floor not in (g1, g2):
                continue
            other = g2 if g1 == self.floor else g1
            if other not in self.geom_leg:
                row["other_floor"] += 1
                continue
            k = self.geom_leg[other]
            mujoco.mj_contactForce(m, d, i, self.f6)
            f = c.frame.reshape(3, 3).T @ self.f6[:3]  # força em geom2, no referencial do mundo
            if other == g1:
                f = -f
            row["contact"][k] = True
            row["fz"][k] += f[2]
            row["f_fwd"][k] += f @ fwd
            ax = d.xmat[self.skate[k]].reshape(3, 3)[:, 0]
            side_h = np.array([-ax[1], ax[0], 0.0])
            side_h /= max(np.linalg.norm(side_h), 1e-12)
            row["f_lat"][k] += f @ side_h
        self.rows.append(row)

    def arrays(self) -> dict:
        out = {}
        for key in self.rows[0]:
            out[key] = np.array([r[key] for r in self.rows])
        return out


def run_test(path: str, speed: float, test: int, yaw: float, seconds: float, device) -> tuple[dict, dict, float, str]:
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    if "ac" in ckpt:  # MLP do m2_train
        from m2_eval import load, policy

        env, ac, norm, state = load(path, 1, 1)
        rec = Recorder(env.model)
        env.substep_callback = rec
        obs, _ = env.reset(np.array([TEST_ATTEMPT_BASE + test]), np.array([speed]), yaw_cmd=np.array([yaw]))
        while env.alive.any():
            obs, *_ = env.step(policy(ac, norm, obs))
        kind = "MLP"
    else:
        from m6_eval import load, make_env

        pol, graph, ckpt, train_args = load(path, device)
        env = make_env(ckpt, 1, 1, seconds)
        rec = Recorder(env.model)
        env.substep_callback = rec
        obs, _ = env.reset(np.array([TEST_ATTEMPT_BASE + test]), np.array([speed]), yaw_cmd=np.array([yaw]))
        v = torch.full((1,), float(speed), device=device)
        r = pol.initial_state(1)
        with torch.no_grad():
            while env.alive.any():
                r = pol.step_net(r, pol.currents(torch.as_tensor(obs, dtype=torch.float32, device=device), v))
                obs, *_ = env.step(pol.decode(r).cpu().numpy())
        kind = "conectoma embaralhado" if "shuffled" in str(train_args.get("graph", "")) else "conectoma real"
    stats = env.episode_stats()[0]
    dt = env.model.opt.timestep
    env.close()
    return rec.arrays(), stats, dt, kind


def summarize(a: dict, dt: float, start: float) -> dict:
    k0 = int(start / dt)
    sl = slice(k0, None)
    pos = a["pos"][sl]
    heading = a["heading"][sl]
    fwd = np.stack([np.cos(heading), np.sin(heading)], 1)
    v_body = np.gradient(pos[:, :2], dt, axis=0)
    v_fwd = (v_body * fwd).sum(1)
    contact = a["contact"][sl]
    f_fwd = a["f_fwd"][sl]
    push = np.clip(f_fwd, 0, None).sum(0) * dt
    brake = np.clip(-f_fwd, 0, None).sum(0) * dt
    out = {"vel_media": float(v_fwd.mean()), "giro_medio": float(np.diff(np.unwrap(heading)).sum() / (len(heading) * dt)),
           "inclinacao_media_graus": float(np.degrees(np.arccos(np.clip(a["up_z"][sl], -1, 1))).mean()),
           "outros_contatos_com_chao": float((a["other_floor"][sl] > 0).mean()),
           "empurrao_total": float(push.sum()), "freio_total": float(brake.sum()), "patins": {}}
    for k, leg in enumerate(LEGS):
        c = contact[:, k]
        ang = np.degrees(a["angle"][sl][:, k])
        out["patins"][leg] = {
            "no_chao": float(c.mean()),
            "angulo_medio_graus": float(ang.mean()),
            "angulo_p5_p95_graus": [float(np.percentile(ang, 5)), float(np.percentile(ang, 95))],
            "parcela_do_empurrao": float(push[k] / max(push.sum(), 1e-12)),
            "empurrao": float(push[k]), "freio": float(brake[k]),
            "escorregao_lateral_cm_s": float(np.abs(a["v_side"][sl][c, k]).mean()) if c.any() else float("nan"),
            "rolamento": float((np.abs(a["v_along"][sl][c, k]) / np.maximum(np.abs(v_fwd[c]), 0.3)).mean())
            if c.any() else float("nan"),
        }
    return out


def plot(a: dict, dt: float, summary: dict, title: str, out: Path, window: tuple[float, float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#e8590c", "#f59f00", "#2f9e44", "#20c997", "#1c7ed6", "#9775fa"]
    t = np.arange(len(a["contact"])) * dt
    w = (t >= window[0]) & (t < window[1])
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, width_ratios=[2.2, 1, 1], hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    for k, leg in enumerate(LEGS):
        on = a["contact"][w, k]
        ax1.fill_between(t[w], k - 0.4, k + 0.4, where=on, color=colors[k], step="mid", linewidth=0)
    ax1.set_yticks(range(6), [SHORT[leg] for leg in LEGS])
    ax1.set_title("patim no chão (barra cheia) ou no ar", loc="left", fontsize=11)
    ax1.set_xlim(*window)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    for k, leg in enumerate(LEGS):
        ax2.plot(t[w], np.degrees(a["angle"][w, k]), color=colors[k], lw=1.5, label=SHORT[leg])
    ax2.axhline(0, color="#999", lw=0.8)
    ax2.set_ylabel("graus (+ = ponta para fora)")
    ax2.set_title("ângulo do patim em relação ao rumo do corpo", loc="left", fontsize=11)
    ax2.legend(ncol=6, fontsize=9, loc="upper right")
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    smooth = max(1, round(0.002 / dt))
    kernel = np.ones(smooth) / smooth
    weight = 9.8e-4 * 981  # peso da mosca (dyn), para escala
    for k, leg in enumerate(LEGS):
        ax3.plot(t[w], np.convolve(a["f_fwd"][:, k], kernel, "same")[w] / weight, color=colors[k], lw=1.5)
    ax3.axhline(0, color="#999", lw=0.8)
    ax3.set_ylabel("× peso da mosca")
    ax3.set_xlabel("tempo (s)")
    ax3.set_title("força do chão no patim, na direção do rumo (+ empurra, − freia)", loc="left", fontsize=11)
    ax4 = fig.add_subplot(gs[0, 1:])
    share = [summary["patins"][leg]["parcela_do_empurrao"] * 100 for leg in LEGS]
    ax4.bar([SHORT[leg] for leg in LEGS], share, color=colors)
    ax4.set_ylabel("% do empurrão total")
    ax4.set_title("quem empurra (de 1 s ao fim)", loc="left", fontsize=11)
    ax5 = fig.add_subplot(gs[1:, 1:])
    k0 = int(1.0 / dt)
    for k, leg in enumerate(LEGS):
        p = a["skate_pos"][k0:, k]
        ax5.plot(p[:, 0], p[:, 1], color=colors[k], lw=0.8, label=SHORT[leg])
    p = a["pos"][k0:]
    ax5.plot(p[:, 0], p[:, 1], color="#333", lw=1.5, ls="--", label="tórax")
    ax5.set_aspect("equal")
    ax5.set_xlabel("cm")
    ax5.set_title("rastro dos patins visto de cima (de 1 s ao fim)", loc="left", fontsize=11)
    ax5.legend(fontsize=8, loc="best")
    fig.suptitle(title, fontsize=14, x=0.01, ha="left")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--name", required=True)
    p.add_argument("--speed", type=float, default=3.5)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--test", type=int, default=0)
    p.add_argument("--seconds", type=float, default=0.0)
    p.add_argument("--window", type=float, nargs=2, default=(2.0, 2.4), help="janela dos gráficos de tempo (s)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(4)
    a, stats, dt, kind = run_test(args.checkpoint, args.speed, args.test, args.yaw, args.seconds, torch.device(args.device))
    summary = {"checkpoint": args.checkpoint, "tipo": kind, "teste": args.test, "pedido_cm_s": args.speed,
               "giro_pedido": args.yaw, "segundos": stats["seconds"], "caiu": bool(stats["fell"]),
               "desliza": stats["glide_frac"], **summarize(a, dt, 1.0)}
    Path("outputs").mkdir(exist_ok=True)
    (Path("outputs") / f"{args.name}_marcha.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                                              encoding="utf-8")
    title = (f"{args.name} ({kind}): {summary['vel_media']:.2f} cm/s pedindo {args.speed:g}, "
             f"giro {summary['giro_medio']:+.2f} rad/s, desliza {summary['desliza']:.0%}")
    plot(a, dt, summary, title, Path("outputs") / f"{args.name}_marcha.png", tuple(args.window))
    print(title)
    print(f"{'patim':10s} {'no chão':>8s} {'ângulo':>8s} {'faixa':>14s} {'empurrão':>9s} {'freio/emp':>9s} "
          f"{'escorregão':>10s} {'rolamento':>9s}")
    for leg in LEGS:
        s = summary["patins"][leg]
        print(f"{SHORT[leg]:10s} {s['no_chao']:8.0%} {s['angulo_medio_graus']:+8.1f} "
              f"{s['angulo_p5_p95_graus'][0]:+6.1f}..{s['angulo_p5_p95_graus'][1]:+5.1f} {s['parcela_do_empurrao']:9.0%} "
              f"{s['freio'] / max(s['empurrao'], 1e-12):9.2f} {s['escorregao_lateral_cm_s']:10.3f} {s['rolamento']:9.2f}")
    print(f"inclinação média do corpo {summary['inclinacao_media_graus']:.1f}°, outras partes no chão em "
          f"{summary['outros_contatos_com_chao']:.0%} do tempo")


if __name__ == "__main__":
    main()
