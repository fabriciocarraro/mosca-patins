"""M8: figura da marcha para o vídeo, uma linha por geração: quais patins tocam o chão numa janela de tempo (barras) e
quanto cada patim empurra (saídas do m8_gait.py, outputs/<nome>_marcha.npz). Quadro de 1280×720, fundo escuro como o
do vídeo.

Uso:
    python scripts/m8_gait_video.py outputs/final2_g100_marcha.npz outputs/final2_g500_marcha.npz \
        --labels "geração 100" "geração 500" --out outputs/marcha_fases.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.body.fly import LEGS  # noqa: E402

COLORS = {"T1_left": "#e8590c", "T1_right": "#f59f00", "T2_left": "#2f9e44", "T2_right": "#20c997",
          "T3_left": "#1c7ed6", "T3_right": "#9775fa"}  # as mesmas do m8_gait.py
ORDER = ["T1_left", "T2_left", "T3_left", "T1_right", "T2_right", "T3_right"]  # esquerdas em cima, direitas embaixo
SHORT = {"T1_left": "frente E", "T1_right": "frente D", "T2_left": "meio E", "T2_right": "meio D",
         "T3_left": "trás E", "T3_right": "trás D"}
BG, FG, DIM = "#0e1016", "#e6e6ef", "#9aa0b4"


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gaits", nargs="+", help="outputs/<nome>_marcha.npz, uma por linha")
    p.add_argument("--labels", nargs="+", required=True, help="rótulo de cada linha (ex.: 'geração 100')")
    p.add_argument("--window", type=float, nargs=2, default=(2.0, 2.5), help="janela do diagrama de contato (s)")
    p.add_argument("--title", default="Quais patins tocam o chão, e quem empurra")
    p.add_argument("--out", default="outputs/marcha_fases.png")
    args = p.parse_args()
    if len(args.labels) != len(args.gaits):
        p.error("um rótulo por arquivo")

    rows = len(args.gaits)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(rows, 2, width_ratios=[3.2, 1], left=0.09, right=0.97, top=0.86, bottom=0.07,
                          hspace=0.55, wspace=0.08)
    sub = None
    for i, (path, label) in enumerate(zip(args.gaits, args.labels)):
        z = np.load(path)
        contact, dt = z["contact"], float(z["dt"])
        s = json.loads(str(z["summary"]))
        sub = sub or (f"teste fixo {s['teste'] + 1}, pedindo {s['pedido_cm_s']:g} cm/s em linha reta, sem variação · "
                      f"barras: patim no chão numa janela de {args.window[1] - args.window[0]:g} s".replace(".", ","))
        t = np.arange(len(contact)) * dt
        w = (t >= args.window[0]) & (t < args.window[1])
        ax = fig.add_subplot(gs[i, 0], facecolor=BG)
        for lane, leg in enumerate(ORDER):
            k = LEGS.index(leg)
            y = len(ORDER) - 1 - lane
            ax.fill_between(t[w], y - 0.38, y + 0.38, where=contact[w, k], color=COLORS[leg], step="mid", linewidth=0)
        ax.axhline(2.5, color="#2a2e3a", lw=1)
        ax.set_xlim(*args.window)
        ax.set_ylim(-0.6, len(ORDER) - 0.4)
        ax.set_yticks(range(len(ORDER)), [SHORT[leg] for leg in reversed(ORDER)], fontsize=8, color=DIM)
        ax.set_xticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        grounded = np.mean([s["patins"][leg]["no_chao"] for leg in LEGS])
        lifted = [SHORT[leg] for leg in LEGS if s["patins"][leg]["no_chao"] < 0.5]
        note = f"patins no chão: {pct(grounded)}" + (f" · no ar quase sempre: {', '.join(lifted)}" if lifted else "")
        ax.set_title(f"{label}   ", loc="left", fontsize=13, color=FG, fontweight="bold", pad=4)
        ax.annotate(note, xy=(1.0, 1.02), xycoords="axes fraction", fontsize=10.5, color=DIM, va="bottom", ha="right")
        bx = fig.add_subplot(gs[i, 1], facecolor=BG)
        share = [s["patins"][leg]["parcela_do_empurrao"] for leg in ORDER]
        ys = np.arange(len(ORDER))[::-1]
        bx.barh(ys, share, color=[COLORS[leg] for leg in ORDER], height=0.76)
        for y, v in zip(ys, share):
            bx.text(v + 0.02, y, pct(v), va="center", fontsize=8, color=FG)
        bx.set_xlim(0, 1.0)
        bx.set_ylim(-0.6, len(ORDER) - 0.4)
        bx.axis("off")
        if i == 0:
            bx.set_title("parcela do empurrão", fontsize=10.5, color=DIM, pad=4)
    fig.text(0.09, 0.95, args.title, fontsize=20, color=FG, fontweight="bold", va="center")
    fig.text(0.09, 0.905, sub, fontsize=10.5, color=DIM, va="center")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=BG)
    print(f"{rows} linhas -> {args.out}")


if __name__ == "__main__":
    main()
