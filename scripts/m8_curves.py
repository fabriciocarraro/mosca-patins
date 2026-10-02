"""M8: curvas de aprendizado de um treino (avaliação da média, sem perturbação) para o vídeo.

Painéis, por nº de tentativas de treino: velocidade na avaliação (com a velocidade pedida pelo currículo) e fração do
tempo deslizando. As retomadas que mudaram a receita (resumes.jsonl) aparecem como linhas verticais rotuladas.

Uso:
    python scripts/m8_curves.py --run final_s0 --out outputs/final_s0_curvas.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.paths import RUNS  # noqa: E402

RECIPE_KEYS = ("w_roll", "w_glide", "w_yaw", "yaw_gated", "yaw_filtered", "p_stand", "lr")
LABELS = {"w_roll": "rolamento", "w_glide": "deslize", "w_yaw": "giro", "yaw_gated": "giro só andando",
          "yaw_filtered": "giro médio de 200 ms", "p_stand": "pedidos de parar", "lr": "taxa"}


def describe(key: str, old, new) -> str:
    if isinstance(new, bool):
        return f"{LABELS[key]}: {'sim' if new else 'não'}"
    return f"{LABELS[key]} {old:g}→{new:g}".replace(".", ",")


def recipe_changes(run_dir: Path) -> list[tuple[int, str]]:
    """Gerações em que uma retomada mudou a receita, com o que mudou."""
    path = run_dir / "resumes.jsonl"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["args"]
    out, prev = [], {k: config.get(k) for k in RECIPE_KEYS}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        now = {k: r["args"].get(k) for k in RECIPE_KEYS}
        diff = [describe(k, prev[k], now[k]) for k in RECIPE_KEYS if now[k] != prev[k]]
        if diff:
            out.append((r["gen"], "\n ".join(diff)))
        prev = now
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--out", default="")
    args = p.parse_args()
    run_dir = RUNS / args.run
    rows = {}
    for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        if "eval_speed" in m:
            rows[m.get("gen", m.get("it"))] = m  # retomadas repetem gerações: vale a última
    gens = sorted(rows)
    att = np.array([rows[g]["attempts"] for g in gens])
    speed = np.array([rows[g]["eval_speed"] for g in gens])
    asked = np.array([rows[g]["eval_v_cmd"] for g in gens])
    glide = np.array([rows[g].get("eval_glide", np.nan) for g in gens])
    per_gen = att[-1] / (gens[-1] + 1)

    plt.rcParams.update({"font.size": 15, "axes.edgecolor": "#888", "axes.labelcolor": "#ddd", "xtick.color": "#bbb",
                         "ytick.color": "#bbb", "text.color": "#eee"})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, facecolor="#0e1016")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0e1016")
        ax.grid(color="#2a2e38")
    ax1.plot(att, asked, "--", color="#888", lw=2, label="velocidade pedida")
    ax1.plot(att, speed, color="#ffaa3c", lw=2.5, label="velocidade (avaliação sem variação)")
    ax1.set_ylabel("cm/s")
    ax1.legend(loc="lower right", facecolor="#0e1016", edgecolor="#444")
    ax2.plot(att, 100 * glide, color="#50c8ff", lw=2.5)
    ax2.set_ylabel("% do tempo deslizando")
    ax2.set_xlabel("tentativas de treino")
    ax2.set_ylim(0, 100)
    for gen, what in recipe_changes(run_dir):
        x = gen * per_gen
        for ax in (ax1, ax2):
            ax.axvline(x, color="#ff5a8c", lw=1.5, ls=":")
        ax2.text(x, 97, f" recompensa mudou:\n {what}", va="top", fontsize=12, color="#ff8cb0")
    ax1.set_title(f"{args.run}: aprendendo a patinar", loc="left")
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
    fig.tight_layout()
    out = args.out or f"outputs/{args.run}_curvas.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, facecolor=fig.get_facecolor())
    print(f"{len(gens)} avaliações até a geração {gens[-1]} ({att[-1]:,} tentativas) -> {out}")


if __name__ == "__main__":
    main()
