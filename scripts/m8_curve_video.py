"""M8: curva de aprendizado animada para o vídeo: a velocidade na avaliação sem variação (a mosca "média" da população,
nos testes fixos do treino) se desenha enquanto um contador de tentativas corre, e as trocas de regra aparecem quando
a curva passa por elas. Quadro de 1280×720, fundo escuro do vídeo.

Aceita várias execuções em sequência (ex.: a linhagem evolui_q → evolui_r → evolui_u, que o final2_s0 reproduz).

Uso:
    python scripts/m8_curve_video.py --runs final2_s0 --marks 32000:"cada patim no chão conta" \
        64000:"empurrãozinho no começo" --out outputs/curva.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m8_render import BG, H, W, br, font  # noqa: E402
from mosca.paths import RUNS  # noqa: E402


def load(runs: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = {}
    for run in runs:
        for line in (RUNS / run / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            m = json.loads(line)
            if "eval_speed" in m:
                rows[m.get("gen", m.get("it"))] = m  # retomadas repetem gerações: vale a última
    gens = sorted(rows)
    return (np.array([rows[g]["attempts"] for g in gens], float), np.array([rows[g]["eval_speed"] for g in gens]),
            np.array([rows[g]["eval_v_cmd"] for g in gens]))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=[], help="execuções (curva da velocidade na avaliação)")
    p.add_argument("--series", default="", help="JSON com 'attempts' e uma série (ex.: m8 grounded_series.json)")
    p.add_argument("--key", default="grounded", help="série do JSON (--series)")
    p.add_argument("--percent", action="store_true", help="série em fração: mostra em % (eixo de 0 a 100)")
    p.add_argument("--smooth", type=int, default=1, help="média móvel centrada da série (pontos)")
    p.add_argument("--subtitle", default="avaliação sem variação (a mosca média da população), nos testes do treino")
    p.add_argument("--value", default="{v} cm/s", help="rótulo do ponto atual ({v} = valor)")
    p.add_argument("--marks", nargs="*", default=[], help="tentativa:rótulo das trocas de regra")
    p.add_argument("--title", default="Velocidade, tentativa após tentativa")
    p.add_argument("--seconds", type=float, default=7.0, help="duração da animação")
    p.add_argument("--hold", type=float, default=2.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--ymax", type=float, default=4.5)
    p.add_argument("--scale", choices=("linear", "log", "sqrt"), default="sqrt",
                   help="eixo das tentativas (sqrt: o começo, onde ela mais aprende, fica visível sem espremer o fim)")
    p.add_argument("--first", default="", help="ponto à parte, rotulado: tentativa:velocidade:rótulo (ex.: a #1)")
    p.add_argument("--bottom", type=int, default=90, help="margem de baixo (px), para caber a legenda do corte")
    p.add_argument("--out", default="outputs/curva.mp4")
    args = p.parse_args()

    if args.series:
        data = json.loads(Path(args.series).read_text(encoding="utf-8"))
        att = np.array(data["attempts"], float)
        speed = np.array(data[args.key], float) * (100.0 if args.percent else 1.0)
        if args.smooth > 1:  # média móvel centrada (as médias por geração saem de ~40 tentativas)
            n = args.smooth // 2
            pad = np.concatenate([np.full(n, speed[0]), speed, np.full(n, speed[-1])])
            speed = np.convolve(pad, np.ones(2 * n + 1) / (2 * n + 1), "valid")
        asked = None
        if args.percent:
            args.ymax = 100.0
    else:
        att, speed, asked = load(args.runs)
    marks = [(float(m.split(":", 1)[0]), m.split(":", 1)[1]) for m in args.marks]
    x0, x1, y0, y1 = 130, W - 60, 150, H - args.bottom  # área do gráfico
    total = att[-1]

    def px(a, v):
        u = (np.log10(max(a, 1.0)) / np.log10(total) if args.scale == "log" else
             np.sqrt(max(a, 0.0) / total) if args.scale == "sqrt" else a / total)
        return x0 + u * (x1 - x0), y1 - min(v, args.ymax) / args.ymax * (y1 - y0)

    f_title, f_count, f_axis, f_lab = font(34, True), font(30, True), font(18), font(20, True)
    frames = int(args.seconds * args.fps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", f"{args.fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                           args.out], stdin=subprocess.PIPE)
    for f in range(frames + int(args.hold * args.fps)):
        frac = min(1.0, f / frames)
        # Começa devagar: as primeiras tentativas são as que mais mudam (no eixo em potências de 10, o contador acelera
        # junto com o eixo).
        now = (total ** frac if args.scale == "log" else 1 + (total - 1) * frac ** 2 if args.scale == "sqrt" else
               1 + (total - 1) * frac ** 1.6)
        k = int(np.searchsorted(att, now, side="right"))
        img = Image.new("RGB", (W, H), tuple(int(c) for c in BG))
        d = ImageDraw.Draw(img)
        d.text((40, 30), args.title, font=f_title, fill=(235, 235, 245))
        d.text((40, 76), args.subtitle, font=f_axis, fill=(150, 155, 170))
        d.text((W - 40, 34), f"tentativa #{int(now):,}".replace(",", "."), font=f_count, fill=(255, 200, 90), anchor="ra")
        for v in (range(0, 101, 20) if args.percent else range(0, int(args.ymax) + 1)):
            _, y = px(0, v)
            d.line([(x0, y), (x1, y)], fill=(34, 38, 48), width=1)
            d.text((x0 - 14, y), f"{v}%" if args.percent else f"{v} cm/s", font=f_axis, fill=(130, 135, 150), anchor="rm")
        ticks = ([1, 10, 100, 1000, 10000, int(total)] if args.scale == "log" else
                 [1, 1000, 5000, 20000, 40000, 60000, int(total)] if args.scale == "sqrt" else
                 range(0, int(total) + 1, 20000))
        for a in ticks:
            x, _ = px(a, 0)
            d.text((x, y1 + 14), f"#{a:,}".replace(",", "."), font=f_axis, fill=(130, 135, 150), anchor="ma")
        d.text(((x0 + x1) / 2, y1 + 44), "tentativas", font=f_axis, fill=(130, 135, 150), anchor="ma")
        for j, (a, label) in enumerate(marks):
            if now >= a:
                x, _ = px(a, 0)
                d.line([(x, y0 - 10 + 30 * j), (x, y1)], fill=(255, 90, 140), width=2)
                d.text((x - 8, y0 - 8 + 30 * j), label, font=f_lab, fill=(255, 140, 176), anchor="ra")
        if args.first:
            a1, v1, lab1 = args.first.split(":", 2)
            fx, fy = px(float(a1), float(v1))
            d.ellipse([fx - 6, fy - 6, fx + 6, fy + 6], fill=(160, 165, 180))
            d.text((fx + 10, fy - 30), lab1, font=f_axis, fill=(170, 175, 190))
        if k > 1:
            if asked is not None:
                d.line([px(a, v) for a, v in zip(att[:k], asked[:k])], fill=(110, 110, 120), width=2)
            d.line([px(a, v) for a, v in zip(att[:k], speed[:k])], fill=(255, 170, 60), width=4)
            hx, hy = px(att[k - 1], speed[k - 1])
            d.ellipse([hx - 7, hy - 7, hx + 7, hy + 7], fill=(255, 170, 60))
            value = args.value.format(v=f"{speed[k - 1]:.0f}" if args.percent else br(speed[k - 1]))
            d.text((min(hx + 12, x1 - 150), hy - 34), value, font=f_lab, fill=(255, 210, 150))
            if asked is not None:
                lx, ly = px(att[min(k, len(att)) - 1], asked[min(k, len(asked)) - 1])
                d.text((x1, ly - 26), "pedido", font=f_axis, fill=(130, 130, 140), anchor="ra")
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{len(att)} pontos até {int(total):,} tentativas -> {args.out}".replace(",", "."))


if __name__ == "__main__":
    main()
