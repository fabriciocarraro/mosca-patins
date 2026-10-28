"""M8: a cena dos controles (M7) para o vídeo: uma linha do tempo por treino (as 3 sementes de cada braço do registro
de 24/10: conectoma real, embaralhado e MLP), com o teste do critério do M2 a cada 50 gerações ou iterações
(runs/NOME/m7_eval.json, do m7_eval.py). Um cursor corre as tentativas de treino; cada avaliação acende cheia (passou)
ou vazada (não passou) quando o cursor chega nela, e a primeira que passa ganha o nº da tentativa ao lado da linha.
Quadro de 1280×720, fundo escuro do vídeo.

Uso:
    python scripts/m8_m7_video.py --bottom 140 --out outputs/v3_controles.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m8_render import BG, H, W, font  # noqa: E402
from mosca.paths import RUNS  # noqa: E402

PER_GEN = 64  # tentativas por geração (estratégias evolutivas) ou iteração (PPO)
EVERY = 50  # o m7_eval.py avalia a cada 50 gerações ou iterações
ARMS = [  # (rótulo, nota, cor, execuções das sementes 0, 1 e 2), na ordem do registro (docs/m7_controles.md)
    ("fiação real", "da mosca", (120, 255, 120), ["final2_s0", "m7f_real_s1", "m7f_real_s2"]),
    ("fiação sorteada", "mesmos neurônios", (255, 110, 150), ["m7f_emb_s0", "m7f_emb_s1", "m7f_emb_s2"]),
    ("rede neural comum", "(MLP)", (80, 200, 255), ["m7f_mlp_s0", "m7f_mlp_s1", "m7f_mlp_s2"]),
]


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--title", default="Quem aprende a patinar primeiro?")
    p.add_argument("--subtitle", default="a mesma tarefa, 3 treinos de cada · a cada 3.200 tentativas, 20 corridas de teste "
                   "sem ajuda, pedindo 3,5 cm/s")
    p.add_argument("--seconds", type=float, default=7.0, help="duração da passagem do cursor")
    p.add_argument("--hold", type=float, default=6.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--gens", type=int, default=1400, help="fim do treino (gerações ou iterações)")
    p.add_argument("--bottom", type=int, default=140, help="faixa livre embaixo (px) para a legenda do corte")
    p.add_argument("--out", default="outputs/v3_controles.mp4")
    args = p.parse_args()

    tests = {}  # execução -> {geração: passou no critério do M2}
    for _, _, _, runs in ARMS:
        for run in runs:
            data = json.loads((RUNS / run / "m7_eval.json").read_text(encoding="utf-8"))
            tests[run] = {int(g): s["m2_ok"] for g, s in data.items() if int(g) % EVERY == 0 and int(g) <= args.gens}

    row_h, row_gap, group_gap, y_top = 28, 6, 30, 160
    x0, x1 = 330, W - 175
    span = args.gens + EVERY / 2  # a última célula é centrada na geração final

    def px(g: float) -> float:
        return x0 + g / span * (x1 - x0)

    def row_y(k: int, s: int) -> int:
        return y_top + k * (3 * row_h + 2 * row_gap + group_gap) + s * (row_h + row_gap)

    y_end = row_y(len(ARMS) - 1, 2) + row_h
    assert y_end + 70 <= H - args.bottom, "o quadro não cabe acima da faixa da legenda"
    f_title, f_count, f_sub, f_arm, f_note, f_small, f_first = (font(34, True), font(30, True), font(18), font(23, True),
                                                                font(16), font(15), font(20, True))
    dim, track, text_dim = (60, 66, 80), (24, 27, 35), (130, 135, 150)
    frames = int(args.seconds * args.fps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", f"{args.fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                           args.out], stdin=subprocess.PIPE)
    for f in range(frames + int(args.hold * args.fps)):
        now = span * min(1.0, f / frames)  # geração em que está o cursor
        img = Image.new("RGB", (W, H), tuple(int(c) for c in BG))
        d = ImageDraw.Draw(img)
        d.text((40, 30), args.title, font=f_title, fill=(235, 235, 245))
        d.text((40, 76), args.subtitle, font=f_sub, fill=(150, 155, 170))
        d.text((W - 40, 34), f"tentativa #{thousands(int(min(now, args.gens) * PER_GEN))}", font=f_count,
               fill=(255, 200, 90), anchor="ra")
        lx = 40  # legenda: cheia = passou, vazada = não passou
        d.rectangle([lx, 112, lx + 18, 128], fill=(200, 205, 215))
        d.text((lx + 26, 120), "passou: 80% das corridas acima de 3 cm/s, deslizando, quase sem quedas", font=f_note,
               fill=(180, 185, 195), anchor="lm")
        lx += 620
        d.rectangle([lx, 112, lx + 18, 128], outline=dim, width=2)
        d.text((lx + 26, 120), "não passou", font=f_note, fill=(180, 185, 195), anchor="lm")
        for k, (label, note, color, runs) in enumerate(ARMS):
            yc = (row_y(k, 0) + row_y(k, 2) + row_h) / 2
            d.text((40, yc - 4), label, font=f_arm, fill=color, anchor="ls")
            d.text((40, yc + 6), note, font=f_note, fill=text_dim, anchor="lt")
            for s, run in enumerate(runs):
                y = row_y(k, s)
                d.text((x0 - 12, y + row_h / 2), f"treino {s + 1}", font=f_small, fill=text_dim, anchor="rm")
                d.rectangle([x0, y + row_h / 2 - 1, x1, y + row_h / 2 + 1], fill=track)
                first = None
                for g, ok in sorted(tests[run].items()):
                    if g > now:
                        break
                    c0, c1 = px(g - EVERY / 2) + 1.5, px(g + EVERY / 2) - 1.5
                    if ok:
                        d.rectangle([c0, y, c1, y + row_h], fill=color)
                        if first is None:
                            first = g
                            d.rectangle([c0 - 2, y - 2, c1 + 2, y + row_h + 2], outline=(245, 245, 250), width=2)
                    else:
                        d.rectangle([c0, y + 1, c1, y + row_h - 1], fill=track, outline=dim, width=1)
                if first is not None:
                    d.text((x1 + 18, y + row_h / 2), f"#{thousands(first * PER_GEN)}", font=f_first, fill=color,
                           anchor="lm")
        for a in range(0, args.gens * PER_GEN + 1, 20000):
            d.text((px(a / PER_GEN), y_end + 12), thousands(a), font=f_small, fill=text_dim, anchor="ma")
        d.text(((x0 + x1) / 2, y_end + 38), "tentativas de treino", font=f_small, fill=text_dim, anchor="ma")
        d.text((x1 + 18, y_top - 22), "passou 1ª vez", font=f_small, fill=text_dim, anchor="lm")
        if f < frames:
            d.line([(px(now), y_top - 10), (px(now), y_end + 6)], fill=(255, 200, 90), width=2)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    firsts = {run: next((g for g, ok in sorted(t.items()) if ok), None) for run, t in tests.items()}
    print("primeira geração no critério do M2: " + ", ".join(f"{r} {g}" for r, g in firsts.items()) + f" -> {args.out}")


if __name__ == "__main__":
    main()
