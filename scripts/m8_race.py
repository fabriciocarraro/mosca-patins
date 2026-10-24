"""M8: "corrida" de tentativas gravadas, vista de cima: o rastro de cada tentativa cresce no tempo real da simulação,
uma raia por tentativa, com a distância percorrida. Mostra o aprendizado de um jeito que a câmera que acompanha a
mosca não mostra (nela, a velocidade quase não aparece). Entradas: traços do m8_trace.py (modo captura ou teste).

Cada rastro é girado para que o deslocamento total aponte para a direita; o desvio de lado aparece como desvio na
raia, na mesma escala.

Uso:
    python scripts/m8_race.py runs/final2_s0/video/tentativa_{1,47,318,4658}.npz --out outputs/corrida.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m8_render import BG, H, W, br, font  # noqa: E402

COLORS = [(140, 145, 160), (80, 200, 255), (120, 255, 120), (255, 200, 90), (255, 110, 150), (190, 140, 255)]


def path_of(z) -> tuple[np.ndarray, float, dict]:
    meta = json.loads(str(z["meta"]))
    root = meta.get("root_qadr", 0)
    p = z["qpos"][:, root:root + 2].astype(np.float64)
    p -= p[0]
    d = p[-1]
    a = np.arctan2(d[1], d[0]) if np.linalg.norm(d) > 0.05 else 0.0
    c, s = np.cos(a), np.sin(a)  # gira por −a: o deslocamento total fica apontando para +x
    dt = meta["control_dt"] / meta["substeps"]
    return np.stack([p[:, 0] * c + p[:, 1] * s, -p[:, 0] * s + p[:, 1] * c], 1), dt, meta


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traces", nargs="+")
    p.add_argument("--title", default="5 segundos de patins, vistos de cima")
    p.add_argument("--slow", type=float, default=1.0, help="câmera lenta (1 = tempo real)")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--hold", type=float, default=1.5, help="segundos parados no fim")
    p.add_argument("--out", default="outputs/corrida.mp4")
    args = p.parse_args()

    runs = [path_of(np.load(t)) for t in args.traces]
    n = len(runs)
    left, right, top = 250, W - 120, 120
    lane = (H - top - 40) / n
    longest = max(float(np.linalg.norm(r[0][-1])) for r in runs)
    scale = (right - left) / max(longest, 1.0)  # px por cm
    seconds = max(len(r[0]) * r[1] for r in runs)
    f_title, f_lab, f_num, f_small = font(34, True), font(24, True), font(24, True), font(16)
    times = np.arange(0.0, seconds + 1e-9, 1.0 / (args.fps * args.slow))
    times = np.concatenate([times, np.full(int(args.hold * args.fps), times[-1])])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", f"{args.fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                           args.out], stdin=subprocess.PIPE)
    for t in times:
        img = Image.new("RGB", (W, H), tuple(int(c) for c in BG))
        d = ImageDraw.Draw(img)
        d.text((40, 30), args.title, font=f_title, fill=(235, 235, 245))
        d.text((40, 74), f"t = {br(t, 1)} s · tempo real" if args.slow == 1 else f"t = {br(t, 2)} s · câmera lenta "
               f"{args.slow:g}×", font=f_small, fill=(160, 165, 180))
        for cm in range(0, int(longest) + 1, 5):  # régua a cada 5 cm
            x = left + cm * scale
            d.line([(x, top - 10), (x, H - 30)], fill=(32, 36, 46), width=1)
            d.text((x, top - 32), f"{cm} cm", font=f_small, fill=(110, 115, 130), anchor="ma")
        for i, (path, dt, meta) in enumerate(runs):
            y0 = top + lane * (i + 0.5)
            k = min(len(path) - 1, int(t / dt))
            pts = path[: k + 1: 25]
            col = COLORS[i % len(COLORS)]
            xy = [(left + a * scale, y0 - b * scale) for a, b in pts]
            if len(xy) > 1:
                d.line(xy, fill=col, width=4)
            hx, hy = left + path[k, 0] * scale, y0 - path[k, 1] * scale
            d.ellipse([hx - 7, hy - 7, hx + 7, hy + 7], fill=col)
            name = f"#{meta['video_number']:,}".replace(",", ".") if meta.get("video_number") else \
                f"teste {meta.get('test', 0) + 1}"
            d.text((40, y0 - 30), f"Tentativa {name}" if meta.get("video_number") else name, font=f_lab, fill=col)
            d.text((40, y0 + 2), f"geração {meta['generation']}", font=f_small, fill=(150, 155, 170))
            dist = float(np.linalg.norm(path[k]))
            fell = meta["stats"]["fell"] and t >= len(path) * dt - dt
            d.text((min(hx + 14, W - 110), hy - 14), f"{br(dist)} cm" + (" · caiu" if fell else ""), font=f_num,
                   fill=(235, 235, 245))
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{n} tentativas, {len(times)} quadros -> {args.out}")


if __name__ == "__main__":
    main()
