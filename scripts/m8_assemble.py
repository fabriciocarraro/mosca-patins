"""M8: monta um corte do vídeo a partir de um plano em JSON: cartelas de texto, trechos de clipes (saídas do
m8_render.py) e imagens paradas (gráficos), tudo em 1280×720 e no mesmo ritmo de quadros, concatenados com o ffmpeg.
Narração e trilha ficam com o editor.

Plano (JSON):
    {"fps": 30, "segments": [
        {"card": ["Tentativa #1", "a mosca nunca viu um patim"], "seconds": 3},
        {"clip": "outputs/tentativa_1.mp4", "start": 0, "seconds": 6},
        {"image": "outputs/marcha_fases.png", "seconds": 8}
    ]}

Uso:
    python scripts/m8_assemble.py docs/m8_corte.json --out outputs/corte.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m8_render import BG, H, W, font  # noqa: E402

ENCODE = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r"]


def card(lines: list[str], path: Path) -> None:
    """Cartela: a primeira linha grande, as outras menores, centralizadas no fundo do vídeo."""
    img = Image.new("RGB", (W, H), tuple(int(c) for c in BG))
    d = ImageDraw.Draw(img)
    big, small = font(54, True), font(28)
    sizes = [(big if i == 0 else small) for i in range(len(lines))]
    heights = [f.getbbox(line)[3] - f.getbbox(line)[1] + (28 if i == 0 else 16) for i, (f, line) in
               enumerate(zip(sizes, lines))]
    y = (H - sum(heights)) // 2
    for i, (f, line, h) in enumerate(zip(sizes, lines, heights)):
        d.text((W // 2, y), line, font=f, fill=(235, 235, 245) if i == 0 else (170, 175, 190), anchor="ma")
        y += h
    img.save(path)


def segment(seg: dict, k: int, tmp: Path, fps: float) -> Path:
    out = tmp / f"seg{k:03d}.mp4"
    if "card" in seg or "image" in seg:
        still = tmp / f"seg{k:03d}.png"
        if "card" in seg:
            card(seg["card"], still)
        else:
            Image.open(seg["image"]).convert("RGB").resize((W, H)).save(still)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(still), "-t", f"{seg['seconds']:g}",
               "-vf", f"scale={W}:{H}", *ENCODE, f"{fps:g}", str(out)]
    else:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{seg.get('start', 0):g}", "-i", seg["clip"]]
        if seg.get("seconds"):
            cmd += ["-t", f"{seg['seconds']:g}"]
        cmd += ["-vf", f"scale={W}:{H}", "-an", *ENCODE, f"{fps:g}", str(out)]
    subprocess.run(cmd, check=True)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan")
    p.add_argument("--out", default="outputs/corte.mp4")
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    fps = plan.get("fps", 30)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        parts = [segment(seg, k, tmp, fps) for k, seg in enumerate(plan["segments"])]
        listing = tmp / "lista.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", args.out], check=True)
    seconds = sum(seg.get("seconds", 0) for seg in plan["segments"])
    print(f"{len(plan['segments'])} segmentos (~{seconds:g} s sem contar clipes inteiros) -> {args.out}")


if __name__ == "__main__":
    main()
