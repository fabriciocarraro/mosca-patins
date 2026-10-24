"""M8: monta um corte do vídeo a partir de um plano em JSON: cartelas de texto, trechos de clipes (saídas do
m8_render.py) e imagens paradas (gráficos), tudo em 1280×720 e no mesmo ritmo de quadros, concatenados com o ffmpeg.
Narração e trilha ficam com o editor.

Cada segmento aceita:
- "card": [título, linhas…] (cartela) | "clip": caminho (com "start" e "seconds") | "image": caminho;
- "seconds": duração (cartela, imagem ou trecho do clipe);
- "caption": texto ou [texto, linha menor] — legenda grande sobre uma faixa escura embaixo (legível no celular);
- "hold": segundos com o último quadro congelado;
- "zoom": true — aproximação lenta na imagem parada.

Uso:
    python scripts/m8_assemble.py docs/m8_corte_2min.json --out outputs/corte_2min.mp4
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
    big, small = font(54, True), font(30)
    sizes = [(big if i == 0 else small) for i in range(len(lines))]
    heights = [f.getbbox(line)[3] - f.getbbox(line)[1] + (30 if i == 0 else 18) for i, (f, line) in
               enumerate(zip(sizes, lines))]
    y = (H - sum(heights)) // 2
    for i, (f, line, h) in enumerate(zip(sizes, lines, heights)):
        d.text((W // 2, y), line, font=f, fill=(235, 235, 245) if i == 0 else (175, 180, 195), anchor="ma")
        y += h
    img.save(path)


def caption_layer(caption, path: Path) -> None:
    """Legenda em camada transparente: faixa escura embaixo, texto grande centralizado (e uma linha menor opcional)."""
    lines = [caption] if isinstance(caption, str) else list(caption)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f_big, f_small = font(40, True), font(28)
    band = 70 + 44 * (len(lines) - 1)
    d.rectangle([0, H - band - 30, W, H], fill=(10, 12, 18, 242))
    y = H - band - 12
    for i, line in enumerate(lines):
        d.text((W // 2, y), line, font=f_big if i == 0 else f_small, fill=(245, 245, 250) if i == 0 else
               (205, 210, 222), anchor="ma")
        y += 52 if i == 0 else 40
    img.save(path)


def segment(seg: dict, k: int, tmp: Path, fps: float) -> Path:
    out = tmp / f"seg{k:03d}.mp4"
    cap = None
    if seg.get("caption"):
        cap = tmp / f"cap{k:03d}.png"
        caption_layer(seg["caption"], cap)
    if "card" in seg or "image" in seg:
        still = tmp / f"seg{k:03d}.png"
        if "card" in seg:
            card(seg["card"], still)
        else:
            Image.open(seg["image"]).convert("RGB").resize((W, H)).save(still)
        if cap is not None:  # na imagem parada, a legenda entra direto no quadro
            base = Image.open(still).convert("RGBA")
            Image.alpha_composite(base, Image.open(cap)).convert("RGB").save(still)
        frames = int(round(seg["seconds"] * fps))
        vf = (f"scale={W * 2}:{H * 2},zoompan=z='min(1+0.0006*on,1.06)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
              f":d={frames}:s={W}x{H}:fps={fps:g}" if seg.get("zoom") else f"scale={W}:{H}")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(still), "-t", f"{seg['seconds']:g}",
               "-vf", vf, *ENCODE, f"{fps:g}", str(out)]
    else:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{seg.get('start', 0):g}"]
        if seg.get("seconds"):
            cmd += ["-t", f"{seg['seconds']:g}"]
        cmd += ["-i", seg["clip"]]
        chain = f"[0:v]scale={W}:{H},setsar=1,fps={fps:g}"
        if seg.get("hold"):
            chain += f",tpad=stop_mode=clone:stop_duration={seg['hold']:g}"
        if cap is not None:
            cmd += ["-loop", "1", "-i", str(cap)]
            graph = f"{chain}[v];[v][1:v]overlay=0:0:shortest=1[o]"
        else:
            graph = f"{chain}[o]"
        cmd += ["-filter_complex", graph, "-map", "[o]", "-an", *ENCODE, f"{fps:g}", str(out)]
    subprocess.run(cmd, check=True)
    return out


def duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


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
        times = [duration(p) for p in parts]
        listing = tmp / "lista.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", args.out], check=True)
    t = 0.0
    for seg, dt in zip(plan["segments"], times):  # minutagem de cada segmento (roteiro da narração)
        what = seg.get("caption") or seg.get("card") or seg.get("clip") or seg.get("image")
        print(f"  {int(t // 60)}:{t % 60:05.2f}  {dt:5.2f} s  {what if isinstance(what, str) else ' / '.join(what)}")
        t += dt
    print(f"{len(parts)} segmentos, {t:.1f} s -> {args.out}")


if __name__ == "__main__":
    main()
