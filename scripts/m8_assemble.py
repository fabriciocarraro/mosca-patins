"""M8: monta um corte do vídeo a partir de um plano em JSON: cartelas de texto, trechos de clipes (saídas do
m8_render.py) e imagens paradas (gráficos), em 1280×720 e no mesmo ritmo de quadros, concatenados com o ffmpeg, com
narração, legendas e trilha.

Cada segmento aceita:
- "card": [título, linhas…] (cartela) | "clip": caminho (com "start" e "seconds") | "image": caminho;
- "seconds": duração mínima da cena (num clipe que precisa terminar, como uma queda no último quadro, a duração dele);
- "narration": texto falado (áudio do m8_narrate.py). A cena dura o bastante pra fala caber, e o clipe é cortado nessa
  duração; se acabar antes, o último quadro fica congelado. As legendas saem desse texto (ou de "subtitle", se o escrito for diferente do
  falado), em blocos de até duas linhas, no tempo proporcional ao tamanho de cada bloco;
- "caption": legenda fixa (texto ou [texto, linha menor]), pra cortes sem narração;
- "label": texto grande no alto (um número de impacto), a partir de "label_at" segundos da cena;
- "hold": segundos com o último quadro congelado; "zoom": true — aproximação lenta na imagem parada;
- "music": nome do trecho da trilha (m8_music.py) que começa nessa cena.

No topo do plano: "narration_dir" (pasta com segNN.wav e duracoes.json) e "music_lufs" (volume da trilha, em LUFS;
a trilha baixa sozinha quando a voz entra). O áudio sai em −14 LUFS, o volume que YouTube e Instagram usam.

Uso:
    python scripts/m8_assemble.py docs/m8_corte_v3.json --out outputs/corte_v3.mp4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m8_render import BG, H, W, font  # noqa: E402

ENCODE = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r"]
LEAD, TAIL = 0.06, 0.18  # respiro antes e depois de cada fala (s)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


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


def band(lines: list[str], path: Path, sizes: list) -> None:
    """Camada transparente com faixa escura embaixo e as linhas centralizadas."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    heights = [52 if i == 0 else 44 for i in range(len(lines))]
    top = H - sum(heights) - 42
    d.rectangle([0, top, W, H], fill=(10, 12, 18, 242))
    y = top + 18
    for i, (line, f) in enumerate(zip(lines, sizes)):
        d.text((W // 2, y), line, font=f, fill=(245, 245, 250) if i == 0 or len(set(map(id, sizes))) == 1 else
               (205, 210, 222), anchor="ma")
        y += heights[i]
    img.save(path)


def caption_layer(caption, path: Path) -> None:
    """Legenda fixa: texto grande e, opcionalmente, uma linha menor."""
    lines = [caption] if isinstance(caption, str) else list(caption)
    band(lines, path, [font(40, True)] + [font(28)] * (len(lines) - 1))


SUB_WIDTH = 46  # caracteres por linha de legenda (duas linhas por bloco)


def subtitle_layer(text: str, path: Path) -> None:
    """Legenda da narração: até duas linhas do mesmo tamanho."""
    lines = textwrap.wrap(text, width=SUB_WIDTH)
    band(lines, path, [font(36, True)] * len(lines))


def label_layer(text: str, path: Path) -> None:
    """Texto grande no alto, numa caixa escura (número de impacto)."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = font(56, True)
    box = d.textbbox((W // 2, 120), text, font=f, anchor="ma")
    d.rounded_rectangle([box[0] - 24, box[1] - 16, box[2] + 24, box[3] + 16], radius=18, fill=(10, 12, 18, 225))
    d.text((W // 2, 120), text, font=f, fill=(255, 205, 95), anchor="ma")
    img.save(path)


def fits(text: str) -> bool:
    return len(textwrap.wrap(text, width=SUB_WIDTH)) <= 2


def split_fit(text: str) -> list[str]:
    """Quebra uma frase em blocos de até duas linhas, na vírgula (ou no espaço) mais perto do meio."""
    if fits(text):
        return [text]
    mid = len(text) // 2
    cuts = [m.end() for m in re.finditer(r"[,;:]\s", text)] or [m.end() for m in re.finditer(r"\s", text)]
    cut = min(cuts, key=lambda c: abs(c - mid))
    return split_fit(text[:cut].strip()) + split_fit(text[cut:].strip())


def chunks(text: str) -> list[str]:
    """Blocos de legenda de até duas linhas: por frase, frases longas quebradas no meio, e frases curtas juntas."""
    out = []
    for part in re.split(r"(?<=[.!?…])\s+", text.strip()):
        out += split_fit(part)
    merged = []
    for c in out:
        if merged and fits(merged[-1] + " " + c):
            merged[-1] += " " + c
        else:
            merged.append(c)
    return merged


def segment(seg: dict, k: int, tmp: Path, fps: float, dur: float, speech: float) -> Path:
    out = tmp / f"seg{k:03d}.mp4"
    inputs, overlays = [], []
    if "card" in seg or "image" in seg:
        still = tmp / f"seg{k:03d}.png"
        if "card" in seg:
            card(seg["card"], still)
        else:
            Image.open(seg["image"]).convert("RGB").resize((W, H)).save(still)
        frames = int(round(dur * fps))
        base = (f"[0:v]scale={W * 2}:{H * 2},zoompan=z='min(1+0.0006*on,1.06)':x='iw/2-(iw/zoom/2)':"
                f"y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}:fps={fps:g}" if seg.get("zoom") else
                f"[0:v]scale={W}:{H},fps={fps:g}")
        inputs += ["-loop", "1", "-t", f"{dur:.3f}", "-i", str(still)]
    else:
        start = seg.get("start", 0)
        avail = duration(Path(seg["clip"])) - start
        take = min(avail, dur)  # o clipe vai até onde a cena precisa; se acabar antes, o último quadro fica
        inputs += ["-ss", f"{start:g}", "-t", f"{take:.3f}", "-i", seg["clip"]]
        base = f"[0:v]scale={W}:{H},setsar=1,fps={fps:g}"
        hold = max(seg.get("hold", 0.0), dur - take)
        if hold > 0:
            base += f",tpad=stop_mode=clone:stop_duration={hold:.3f}"
    if seg.get("label"):  # "label_at": quando o texto grande aparece (s desde o começo da cena)
        png = tmp / f"lab{k:03d}.png"
        label_layer(seg["label"], png)
        overlays.append((png, (seg["label_at"], dur) if seg.get("label_at") else None))
    if seg.get("caption"):
        png = tmp / f"cap{k:03d}.png"
        caption_layer(seg["caption"], png)
        overlays.append((png, None))
    text = seg.get("subtitle") or seg.get("narration")
    if text and speech > 0:  # blocos no tempo da fala, proporcionais ao tamanho
        parts = chunks(text)
        total = sum(len(p) for p in parts)
        t = LEAD
        for j, part in enumerate(parts):
            span = speech * len(part) / total
            png = tmp / f"sub{k:03d}_{j}.png"
            subtitle_layer(part, png)
            overlays.append((png, (t, t + span + (TAIL if j == len(parts) - 1 else 0.05))))
            t += span
    for png, _ in overlays:
        inputs += ["-loop", "1", "-i", str(png)]
    graph, last = base + "[v0]", "v0"
    for j, (_, when) in enumerate(overlays):
        enable = f":enable='between(t,{when[0]:.3f},{when[1]:.3f})'" if when else ""
        graph += f";[{last}][{j + 1}:v]overlay=0:0{enable}[v{j + 1}]"
        last = f"v{j + 1}"
    run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", f"[{last}]",
         "-t", f"{dur:.3f}", "-an", *ENCODE, f"{fps:g}", str(out)])
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan")
    p.add_argument("--out", default="outputs/corte.mp4")
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    fps = plan.get("fps", 30)
    ndir = Path(plan["narration_dir"]) if plan.get("narration_dir") else None
    speech = {int(k): v for k, v in json.loads((ndir / "duracoes.json").read_text()).items()} if ndir else {}
    durs = []
    for k, seg in enumerate(plan["segments"]):  # duração de cada cena: a mínima pedida, ou o bastante pra fala caber
        want = seg.get("seconds")
        if want is None and "clip" in seg and k not in speech:  # clipe sem fala e sem duração: inteiro
            want = duration(Path(seg["clip"])) - seg.get("start", 0)
        want = (want or 0) + seg.get("hold", 0)
        durs.append(max(want, speech[k] + LEAD + TAIL) if k in speech else want)
    starts = [sum(durs[:k]) for k in range(len(durs))]
    total = sum(durs)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        parts = [segment(seg, k, tmp, fps, durs[k], speech.get(k, 0.0)) for k, seg in enumerate(plan["segments"])]
        listing = tmp / "lista.txt"
        listing.write_text("".join(f"file '{q.as_posix()}'\n" for q in parts), encoding="utf-8")
        video = tmp / "video.mp4"
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
             str(video)])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tags = [(starts[k], seg["music"]) for k, seg in enumerate(plan["segments"]) if seg.get("music")]
        if not speech and not tags:
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-c", "copy", args.out])
        else:  # áudio em etapas, com arquivos no meio (um grafo só, com a voz dividida em dois ramos, travava)
            final = "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=44100,alimiter=limit=0.95"  # o volume que as redes usam
            mix = tmp / "mix.wav"
            narr = tmp / "narracao.wav"
            if speech:  # cada fala no seu lugar; voz normalizada em −16 LUFS
                inputs, graph = [], []
                for j, k in enumerate(sorted(speech)):
                    ms = int(round((starts[k] + LEAD) * 1000))
                    inputs += ["-i", str(ndir / f"seg{k:02d}.wav")]
                    graph.append(f"[{j}:a]adelay={ms}|{ms},aformat=channel_layouts=stereo[n{j}]")
                graph.append(f"{''.join(f'[n{j}]' for j in range(len(speech)))}amix=inputs={len(speech)}:normalize=0:"
                             f"dropout_transition=0,apad,atrim=0:{total:.3f},loudnorm=I=-16:TP=-2:LRA=9,"
                             f"aresample=44100[o]")
                run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(graph), "-map", "[o]",
                     str(narr)])
            if tags:
                raw, music = tmp / "musica_crua.wav", tmp / "musica.wav"
                run([sys.executable, str(Path(__file__).with_name("m8_music.py")), "--total", f"{total:.2f}",
                     "--sections", ",".join(f"{t:.2f}:{name}" for t, name in tags), "--out", str(raw)])
                lufs = plan.get("music_lufs", -24)  # trilha bem abaixo da voz (−16), e ainda mais baixa sob a fala
                run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-af",
                     f"loudnorm=I={lufs}:TP=-3,aresample=44100", str(music)])
                if speech:  # a trilha baixa quando a voz entra, e as duas se somam
                    duck = tmp / "musica_abaixada.wav"
                    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(music), "-i", str(narr), "-filter_complex",
                         "[0:a][1:a]sidechaincompress=threshold=0.02:ratio=6:attack=30:release=500[o]", "-map", "[o]",
                         str(duck)])
                    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(duck), "-i", str(narr), "-filter_complex",
                         f"[0:a][1:a]amix=inputs=2:normalize=0,{final}[o]", "-map", "[o]", str(mix)])
                else:
                    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(music), "-af", final, str(mix)])
            else:
                run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(narr), "-af", final, str(mix)])
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(mix), "-map", "0:v", "-map", "1:a",
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", args.out])
    for k, seg in enumerate(plan["segments"]):  # minutagem de cada cena (roteiro da narração)
        what = seg.get("narration") or seg.get("caption") or seg.get("card") or seg.get("clip") or seg.get("image")
        t = starts[k]
        print(f"  {int(t // 60)}:{t % 60:05.2f}  {durs[k]:5.2f} s  {what if isinstance(what, str) else ' / '.join(what)}")
    print(f"{len(parts)} cenas, {total:.1f} s -> {args.out}")


if __name__ == "__main__":
    main()
