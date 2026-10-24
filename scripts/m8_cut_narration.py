"""M8: corta a narração gravada numa tomada só em uma fala por cena, sem os erros e recomeços.

Entradas:
- a gravação original;
- as "ilhas" de fala entre pausas (silencedetect do ffmpeg, em ilhas.json);
- o mapa de edição (docs/m8_narracao_edicao.json): pra cada cena, as ilhas mantidas, na ordem (uma ilha também pode
  vir como [índice, início, fim], em segundos da gravação, pra tirar uma palavra da ponta numa micropausa; null mantém a
  borda da ilha). As ilhas que ficam de fora são os erros, recomeços e trechos cortados por tempo, cada uma com o texto
  transcrito ao lado.

Cada fala junta as ilhas mantidas com as pausas originais, limitadas a `--max-pause` (e uma pausa fixa onde uma ilha
foi tirada do meio), com fades curtos nos cortes pra não estalar. Saída: segNN.wav em `--out`, pronta para
`m8_narrate.py --engine files`.

Uso:
    python scripts/m8_cut_narration.py outputs/narracao/gravacao_original/narracao_fabricio.m4a \
        --islands outputs/narracao/gravacao_original/ilhas.json --edit docs/m8_narracao_edicao.json \
        --out outputs/narracao/gravada
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SR = 48000


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio")
    p.add_argument("--islands", required=True)
    p.add_argument("--edit", required=True)
    p.add_argument("--out", default="outputs/narracao/gravada")
    p.add_argument("--pad", type=float, default=0.06, help="margem em volta de cada ilha (s)")
    p.add_argument("--max-pause", type=float, default=0.45, help="pausa máxima entre ilhas mantidas em sequência (s)")
    p.add_argument("--cut-pause", type=float, default=0.32, help="pausa onde uma ilha foi tirada do meio (s)")
    p.add_argument("--tempo", type=float, default=1.0, help="acelera a fala (atempo, sem mudar o tom)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "_original.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", args.audio, "-ac", "1", "-ar", str(SR),
                    str(raw)], check=True)
    sr, x = wavfile.read(raw)
    x = x.astype(np.float32) / 32768.0
    islands = json.loads(Path(args.islands).read_text(encoding="utf-8"))["islands"]
    edit = json.loads(Path(args.edit).read_text(encoding="utf-8"))
    fade = int(0.01 * sr)
    total = 0.0
    for scene in edit["scenes"]:
        keep = [k if isinstance(k, list) else [k, None, None] for k in scene["keep"]]
        pieces = []
        for j, (i, start, end) in enumerate(keep):
            a, b = islands[i]
            lo = a - args.pad if i == 0 else max(a - args.pad, (islands[i - 1][1] + a) / 2)
            hi = b + args.pad if i == len(islands) - 1 else min(b + args.pad, (b + islands[i + 1][0]) / 2)
            if start is not None:  # corte dentro da ilha, numa micropausa
                a, lo = start, start
            if end is not None:
                b, hi = end, end
            seg = x[int(lo * sr):int(hi * sr)].copy()
            seg[:fade] *= np.linspace(0, 1, fade)
            seg[-fade:] *= np.linspace(1, 0, fade)
            if j:
                prev = keep[j - 1][0]
                gap = args.cut_pause if i != prev + 1 else min(args.max_pause, max(0.0, a - islands[prev][1]))
                pieces.append(np.zeros(int(gap * sr), np.float32))
            pieces.append(seg)
        y = np.concatenate(pieces)
        path = out / f"seg{scene['scene']:02d}.wav"
        wavfile.write(path, sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))
        if args.tempo != 1.0:
            tmp = path.with_suffix(".tmp.wav")
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(path), "-af",
                            f"atempo={args.tempo}", str(tmp)], check=True)
            tmp.replace(path)
        dur = len(y) / sr / args.tempo
        total += dur
        print(f"  seg{scene['scene']:02d}  {dur:5.2f} s  ilhas {[k[0] for k in keep]}  {scene['text'][:70]}")
    raw.unlink()
    print(f"{len(edit['scenes'])} falas, {total:.1f} s de fala -> {out}")


if __name__ == "__main__":
    main()
