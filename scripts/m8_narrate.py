"""M8: narração do corte: um áudio por segmento, a partir do texto falado ("narration") do plano em JSON.

Motores:
- "sapi": vozes instaladas no Windows (System.Speech), sem baixar nada. A pt-BR disponível ("Microsoft Maria Desktop")
  soa robótica: serve pra cronometrar o corte, não pra publicar.
- "edge": vozes neurais da Microsoft (pacote edge-tts, que manda o texto pro serviço on-line da Microsoft).
- "files": áudios gravados ou gerados fora daqui, um por segmento, em `--dir` (segNN.wav, NN = índice do segmento).

Saída: outputs/narracao/<motor>/segNN.wav e duracoes.json ({índice: segundos}), lidos pelo m8_assemble.py.

Uso:
    python scripts/m8_narrate.py docs/m8_corte_v3.json --engine sapi --rate 1
    python scripts/m8_narrate.py docs/m8_corte_v3.json --engine edge --voice pt-BR-AntonioNeural
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SAPI = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('{voice}')
$s.Rate = {rate}
$text = Get-Content -LiteralPath '{text}' -Encoding UTF8 -Raw
$s.SetOutputToWaveFile('{wav}')
$s.Speak($text)
$s.Dispose()
"""


def duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def to_wav(src: Path, dst: Path) -> None:
    """Padroniza: 44,1 kHz mono, sem silêncio no começo e no fim."""
    trim = "silenceremove=start_periods=1:start_threshold=-50dB,areverse,silenceremove=start_periods=1:" \
           "start_threshold=-50dB,areverse"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-af", trim, "-ar", "44100", "-ac", "1",
                    str(dst)], check=True)


def sapi(text: str, wav: Path, voice: str, rate: int) -> None:
    with tempfile.TemporaryDirectory() as d:
        txt, raw, ps1 = Path(d) / "t.txt", Path(d) / "raw.wav", Path(d) / "s.ps1"
        txt.write_text(text, encoding="utf-8")
        ps1.write_text(SAPI.format(voice=voice, rate=rate, text=txt, wav=raw), encoding="utf-8-sig")
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)], check=True)
        to_wav(raw, wav)


def edge(text: str, wav: Path, voice: str, rate: int) -> None:
    import edge_tts  # só com o pacote instalado (serviço on-line da Microsoft)

    with tempfile.TemporaryDirectory() as d:
        mp3 = Path(d) / "raw.mp3"
        asyncio.run(edge_tts.Communicate(text, voice, rate=f"{rate:+d}%").save(str(mp3)))
        to_wav(mp3, wav)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan")
    p.add_argument("--engine", choices=("sapi", "edge", "files"), default="sapi")
    p.add_argument("--voice", default="", help="padrão: Microsoft Maria Desktop (sapi) ou pt-BR-AntonioNeural (edge)")
    p.add_argument("--rate", type=int, default=0, help="velocidade: −10..10 no sapi; % no edge")
    p.add_argument("--dir", default="", help="motor files: pasta com segNN.wav")
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    voice = args.voice or {"sapi": "Microsoft Maria Desktop", "edge": "pt-BR-AntonioNeural"}.get(args.engine, "")
    out = Path("outputs") / "narracao" / args.engine
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache.json"
    done = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    durations = {}
    for k, seg in enumerate(plan["segments"]):
        text = seg.get("narration")
        if not text:
            continue
        wav = out / f"seg{k:02d}.wav"
        key = hashlib.sha256(f"{args.engine}|{voice}|{args.rate}|{text}".encode()).hexdigest()[:16]
        if args.engine == "files":
            shutil.copy(Path(args.dir) / f"seg{k:02d}.wav", wav)
        elif done.get(str(k)) != key or not wav.exists():  # só regera o que mudou
            (sapi if args.engine == "sapi" else edge)(text, wav, voice, args.rate)
            done[str(k)] = key
        durations[k] = duration(wav)
        print(f"  seg{k:02d}  {durations[k]:5.2f} s  {len(text.split()):3d} palavras  {text[:70]}")
    cache.write_text(json.dumps(done, indent=1), encoding="utf-8")
    (out / "duracoes.json").write_text(json.dumps(durations, indent=1), encoding="utf-8")
    total = sum(durations.values())
    print(f"{len(durations)} trechos, {total:.1f} s de fala ({voice}) -> {out}")


if __name__ == "__main__":
    main()
