"""M8: trilha de fundo sintetizada aqui mesmo (sem amostras de terceiros, sem licença a pagar), com trechos que
acompanham a história: gancho calmo, explicação curiosa, tensão na brecha, crescendo no aprendizado, clímax cheio,
provas mais contidas e fecho. Um andamento só (96 bpm, lá menor); as trocas de trecho caem no compasso mais próximo.

Instrumentos: pad (serras desafinadas com filtro), dedilhado (Karplus-Strong), baixo, bumbo, caixa, chimbal e um
"riser" de ruído antes do clímax, com reverberação por convolução. Semente fixa: a mesma trilha sempre.

Uso:
    python scripts/m8_music.py --total 104 --sections "0:intro,8:setup,33:tension,41:build,60:climax,74:proofs,90:outro" \
        --out outputs/musica.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, lfilter

SR = 44100
BPM = 96
BEAT = 60.0 / BPM
BAR = 4 * BEAT
SEMI = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}
CHORDS = {"Am": [("A", 3), ("C", 4), ("E", 4)], "F": [("F", 3), ("A", 3), ("C", 4)], "C": [("C", 4), ("E", 4), ("G", 4)],
          "G": [("G", 3), ("B", 3), ("D", 4)], "E": [("E", 3), ("G#", 3), ("B", 3)], "Dm": [("D", 4), ("F", 4), ("A", 4)]}
PROGRESSION = {"intro": ["Am", "F", "C", "G"], "setup": ["Am", "F", "C", "G"], "tension": ["Am", "E", "Dm", "E"],
               "build": ["F", "G", "Am", "Am"], "climax": ["F", "G", "C", "Am"], "proofs": ["Am", "F", "C", "G"],
               "outro": ["F", "G", "Am", "Am"]}
LAYERS = {  # volume de cada camada por trecho; div = notas do dedilhado por tempo
    "intro": dict(pad=0.9, arp=0.25, div=1),
    "setup": dict(pad=0.7, arp=0.6, div=2),
    "tension": dict(pad=0.55, arp=0.7, div=4, low=True, bass=0.5),
    "build": dict(pad=0.7, arp=0.7, div=2, bass=0.7, kick=0.6),
    "climax": dict(pad=0.9, arp=0.75, div=4, bass=0.9, kick=0.9, snare=0.7, hat=0.5),
    "proofs": dict(pad=0.6, arp=0.5, div=2, bass=0.45, kick=0.35),
    "outro": dict(pad=0.7, arp=0.3, div=1),
}
rng = np.random.default_rng(20261024)


def hz(name: str, octave: int) -> float:
    return 440.0 * 2 ** ((SEMI[name] + 12 * (octave - 4)) / 12)


def envelope(n: int, attack: float, release: float) -> np.ndarray:
    e = np.ones(n)
    na, nr = min(n, int(attack * SR)), min(n, int(release * SR))
    if na:
        e[:na] = np.linspace(0.0, 1.0, na)
    if nr:
        e[-nr:] *= np.linspace(1.0, 0.0, nr)
    return e


def lowpass(x: np.ndarray, fc: float) -> np.ndarray:
    b, a = butter(2, fc / (SR / 2))
    return lfilter(b, a, x)


def pad_note(f: float, dur: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = sum(2 * ((f * (1 + d) * t) % 1.0) - 1 for d in (-0.004, 0.0, 0.0045)) / 3
    return lowpass(x, 1100) * envelope(n, 0.6, 0.9)


def pluck(f: float, dur: float, decay: float = 0.995) -> np.ndarray:
    n, period = int(dur * SR), max(2, int(SR / f))
    y = rng.uniform(-1, 1, period)
    out = np.zeros(n)
    for pos in range(0, n, period):
        m = min(period, n - pos)
        out[pos:pos + m] = y[:m]
        y = decay * 0.5 * (y + np.roll(y, -1))
    return lowpass(out, 3500) * envelope(n, 0.002, 0.08)


def bass_note(f: float, dur: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    return np.tanh(1.4 * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t))) * envelope(n, 0.01, 0.1)


def kick() -> np.ndarray:
    t = np.arange(int(0.35 * SR)) / SR
    return np.sin(2 * np.pi * np.cumsum(45 + 75 * np.exp(-t / 0.035)) / SR) * np.exp(-t / 0.13)


def snare() -> np.ndarray:
    t = np.arange(int(0.25 * SR)) / SR
    b, a = butter(2, [1500 / (SR / 2), 6500 / (SR / 2)], btype="band")
    return lfilter(b, a, rng.uniform(-1, 1, len(t))) * np.exp(-t / 0.06) + 0.4 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)


def hat() -> np.ndarray:
    t = np.arange(int(0.06 * SR)) / SR
    b, a = butter(2, 7000 / (SR / 2), btype="high")
    return lfilter(b, a, rng.uniform(-1, 1, len(t))) * np.exp(-t / 0.015)


def add(buf: np.ndarray, x: np.ndarray, t: float, gain: float, pan: float = 0.0) -> None:
    i = int(t * SR)
    if i >= buf.shape[1]:
        return
    x = x[: buf.shape[1] - i] * gain
    buf[0, i:i + len(x)] += x * np.sqrt(0.5 * (1 - pan))
    buf[1, i:i + len(x)] += x * np.sqrt(0.5 * (1 + pan))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--total", type=float, required=True, help="duração (s)")
    p.add_argument("--sections", required=True, help="início:trecho separados por vírgula (intro, setup, tension, "
                                                       "build, climax, proofs, outro)")
    p.add_argument("--out", default="outputs/musica.wav")
    args = p.parse_args()
    sections = sorted((float(s.split(":")[0]), s.split(":")[1]) for s in args.sections.split(","))
    n = int((args.total + 4) * SR)
    dry, wet = np.zeros((2, n)), np.zeros((2, n))  # o "wet" passa pela reverberação

    def section_at(t: float) -> str:
        return [name for start, name in sections if start <= t + 1e-9][-1]

    bars = int(np.ceil(args.total / BAR)) + 1
    climax = next((start for start, name in sections if name == "climax"), None)
    for b in range(bars):
        t0 = b * BAR
        name = section_at(t0)
        lay = LAYERS[name]
        prog = PROGRESSION[name]
        first = next(start for start, nm in reversed(sections) if start <= t0 + 1e-9)
        chord = CHORDS[prog[int((t0 - first) // BAR) % len(prog)]]
        for k, (note, octv) in enumerate(chord):  # pad: o acorde inteiro, ligado ao compasso seguinte
            add(wet, pad_note(hz(note, octv), BAR + 0.9), t0, 0.07 * lay["pad"], pan=(k - 1) * 0.5)
        if lay.get("arp"):
            tones = [hz(nt, o - (1 if lay.get("low") else 0)) for nt, o in chord]
            tones = tones + [2 * f for f in tones]
            pattern = [0, 1, 2, 3, 4, 3, 2, 1]
            steps = int(4 * lay["div"])
            for s in range(steps):
                f = tones[pattern[s % len(pattern)] % len(tones)]
                add(wet, pluck(f, 0.7 if lay["div"] <= 2 else 0.35), t0 + s * BEAT / lay["div"], 0.11 * lay["arp"],
                    pan=0.4 if s % 2 else -0.4)
        root = hz(chord[0][0], 2)
        for beat in range(4):
            tb = t0 + beat * BEAT
            if lay.get("bass") and (name != "tension" or beat % 2 == 0):
                add(dry, bass_note(root, BEAT * 0.95), tb, 0.22 * lay["bass"])
            if lay.get("kick") and (name == "climax" or beat % 2 == 0) and not (name == "proofs" and beat):
                add(dry, kick(), tb, 0.55 * lay["kick"])
            if lay.get("snare") and beat % 2 == 1:
                add(wet, snare(), tb, 0.22 * lay["snare"], pan=0.05)
            if lay.get("hat"):
                add(dry, hat(), tb + BEAT / 2, 0.10 * lay["hat"], pan=0.3)
    if climax is not None and climax > 2 * BAR:  # riser de ruído nos dois compassos antes do clímax
        m = int(2 * BAR * SR)
        b, a = butter(2, [600 / (SR / 2), 5000 / (SR / 2)], btype="band")
        riser = lfilter(b, a, rng.uniform(-1, 1, m)) * np.linspace(0, 1, m) ** 2
        add(wet, riser, climax - 2 * BAR, 0.12)
    t = np.arange(int(1.8 * SR)) / SR  # reverberação: ruído com queda exponencial, abafado
    ir = lowpass(rng.uniform(-1, 1, len(t)) * np.exp(-t / 0.45), 5000)
    ir /= np.sqrt(np.sum(ir ** 2))
    rev = np.stack([fftconvolve(wet[c], ir, mode="full")[:n] for c in range(2)])
    mix = dry + wet + 0.35 * rev
    total = int(args.total * SR)
    mix = mix[:, :total]
    fade_in, fade_out = int(0.8 * SR), int(3.0 * SR)
    mix[:, :fade_in] *= np.linspace(0, 1, fade_in)
    mix[:, -fade_out:] *= np.linspace(1, 0, fade_out)
    mix = np.tanh(1.2 * mix / max(1e-9, np.abs(mix).max())) * 0.9
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(args.out, SR, (mix.T * 32767).astype(np.int16))
    print(f"trilha de {args.total:.1f} s, {len(sections)} trechos ({', '.join(nm for _, nm in sections)}) -> {args.out}")


if __name__ == "__main__":
    main()
