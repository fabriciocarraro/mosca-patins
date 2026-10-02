"""M3: o ritmo do DNg100 no nosso grafo do controlador (6 patas, ~23 mil neurônios).

Mesma receita de Pugliese et al. (parâmetros, escala pelo volume, RK4 de 1 ms, nota de
oscilação), aplicada à rede que o controlador vai usar: o cordão nervoso inteiro ligado às
patas, mais os neurônios descendentes e ascendentes, só com as sinapses do cordão nervoso.
O volume vem da tabela deles quando existe e é estimado pelas sinapses no resto.

Diferença que precisa de calibração: o volume entra relativo a uma referência. No código
deles é a mediana da rede simulada; a nossa tem outra composição (seis patas, mais
sensoriais) e, com a mediana própria, fica em silêncio. `--size-ref` multiplica essa
mediana: um único fator global de excitabilidade. Resultado (DNg100 direito, corrente 400):
1,20× quase silêncio; 1,30× ritmo em 100% das réplicas (~10 Hz, ~7 motores ativos, na pata
da frente e do meio esquerdas, como no experimento publicado); 1,40× disparo descontrolado.
Com os dois DNg100 estimulados não há faixa rítmica (do silêncio direto ao disparo): a
fiação sozinha não coordena as seis patas.

Uso:
    python scripts/download_assets.py --malecns --pugliese --malecns-stats
    python scripts/m3_vnc_weights.py && python scripts/m3_build_graph.py
    python scripts/m3_rhythm_full.py --stim DNg100_R --replicates 4
    python scripts/m3_rhythm_full.py --stim DNg100_R --replicates 1 --save-trace runs/video/ritmo_dng100.npz

`--save-trace` grava a atividade de todos os neurônios da réplica 0 (a cada 2 ms) para o painel do cérebro do vídeo
(`m8_render.py`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.body.fly import LEGS  # noqa: E402
from mosca.brain.graph import Connectome  # noqa: E402
from mosca.brain.pugliese import RateNet, estimate_sizes, rhythm, sample_params, signed_matrix, simulate  # noqa: E402
from mosca.paths import MALECNS_DIR  # noqa: E402


def leg_phases(rates: np.ndarray, groups: list[np.ndarray], period: int, skip: int) -> np.ndarray:
    """Fase de cada pata (fração do ciclo) em relação à primeira, pela correlação cruzada da
    atividade média dos seus neurônios motores."""
    traces = []
    for g in groups:
        x = rates[g, skip:].mean(axis=0)
        traces.append(x - x.mean())
    ref = traces[0]
    out = []
    for x in traces:
        if not ref.any() or not x.any():
            out.append(np.nan)
            continue
        lags = np.arange(-period // 2, period // 2 + 1)
        corr = [np.dot(ref[max(0, -k): len(ref) - max(0, k)], x[max(0, k): len(x) - max(0, -k)]) for k in lags]
        out.append(lags[int(np.argmax(corr))] / period)
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default=str(MALECNS_DIR / "controller_graph_min5.npz"))
    parser.add_argument("--stim", nargs="+", default=["DNg100_R"], help="grupos estimulados (ex.: DNg100_L DNg100_R)")
    parser.add_argument("--current", type=float, default=400.0)
    parser.add_argument("--signs", choices=("consensus", "graph"), default="consensus")
    parser.add_argument("--sizes", choices=("mixed", "estimated"), default="mixed",
                        help="mixed: volume real onde Pugliese tem; estimated: todos estimados")
    parser.add_argument("--size-ref", type=float, default=1.30,
                        help="referência do volume, em múltiplos da mediana da rede (1 = como no código deles)")
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--save-trace", default="", help="grava a atividade de toda a rede na réplica 0 (.npz)")
    args = parser.parse_args()

    c = Connectome.load(Path(args.graph))
    w = signed_matrix(c, args.signs)
    sizes, estimated, model = estimate_sizes(c.body_id, use_real=args.sizes == "mixed")
    reference = args.size_ref * np.nanmedian(sizes)
    print(f"grafo: {c.n} neurônios, {w.nnz:,} ligações com sinal ({args.signs}); "
          f"volume real de {int((~estimated).sum())}, estimado para {int(estimated.sum())} "
          f"(ajuste r = {model.r:.2f}, erro típico ×{model.spread:.2f}); referência {args.size_ref:g}× a mediana")

    motor_groups = [c.groups[f"motor_{leg}"] for leg in LEGS]
    motor = np.concatenate(motor_groups)
    rows = np.cumsum([0] + [len(g) for g in motor_groups])
    local = [np.arange(rows[k], rows[k + 1]) for k in range(len(LEGS))]
    current = np.zeros(c.n)
    for name in args.stim:
        current[c.groups[name]] = args.current
    print(f"estímulo: {', '.join(args.stim)} com corrente {args.current:g}; {len(motor)} neurônios motores das patas")

    h, skip = 1e-3, 0.25
    for k in range(args.replicates):
        net = RateNet(w, sample_params(sizes, np.random.default_rng(args.seed + k), reference=reference))
        t0 = time.perf_counter()
        full = bool(args.save_trace) and k == 0
        rec = simulate(net, current, seconds=args.seconds, h=h, record=None if full else motor)
        rates = rec.rates[motor] if full else rec.rates
        whole = rhythm(rates, rec.peak, h, skip)
        print(f"\nréplica {k}: nota {whole.score:.3f}, {whole.freq_hz:5.2f} Hz, {whole.active_motor} motores ativos, "
              f"{whole.active} neurônios ativos na rede ({time.perf_counter() - t0:.0f} s)")
        period = round(1.0 / (whole.freq_hz * h)) if np.isfinite(whole.freq_hz) and whole.freq_hz > 0 else 0
        phases = leg_phases(rates, local, period, round(skip / h)) if period > 2 else np.full(len(LEGS), np.nan)
        for leg, idx, phase in zip(LEGS, local, phases):
            res = rhythm(rates[idx], rec.peak, h, skip)
            peak_rate = rates[idx].max() if len(idx) else 0.0
            print(f"  {leg:9s} {res.active_motor:3d}/{len(idx):3d} ativos, nota {res.score:.3f}, "
                  f"{res.freq_hz:5.2f} Hz, pico {peak_rate:6.1f} Hz, fase {phase:+.2f}")
        if full:
            every = round(0.002 / h)
            out = Path(args.save_trace)
            out.parent.mkdir(parents=True, exist_ok=True)
            meta = {"mode": "ritmo", "graph": Path(args.graph).name, "stim": args.stim, "current": args.current,
                    "size_ref": args.size_ref, "seed": args.seed, "replicate": k, "control_dt": every * h,
                    "pulse": [0.02, 1.999], "freq_hz": float(whole.freq_hz), "score": float(whole.score),
                    "active_motor": int(whole.active_motor), "untrained": True}
            np.savez_compressed(out, rates=rec.rates[:, 1::every].T.astype(np.float16), body_id=c.body_id,
                                meta=json.dumps(meta, ensure_ascii=False))
            print(f"  atividade de {c.n} neurônios a cada {every * h * 1e3:g} ms -> {out}")


if __name__ == "__main__":
    main()
