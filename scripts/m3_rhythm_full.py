"""M3: o ritmo do DNg100 no nosso grafo do controlador (6 patas, ~23 mil neurônios).

Mesma receita de Pugliese et al. (parâmetros, escala pelo volume, RK4 de 1 ms, nota de
oscilação), aplicada à rede que o controlador vai usar: o cordão nervoso inteiro ligado às
patas, mais os neurônios descendentes e ascendentes. Diferenças para a rede deles (só a
pata da frente): as seis patas; ligações contadas no sistema nervoso inteiro, porque os
arquivos abertos não separam por região; e volume estimado pelas sinapses para os
neurônios fora da tabela deles.

Uso:
    python scripts/download_assets.py --malecns --pugliese --malecns-stats
    python scripts/m3_build_graph.py
    python scripts/m3_rhythm_full.py --stim DNg100_R --replicates 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.body.fly import LEGS  # noqa: E402
from mosca.brain.graph import FAST_SIGN, NEUROTRANSMITTERS, Connectome  # noqa: E402
from mosca.brain.pugliese import RateNet, estimate_sizes, rhythm, sample_params, simulate  # noqa: E402
from mosca.paths import MALECNS_DIR  # noqa: E402


def signed_matrix(c: Connectome, signs: str) -> sp.csr_matrix:
    """Pós × pré com sinal × nº de sinapses. "consensus": só o transmissor de consenso, como
    Pugliese; "graph": o sinal do grafo, que completa os incertos pela previsão."""
    if signs == "consensus":
        nt = pd.read_feather(MALECNS_DIR / NEUROTRANSMITTERS, columns=["body", "consensus_nt"]).set_index("body")
        sign = nt.consensus_nt.reindex(c.body_id).map(FAST_SIGN).fillna(0).to_numpy()
    else:
        sign = c.sign.astype(float)
    values = sign[c.pre] * c.count
    keep = values != 0
    return sp.csr_matrix((values[keep], (c.post[keep], c.pre[keep])), shape=(c.n, c.n))


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
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    c = Connectome.load(Path(args.graph))
    w = signed_matrix(c, args.signs)
    sizes, estimated, model = estimate_sizes(c.body_id, use_real=args.sizes == "mixed")
    print(f"grafo: {c.n} neurônios, {w.nnz:,} ligações com sinal ({args.signs}); "
          f"volume real de {int((~estimated).sum())}, estimado para {int(estimated.sum())} "
          f"(ajuste r = {model.r:.2f}, erro típico ×{model.spread:.2f})")

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
        net = RateNet(w, sample_params(sizes, np.random.default_rng(args.seed + k)))
        t0 = time.perf_counter()
        rec = simulate(net, current, seconds=args.seconds, h=h, record=motor)
        whole = rhythm(rec.rates, rec.peak, h, skip)
        print(f"\nréplica {k}: nota {whole.score:.3f}, {whole.freq_hz:5.2f} Hz, {whole.active_motor} motores ativos, "
              f"{whole.active} neurônios ativos na rede ({time.perf_counter() - t0:.0f} s)")
        period = round(1.0 / (whole.freq_hz * h)) if np.isfinite(whole.freq_hz) and whole.freq_hz > 0 else 0
        phases = leg_phases(rec.rates, local, period, round(skip / h)) if period > 2 else np.full(len(LEGS), np.nan)
        for leg, idx, phase in zip(LEGS, local, phases):
            res = rhythm(rec.rates[idx], rec.peak, h, skip)
            peak_rate = rec.rates[idx].max() if len(idx) else 0.0
            print(f"  {leg:9s} {res.active_motor:3d}/{len(idx):3d} ativos, nota {res.score:.3f}, "
                  f"{res.freq_hz:5.2f} Hz, pico {peak_rate:6.1f} Hz, fase {phase:+.2f}")


if __name__ == "__main__":
    main()
