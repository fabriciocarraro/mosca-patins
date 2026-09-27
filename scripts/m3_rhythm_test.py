"""M3: reproduz o ritmo de passos do DNg100 (Pugliese et al. 2026) na rede do MaleCNS deles.

Estimula só o DNg100 que inerva a pata da frente esquerda (linha 9 da tabela, corrente 400,
como no experimento publicado) por 2 s e mede o ritmo dos neurônios motores das patas.
Publicado para essa rede (1024 réplicas): 100% rítmicas, nota média 0,985, ~11 Hz,
8 neurônios motores ativos, ~115 neurônios ativos.

Uso:
    python scripts/download_assets.py --pugliese
    python scripts/m3_rhythm_test.py --replicates 8
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
from mosca.brain.pugliese import RateNet, rhythm, sample_params, simulate  # noqa: E402
from mosca.paths import PUGLIESE_DIR  # noqa: E402

STIM_ROW, STIM_CURRENT = 9, 400.0  # DNg100 (corpo 10056), como na execução publicada


def load_network() -> tuple[sp.csr_matrix, pd.DataFrame]:
    table = pd.read_csv(PUGLIESE_DIR / "wTable_20260210_vncRoisOnly.csv", index_col=0)
    w = pd.read_csv(PUGLIESE_DIR / "W_20260210_vncRoisOnly.csv").drop(columns="bodyId_pre").to_numpy()
    return sp.csr_matrix(w.T.astype(float)), table  # linhas do CSV = pré; aqui pós × pré


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    w, table = load_network()
    motor = np.flatnonzero(table["motor module"].notna().to_numpy())
    print(f"rede: {w.shape[0]} neurônios, {w.nnz:,} ligações; {len(motor)} neurônios motores com módulo anotado")
    print(f"estímulo: {table.iloc[STIM_ROW]['type']} (corpo {table.iloc[STIM_ROW]['bodyId']}), corrente {STIM_CURRENT:g}")
    current = np.zeros(w.shape[0])
    current[STIM_ROW] = STIM_CURRENT

    results = []
    for k in range(args.replicates):
        rng = np.random.default_rng(args.seed + k)
        net = RateNet(w, sample_params(table["size"].to_numpy(), rng))
        t0 = time.perf_counter()
        rec = simulate(net, current)
        res = rhythm(rec.rates[motor], rec.peak)
        results.append(res)
        print(f"réplica {k}: nota {res.score:.3f}, {res.freq_hz:5.2f} Hz, {res.active_motor} motores ativos, "
              f"{res.active} neurônios ativos, DNg100 a {rec.rates[STIM_ROW, -1]:.1f} Hz ({time.perf_counter() - t0:.1f} s)")

    scores = np.array([r.score for r in results])
    print(f"\nnota média {scores.mean():.3f} (publicado 0,985); rítmicas (nota ≥ 0,5): {(scores >= 0.5).mean():.0%} (publicado 100%)")
    print(f"frequência mediana {np.nanmedian([r.freq_hz for r in results]):.1f} Hz (plano: 7 a 15 Hz); "
          f"motores ativos mediana {np.median([r.active_motor for r in results]):.0f} (publicado 8); "
          f"neurônios ativos mediana {np.median([r.active for r in results]):.0f} (publicado ~115)")


if __name__ == "__main__":
    main()
