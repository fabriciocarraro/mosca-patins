"""M3: monta o grafo do controlador a partir do MaleCNS e confere as contagens.

Uso (precisa de ~6 GB de memória para ler as ligações; rode no Spark):
    python scripts/download_assets.py --malecns --malecns-weights
    python scripts/m3_build_graph.py --min-synapses 5
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.body.fly import LEGS  # noqa: E402
from mosca.brain.graph import build_connectome  # noqa: E402
from mosca.paths import MALECNS_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-synapses", type=int, default=5)
    args = parser.parse_args()

    t0 = time.perf_counter()
    c = build_connectome(args.min_synapses)
    out = MALECNS_DIR / f"controller_graph_min{args.min_synapses}.npz"
    c.save(out)
    print(f"grafo: {c.n:,} neurônios, {len(c.pre):,} ligações (≥{args.min_synapses} sinapses), "
          f"{int(c.count.sum()):,} sinapses ({time.perf_counter() - t0:.0f} s) -> {out}")
    print("por superclasse:", dict(Counter(c.superclass.tolist()).most_common()))
    signs = Counter(c.sign.tolist())
    print(f"sinal: {signs.get(1, 0):,} excitatórios, {signs.get(-1, 0):,} inibitórios, {signs.get(0, 0):,} sem ligação rápida")
    print("\npata        motores  sensoriais")
    for leg in LEGS:
        print(f"{leg:10s} {len(c.groups[f'motor_{leg}']):8d} {len(c.groups[f'sensory_{leg}']):11d}")
    total_motor = sum(len(c.groups[f"motor_{leg}"]) for leg in LEGS)
    print(f"total de motores das patas: {total_motor} (esperado 381 antes da poda)")
    print("\ncomando:", {k: len(v) for k, v in c.groups.items() if not k.startswith(("motor_", "sensory_"))})
    missing = [f"{t}_{s}" for t in ("DNg100", "DNa01", "DNa02", "MDN") for s in "LR" if f"{t}_{s}" not in c.groups]
    print("descendentes de comando fora do grafo (sem caminho até as patas):", missing or "nenhum")
    in_deg = np.bincount(c.post, minlength=c.n)
    print(f"entradas por neurônio: mediana {int(np.median(in_deg))}, máximo {int(in_deg.max())}")


if __name__ == "__main__":
    main()
