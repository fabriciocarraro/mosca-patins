"""M7: grafo de controle com o conectoma embaralhado (mesmas ligações e sinais por neurônio).

Uso:
    python scripts/m7_shuffle_graph.py --seed 0   # -> assets/malecns/controller_graph_shuffled_s0_min5.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.brain.graph import Connectome, shuffled_connectome  # noqa: E402
from mosca.paths import MALECNS_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-synapses", type=int, default=5)
    args = parser.parse_args()
    real = Connectome.load(MALECNS_DIR / f"controller_graph_min{args.min_synapses}.npz")
    s = shuffled_connectome(real, seed=args.seed)
    out = MALECNS_DIR / f"controller_graph_shuffled_s{args.seed}_min{args.min_synapses}.npz"
    s.save(out)
    kept = np.isin(s.pre.astype(np.int64) * s.n + s.post, real.pre.astype(np.int64) * real.n + real.post).mean()
    print(f"{len(s.pre):,} ligações; {kept:.1%} coincidem com ligações reais por acaso -> {out}")


if __name__ == "__main__":
    main()
