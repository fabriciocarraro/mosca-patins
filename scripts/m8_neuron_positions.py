"""M8: posição de cada neurônio do controlador, para o painel do cérebro.

Duas posições por neurônio, nas coordenadas do MaleCNS (voxels de 8 nm):
- corpo celular: `somaLocation` das anotações (ou `tosomaLocation`), que falta nos axônios sensoriais (o corpo
  deles fica na pata, fora do volume) e em parte dos outros;
- centro das sinapses: média das posições das sinapses de entrada e de saída do neurônio, da tabela de sinapses
  do MaleCNS (`syn-partners`, versão só com as ligações significativas, ~3 GB). Todo neurônio do controlador tem.

A tabela de sinapses é lida em blocos (memória pequena) e apagada no fim, a menos de `--keep`.

Saída: assets/malecns/controller_positions.npz (body_id, soma, syn_centroid, syn_pre, syn_post, n_pre, n_post).

Uso (no Spark, ~3 GB de download):
    python scripts/m8_neuron_positions.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_assets import download_malecns  # noqa: E402
from mosca.paths import MALECNS_DIR  # noqa: E402

SYNAPSES = "syn-partners-male-cns-v1.0-minconf-0.5-significant-only.feather"
ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", default=str(MALECNS_DIR / "controller_graph_min5.npz"))
    p.add_argument("--keep", action="store_true", help="não apaga a tabela de sinapses no fim")
    args = p.parse_args()

    body_id = np.load(args.graph, allow_pickle=True)["body_id"].astype(np.int64)
    order = np.argsort(body_id)
    sorted_ids = body_id[order]

    ann = pd.read_feather(MALECNS_DIR / ANNOTATIONS, columns=["bodyId", "somaLocation", "tosomaLocation"])
    ann = ann.set_index("bodyId").reindex(body_id)
    soma = np.full((len(body_id), 3), np.nan)
    for col in ("tosomaLocation", "somaLocation"):  # o corpo celular anotado vale mais que o "rumo ao corpo"
        has = ann[col].notna().to_numpy()
        soma[has] = np.stack(ann[col][has].to_numpy()).astype(float)

    download_malecns((SYNAPSES,))
    path = MALECNS_DIR / SYNAPSES
    sums = {side: np.zeros((len(body_id), 3)) for side in ("pre", "post")}
    counts = {side: np.zeros(len(body_id), np.int64) for side in ("pre", "post")}
    ids_arrow = pa.array(sorted_ids)
    with pa.memory_map(str(path)) as source:
        reader = ipc.open_file(source)
        for b in range(reader.num_record_batches):
            batch = reader.get_batch(b)
            for side in ("pre", "post"):
                bodies = batch.column(f"body_{side}")
                mask = pc.is_in(bodies, value_set=ids_arrow).to_numpy(zero_copy_only=False)
                if not mask.any():
                    continue
                idx = order[np.searchsorted(sorted_ids, bodies.to_numpy()[mask])]
                xyz = np.stack([batch.column(f"{c}_{side}").to_numpy()[mask] for c in "xyz"], axis=1).astype(float)
                np.add.at(sums[side], idx, xyz)
                np.add.at(counts[side], idx, 1)
            if b % 200 == 0:
                print(f"bloco {b}/{reader.num_record_batches}: {counts['pre'].sum():,} sinapses de saída e "
                      f"{counts['post'].sum():,} de entrada dos neurônios do controlador", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        pre = sums["pre"] / counts["pre"][:, None]
        post = sums["post"] / counts["post"][:, None]
        both = (sums["pre"] + sums["post"]) / (counts["pre"] + counts["post"])[:, None]
    out = MALECNS_DIR / "controller_positions.npz"
    np.savez_compressed(out, body_id=body_id, soma=soma, syn_centroid=both, syn_pre=pre, syn_post=post,
                        n_pre=counts["pre"], n_post=counts["post"])
    has_syn = (counts["pre"] + counts["post"]) > 0
    print(f"{len(body_id):,} neurônios: corpo celular em {np.isfinite(soma[:, 0]).mean():.0%}, centro das sinapses "
          f"em {has_syn.mean():.0%} -> {out}")
    if not args.keep:
        path.unlink()
        print(f"{SYNAPSES} apagado")


if __name__ == "__main__":
    main()
