"""M3: ligações do MaleCNS contando só as sinapses do cordão nervoso (como Pugliese et al.).

Os pesos abertos (connectome-weights) somam as sinapses de cada par no sistema nervoso
inteiro. Entre descendentes e ascendentes, boa parte delas fica no cérebro, que o modelo
não simula, e esses laços levam a rede a disparar sem controle. Este script lê a tabela de
sinapses (syn-partners, 6,8 GB, com a região de cada sinapse), guarda os pares entre
neurônios do recorte do controlador e conta separadamente as sinapses dentro das regiões do
cordão nervoso. A tabela grande é apagada no fim (--keep para manter).

Uso:
    python scripts/m3_vnc_weights.py            # baixa, reduz e grava assets/malecns/connectome-weights-vnc.feather
    python scripts/m3_vnc_weights.py --list-rois
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.brain.graph import ANNOTATIONS, REGION  # noqa: E402
from mosca.paths import MALECNS_DIR  # noqa: E402

URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
SYN_PARTNERS = "syn-partners-male-cns-v1.0-minconf-0.5.feather"
OUT = "connectome-weights-vnc.feather"

# Regiões primárias do cordão nervoso no MaleCNS (nomes do MANC): neurópilos das patas,
# tratos e neurópilos dorsais, abdominal e os centros sensoriais (mVAC, Ov).
VNC_PREFIXES = ("LegNp", "IntTct", "LTct", "NTct", "WTct", "HTct", "ANm", "mVAC", "Ov", "AMNp", "VProN", "ProNm",
                "MesoNm", "MetaNm", "AbNm", "CV", "DProN", "DMetaN", "VNC")


def is_vnc(roi: str) -> bool:
    return roi.startswith(VNC_PREFIXES)


def download(path: Path) -> str:
    sha = hashlib.sha256()
    req = urllib.request.Request(URL + SYN_PARTNERS, headers={"User-Agent": "mosca-patins-download"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(path.with_suffix(".part"), "wb") as out:
        while chunk := resp.read(1 << 22):
            sha.update(chunk)
            out.write(chunk)
    path.with_suffix(".part").replace(path)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-rois", action="store_true", help="só lista as regiões e o nº de sinapses de cada")
    parser.add_argument("--keep", action="store_true", help="não apaga a tabela de sinapses no fim")
    args = parser.parse_args()

    path = MALECNS_DIR / SYN_PARTNERS
    sha = None
    if not path.exists():
        t0 = time.perf_counter()
        sha = download(path)
        print(f"baixado {path.name} ({path.stat().st_size / 1e9:.1f} GB, {time.perf_counter() - t0:.0f} s)")

    ann = pd.read_feather(MALECNS_DIR / ANNOTATIONS, columns=["bodyId", "superclass"])
    region = pa.array(ann.bodyId[ann.superclass.isin(REGION)].to_numpy())
    reader = pa.ipc.open_file(pa.memory_map(str(path)))
    print(f"{reader.num_record_batches} lotes")

    names: list[str] = []
    parts = []
    t0 = time.perf_counter()
    for k in range(reader.num_record_batches):
        batch = reader.get_batch(k).select(["body_pre", "body_post", "primary_post"])
        names = batch.column("primary_post").dictionary.to_pylist()  # dicionário único no arquivo
        if not args.list_rois:
            keep = pc.and_(pc.is_in(batch.column("body_pre"), region), pc.is_in(batch.column("body_post"), region))
            batch = batch.filter(keep)
        if batch.num_rows == 0:
            continue
        codes = batch.column("primary_post").indices.fill_null(-1).to_numpy().astype(np.int32)
        parts.append(pd.DataFrame({"body_pre": batch.column("body_pre").to_numpy(),
                                   "body_post": batch.column("body_post").to_numpy(), "roi": codes}))
        if args.list_rois:
            parts[-1] = parts[-1][["roi"]]
        if (k + 1) % 50 == 0:
            print(f"  lote {k + 1}/{reader.num_record_batches} ({time.perf_counter() - t0:.0f} s)", flush=True)

    syn = pd.concat(parts, ignore_index=True)
    label = np.array(names + ["(sem região)"], dtype=object)
    vnc = np.array([is_vnc(n or "") for n in names] + [False])
    rois = syn.roi.replace(-1, len(names)).value_counts()
    print(f"\nregiões ({'todas as sinapses' if args.list_rois else 'sinapses entre neurônios do recorte'}):")
    for code, c in rois.items():
        print(f"  {'VNC' if vnc[code] else '   '} {label[code]:28s} {c:12,d}")
    if args.list_rois:
        return

    syn["vnc"] = vnc[syn.roi.replace(-1, len(names)).to_numpy()]
    weights = syn.groupby(["body_pre", "body_post"]).agg(weight=("vnc", "size"), weight_vnc=("vnc", "sum")).reset_index()
    weights["weight"] = weights.weight.astype(np.int32)
    weights["weight_vnc"] = weights.weight_vnc.astype(np.int32)
    weights.to_feather(MALECNS_DIR / OUT)
    print(f"\n{len(weights):,} pares no recorte; {weights.weight.sum():,} sinapses, "
          f"{weights.weight_vnc.sum():,} no cordão nervoso ({weights.weight_vnc.sum() / weights.weight.sum():.1%})")

    manifest_path = MALECNS_DIR / "SOURCE.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = {"size": path.stat().st_size, "derived": OUT}
    if sha:
        entry["sha256"] = sha
    manifest["files"][SYN_PARTNERS] = {**manifest["files"].get(SYN_PARTNERS, {}), **entry}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.keep:
        path.unlink()
        print(f"{SYN_PARTNERS} apagado")


if __name__ == "__main__":
    main()
