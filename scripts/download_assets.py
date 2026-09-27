"""Baixa os assets de terceiros para assets/, com versões fixadas e checagem de integridade.

Uso:
    python scripts/download_assets.py                    # modelo da mosca (flybody)
    python scripts/download_assets.py --walking-sample   # + 100 trechos de moscas reais andando
    python scripts/download_assets.py --malecns          # + anotações e neurotransmissores do MaleCNS
    python scripts/download_assets.py --malecns-weights  # + ligações do MaleCNS (1,1 GB)
    python scripts/download_assets.py --malecns-stats    # + sinapses por neurônio do MaleCNS (baixa 778 MB, guarda 2 MB)
    python scripts/download_assets.py --pugliese         # + rede do MaleCNS usada por Pugliese et al. (75 MB)
    python scripts/download_assets.py --walking-policy   # + política de caminhada do flybody (5 MB)

Os arquivos do flybody são conferidos pelo hash de blob do Git informado pela API do GitHub,
então rodar de novo só baixa o que faltar ou estiver corrompido. A amostra de caminhada é
extraída de dentro do zip de 3 GB do figshare lendo só os trechos necessários (HTTP Range).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.paths import ASSETS, FLYBODY_DATA_DIR, FLYBODY_DIR, MALECNS_DIR, PUGLIESE_DIR, WALKING_SAMPLE  # noqa: E402

FLYBODY_REPO = "TuragaLab/flybody"
FLYBODY_COMMIT = "d015e9bfe441bd90ae431bac24c55cb74bdbce26"  # main em 2025-07-30
FLYBODY_ASSETS = "flybody/fruitfly/assets"
USER_AGENT = "mosca-patins-download"

# MaleCNS v1.0 (Berg et al., Cell 2026; licença CC-BY 4.0): arquivos Feather abertos, sem login.
MALECNS_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
MALECNS_SMALL = (
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
)
MALECNS_WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
# Tabela por corpo (88 milhões de fragmentos): só as sinapses dos neurônios anotados ficam, em
# SYNAPSE_COUNTS, para estimar o volume dos neurônios (mosca.brain.pugliese.estimate_sizes).
MALECNS_STATS = "body-stats-male-cns-v1.0-minconf-0.5.feather"
SYNAPSE_COUNTS = "body-synapse-counts.feather"

# Rede do MaleCNS que Pugliese et al. usaram (neurônios motores da pata da frente, pré-motores e
# descendentes; sinapses só nas regiões do cordão nervoso), no repositório deles.
PUGLIESE_REPO = "smpuglie/Pugliese_2026"
PUGLIESE_COMMIT = "10e7661bf414ba7b4c2edf795cd36d0f878c17c0"  # main em 2026-09-15
PUGLIESE_DIR_IN_REPO = "data/imac t1 connectome data"
PUGLIESE_FILES = ("W_20260210_vncRoisOnly.csv", "wTable_20260210_vncRoisOnly.csv")

# Figshare da Janelia (DOI 10.25378/janelia.25309105, licença GPL-3.0+).
# O host ndownloader.figshare.com redireciona para o S3, que aceita HTTP Range.
WALKING_ZIP_URL = "https://ndownloader.figshare.com/files/51196868"
WALKING_SAMPLE_MEMBER = "walking-dataset-small_female-only_snippets-100_min-len-0.5s_trk-files-0-9.hdf5"
# Políticas treinadas do flybody (mesmo figshare, GPL-3.0+): só a de caminhada, a professora do M4.
POLICIES_ZIP_URL = "https://ndownloader.figshare.com/files/44815195"
POLICIES_DIR = ASSETS / "flybody_policies"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def download_flybody(dest: Path = FLYBODY_DIR) -> None:
    api = f"https://api.github.com/repos/{FLYBODY_REPO}/contents/{FLYBODY_ASSETS}?ref={FLYBODY_COMMIT}"
    listing = [item for item in json.loads(_get(api)) if item["type"] == "file"]
    dest.mkdir(parents=True, exist_ok=True)

    def fetch(item: dict) -> str:
        path = dest / item["name"]
        if path.exists() and _git_blob_sha(path.read_bytes()) == item["sha"]:
            return "ok"
        raw = f"https://raw.githubusercontent.com/{FLYBODY_REPO}/{FLYBODY_COMMIT}/{FLYBODY_ASSETS}/{item['name']}"
        data = _get(raw)
        if _git_blob_sha(data) != item["sha"]:
            raise RuntimeError(f"hash não confere: {item['name']}")
        path.write_bytes(data)
        return "baixado"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fetch, listing))

    manifest = {
        "repo": FLYBODY_REPO,
        "commit": FLYBODY_COMMIT,
        "path": FLYBODY_ASSETS,
        "license": "Apache-2.0",
        "files": {item["name"]: {"size": item["size"], "git_sha": item["sha"]} for item in listing},
    }
    (dest / "SOURCE.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"flybody: {results.count('baixado')} baixados, {results.count('ok')} já presentes -> {dest}")


def download_walking_sample(dest: Path = WALKING_SAMPLE) -> None:
    from remotezip import RemoteZip  # dependência opcional: pip install -e ".[data]"

    if dest.exists():
        print(f"amostra de caminhada: já presente -> {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with RemoteZip(WALKING_ZIP_URL) as z:
        info = z.getinfo(WALKING_SAMPLE_MEMBER)
        with z.open(info) as src, open(dest.with_suffix(".part"), "wb") as out:
            while chunk := src.read(1 << 20):
                out.write(chunk)
    dest.with_suffix(".part").replace(dest)
    manifest = {
        "source": "https://doi.org/10.25378/janelia.25309105",
        "zip_url": WALKING_ZIP_URL,
        "member": WALKING_SAMPLE_MEMBER,
        "crc32": f"{info.CRC:08x}",
        "size": info.file_size,
        "license": "GPL-3.0+",
    }
    (FLYBODY_DATA_DIR / "SOURCE.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"amostra de caminhada: {info.file_size / 1e6:.1f} MB -> {dest}")


def download_walking_policy(dest: Path = POLICIES_DIR) -> None:
    """Extrai walking/ (SavedModel) do zip de políticas do figshare, lendo só os trechos necessários."""
    from remotezip import RemoteZip

    manifest_path = dest / "SOURCE.json"
    if manifest_path.exists() and (dest / "walking" / "variables" / "variables.index").exists():
        print("política de caminhada já presente")
        return
    files = {}
    with RemoteZip(POLICIES_ZIP_URL) as z:
        for info in z.infolist():
            if info.filename.startswith("walking/") and not info.is_dir():
                out = dest / info.filename
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(info))
                files[info.filename] = {"size": info.file_size, "crc32": f"{info.CRC:08x}"}
    manifest = {"source": "https://doi.org/10.25378/janelia.25309105", "zip_url": POLICIES_ZIP_URL,
                "license": "GPL-3.0+", "files": files}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"política de caminhada: {len(files)} arquivos -> {dest / 'walking'}")


def download_malecns(names: tuple[str, ...], dest: Path = MALECNS_DIR) -> None:
    """Baixa arquivos do MaleCNS em blocos, registrando tamanho e SHA-256 (não há hash oficial)."""
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "SOURCE.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "source": MALECNS_URL, "release": "v1.0", "license": "CC-BY 4.0", "files": {}}
    for name in names:
        path = dest / name
        if path.exists() and name in manifest["files"] and path.stat().st_size == manifest["files"][name]["size"]:
            print(f"MaleCNS: {name} já presente")
            continue
        sha = hashlib.sha256()
        req = urllib.request.Request(MALECNS_URL + name, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp, open(path.with_suffix(".part"), "wb") as out:
            while chunk := resp.read(1 << 22):
                sha.update(chunk)
                out.write(chunk)
        path.with_suffix(".part").replace(path)
        manifest["files"][name] = {"size": path.stat().st_size, "sha256": sha.hexdigest()}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"MaleCNS: {name} ({path.stat().st_size / 1e6:.1f} MB)")


def download_malecns_stats(dest: Path = MALECNS_DIR) -> None:
    """Sinapses de entrada e saída de cada neurônio anotado; a tabela original é apagada depois."""
    import pandas as pd
    import pyarrow.feather as feather

    if (dest / SYNAPSE_COUNTS).exists():
        print(f"MaleCNS: {SYNAPSE_COUNTS} já presente")
        return
    download_malecns(MALECNS_SMALL + (MALECNS_STATS,), dest)
    annotated = pd.read_feather(dest / MALECNS_SMALL[0], columns=["bodyId"]).bodyId
    stats = feather.read_table(dest / MALECNS_STATS, columns=["body", "pre", "post"]).to_pandas()
    stats = stats[stats.body.isin(annotated)].reset_index(drop=True)
    stats.to_feather(dest / SYNAPSE_COUNTS)
    (dest / MALECNS_STATS).unlink()
    print(f"MaleCNS: {SYNAPSE_COUNTS} ({len(stats)} neurônios); {MALECNS_STATS} apagado")


def download_pugliese(dest: Path = PUGLIESE_DIR) -> None:
    """Rede do MaleCNS de Pugliese et al., conferida pelo hash de blob do Git."""
    from urllib.parse import quote

    folder = quote(PUGLIESE_DIR_IN_REPO)
    api = f"https://api.github.com/repos/{PUGLIESE_REPO}/contents/{folder}?ref={PUGLIESE_COMMIT}"
    listing = {item["name"]: item for item in json.loads(_get(api)) if item["name"] in PUGLIESE_FILES}
    dest.mkdir(parents=True, exist_ok=True)
    for name in PUGLIESE_FILES:
        item, path = listing[name], dest / name
        if path.exists() and _git_blob_sha(path.read_bytes()) == item["sha"]:
            print(f"Pugliese: {name} já presente")
            continue
        data = _get(f"https://raw.githubusercontent.com/{PUGLIESE_REPO}/{PUGLIESE_COMMIT}/{folder}/{quote(name)}")
        if _git_blob_sha(data) != item["sha"]:
            raise RuntimeError(f"hash não confere: {name}")
        path.write_bytes(data)
        print(f"Pugliese: {name} ({len(data) / 1e6:.1f} MB)")
    manifest = {"repo": PUGLIESE_REPO, "commit": PUGLIESE_COMMIT, "path": PUGLIESE_DIR_IN_REPO,
                "files": {n: {"size": listing[n]["size"], "git_sha": listing[n]["sha"]} for n in PUGLIESE_FILES}}
    (dest / "SOURCE.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walking-sample", action="store_true", help="baixa os 100 trechos de caminhada (~95 MB)")
    parser.add_argument("--malecns", action="store_true", help="anotações e neurotransmissores do MaleCNS (~55 MB)")
    parser.add_argument("--malecns-weights", action="store_true", help="ligações do MaleCNS (~1,1 GB)")
    parser.add_argument("--malecns-stats", action="store_true",
                        help="sinapses por neurônio do MaleCNS (baixa 778 MB, guarda ~2 MB)")
    parser.add_argument("--pugliese", action="store_true", help="rede do MaleCNS de Pugliese et al. (~75 MB)")
    parser.add_argument("--walking-policy", action="store_true", help="política de caminhada do flybody (~5 MB)")
    args = parser.parse_args()
    if args.pugliese:
        download_pugliese()
    download_flybody()
    if args.walking_sample:
        download_walking_sample()
    if args.malecns or args.malecns_weights:
        download_malecns(MALECNS_SMALL + ((MALECNS_WEIGHTS,) if args.malecns_weights else ()))
    if args.malecns_stats:
        download_malecns_stats()
    if args.walking_policy:
        download_walking_policy()


if __name__ == "__main__":
    main()
