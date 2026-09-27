"""Baixa os assets de terceiros para assets/, com versões fixadas e checagem de integridade.

Uso:
    python scripts/download_assets.py                    # modelo da mosca (flybody)
    python scripts/download_assets.py --walking-sample   # + 100 trechos de moscas reais andando

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
from mosca.paths import FLYBODY_DATA_DIR, FLYBODY_DIR, WALKING_SAMPLE  # noqa: E402

FLYBODY_REPO = "TuragaLab/flybody"
FLYBODY_COMMIT = "d015e9bfe441bd90ae431bac24c55cb74bdbce26"  # main em 2025-07-30
FLYBODY_ASSETS = "flybody/fruitfly/assets"
USER_AGENT = "mosca-patins-download"

# Figshare da Janelia (DOI 10.25378/janelia.25309105, licença GPL-3.0+).
# O host ndownloader.figshare.com redireciona para o S3, que aceita HTTP Range.
WALKING_ZIP_URL = "https://ndownloader.figshare.com/files/51196868"
WALKING_SAMPLE_MEMBER = "walking-dataset-small_female-only_snippets-100_min-len-0.5s_trk-files-0-9.hdf5"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walking-sample", action="store_true", help="baixa os 100 trechos de caminhada (~95 MB)")
    args = parser.parse_args()
    download_flybody()
    if args.walking_sample:
        download_walking_sample()


if __name__ == "__main__":
    main()
