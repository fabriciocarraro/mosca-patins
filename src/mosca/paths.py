"""Caminhos do projeto. `MOSCA_ASSETS` sobrescreve a pasta de assets (útil no Spark)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(os.environ.get("MOSCA_ASSETS", ROOT / "assets"))
FLYBODY_DIR = ASSETS / "flybody"
FLYBODY_XML = FLYBODY_DIR / "fruitfly.xml"
FLYBODY_DATA_DIR = ASSETS / "flybody_data"
WALKING_SAMPLE = FLYBODY_DATA_DIR / "walking-sample-100.hdf5"
MALECNS_DIR = ASSETS / "malecns"
RUNS = ROOT / "runs"
