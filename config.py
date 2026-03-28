"""
Central configuration for the ChEMBL GNN IC50 prediction pipeline.

All constants and paths are defined here. No other module should
hardcode paths, directory names, or tuning hyperparameters.

Database credentials are read from environment variables.  For local
development, copy .env.example → .env and run:

    docker compose up -d
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env (if present) before reading environment variables.
# python-dotenv is optional — the file is just a convenience for local dev.
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

_load_dotenv()

# ---------------------------------------------------------------------------
# Database connection
# Defaults match docker-compose.yml so `docker compose up` works out of the box.
# ---------------------------------------------------------------------------

def _require_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"Required environment variable '{key}' is not set. Copy .env.example → .env and fill it in.")
    return value

DB: dict = {
    "host":     _require_env("CHEMBL_DB_HOST"),
    "dbname":   _require_env("CHEMBL_DB_NAME"),
    "user":     _require_env("CHEMBL_DB_USER"),
    "password": _require_env("CHEMBL_DB_PASSWORD"),
    "port":     int(_require_env("CHEMBL_DB_PORT")),
}

# ---------------------------------------------------------------------------
# Directory paths  (all derived from ROOT_DIR)
# ---------------------------------------------------------------------------

ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT_DIR / "data"

PATHS: dict = {
    "ROOT_DIR": ROOT_DIR,
    "DATA_DIR": DATA_DIR,
    "RAW_DIR": DATA_DIR / "raw",
    "CLEANED_DIR": DATA_DIR / "cleaned",
    "GRAPHS_DIR": DATA_DIR / "graphs",
    "SPLITS_DIR": DATA_DIR / "splits",
    "MODELS_DIR": DATA_DIR / "models",
    "LOGS_DIR": DATA_DIR / "logs",
}

# Create directories on import so downstream code never has to mkdir manually.
for _path in PATHS.values():
    if isinstance(_path, Path) and _path != PATHS["ROOT_DIR"]:
        _path.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Extraction settings
# ---------------------------------------------------------------------------

EXTRACTION: dict = {
    "BATCH_SIZE": 300_000,
    "STANDARD_TYPE": "IC50",
    "STANDARD_UNITS": "nM",
    "STANDARD_RELATION": "=",
    "ASSAY_TYPES": ("B", "F"),
    "CONFIDENCE_SCORE": 9,
    "TARGET_TYPE": "SINGLE PROTEIN",
    "VALID_VALIDITY_COMMENTS": (None, "Manually validated"),
}

# ---------------------------------------------------------------------------
# Cleaning thresholds
# ---------------------------------------------------------------------------

CLEANING: dict = {
    "MIN_ATOMS": 5,
    "MAX_ATOMS": 100,
    "MIN_MW": 100.0,
    "MAX_MW": 1000.0,
    "MIN_PCHEMBL": 2.0,
    "MAX_PCHEMBL": 12.0,
}

# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

GRAPH: dict = {
    "CHUNK_SIZE": 10_000,          # Data objects per .pt chunk file
    "NUM_ATOM_FEATURES": 38,
    "NUM_BOND_FEATURES": 6,
}

# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

SPLIT: dict = {
    "FRAC_TRAIN": 0.8,
    "FRAC_VAL": 0.1,
    "FRAC_TEST": 0.1,
    "SEED": 42,
}

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

TRAINING: dict = {
    "BATCH_SIZE": 512,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 1e-5,
    "HIDDEN_DIM": 64,
    "HEAD_DIM": 32,
    "NUM_GCN_LAYERS": 3,
    "MAX_EPOCHS": 200,
    "PATIENCE": 15,
    "EVAL_BATCH_SIZE": 1024,
    "NUM_WORKERS": 4,
}
