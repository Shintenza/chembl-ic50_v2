import os
from pathlib import Path


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


def _require_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(
            f"Required environment variable '{key}' is not set. Copy .env.example → .env and fill it in."
        )
    return value


DB: dict = {
    "host": _require_env("CHEMBL_DB_HOST"),
    "dbname": _require_env("CHEMBL_DB_NAME"),
    "user": _require_env("CHEMBL_DB_USER"),
    "password": _require_env("CHEMBL_DB_PASSWORD"),
    "port": int(_require_env("CHEMBL_DB_PORT")),
}

ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT_DIR / "data"

PATHS: dict = {
    "ROOT_DIR": ROOT_DIR,
    "DATA_DIR": DATA_DIR,
    "RAW_DIR": DATA_DIR / "raw",
    "CLEANED_DIR": DATA_DIR / "cleaned",
    "GRAPHS_DIR": DATA_DIR / "graphs",
    "FINGERPRINTS_DIR": DATA_DIR / "fingerprints",
    "SPLITS_DIR": DATA_DIR / "splits",
    "MODELS_DIR": DATA_DIR / "models",
    "LOGS_DIR": DATA_DIR / "logs",
}

for _path in PATHS.values():
    if isinstance(_path, Path) and _path != PATHS["ROOT_DIR"]:
        _path.mkdir(parents=True, exist_ok=True)

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

CLEANING: dict = {
    "MIN_ATOMS": 5,
    "MAX_ATOMS": 100,
    "MIN_MW": 100.0,
    "MAX_MW": 1000.0,
    "MIN_PCHEMBL": 2.0,
    "MAX_PCHEMBL": 12.0,
}

GRAPH: dict = {
    "CHUNK_SIZE": 10_000,
    "NUM_ATOM_FEATURES": 34,
    "NUM_GLOBAL_FEATURES": 11,
    "NUM_BOND_FEATURES": 7,
}

SPLIT: dict = {
    "FRAC_TRAIN": 0.8,
    "FRAC_VAL": 0.1,
    "FRAC_TEST": 0.1,
    "SEED": 30,
}

FINGERPRINT: dict = {
    "RADIUS": 2,
    "N_BITS": 2048,
    "CHUNK_SIZE": 10_000,
}

TRAINING: dict = {
    "BATCH_SIZE": 512,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "HIDDEN_DIM": 128,
    "HEAD_DIM": 64,
    "NUM_GIN_LAYERS": 4,
    "MAX_EPOCHS": 200,
    "PATIENCE": 20,
    "LEARNING_RATE_PATIENCE": 10,
    "LEARNING_RATE_REDUCE_FACTOR": 0.8,
    "EVAL_BATCH_SIZE": 1024,
    "NUM_WORKERS": 4,
}

MLP_TRAINING: dict = {
    "HIDDEN_DIMS": [512, 128],
    "DROPOUT": 0.3,
    "BATCH_SIZE": 512,
    "EVAL_BATCH_SIZE": 1024,
    "LEARNING_RATE": 1e-3,
    "LEARNING_RATE_PATIENCE": 8,
    "LEARNING_RATE_REDUCE_FACTOR": 0.5,
    "WEIGHT_DECAY": 1e-5,
    "MAX_EPOCHS": 200,
    "PATIENCE": 15,
    "NUM_WORKERS": 4,
}
