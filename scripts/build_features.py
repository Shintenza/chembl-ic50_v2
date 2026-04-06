"""
Usage
-----
    python scripts/04_build_features.py                         # build both
    python scripts/04_build_features.py --features graphs
    python scripts/04_build_features.py --features fingerprints
    python scripts/04_build_features.py --features all
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.enums import FeatureType
from src.graph import build_graphs
from src.features.fingerprints import build_fingerprints

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build molecular feature chunks for GNN and/or MLP training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--features",
        type=FeatureType,
        default=FeatureType.ALL,
        choices=list(FeatureType),
        help="Which features to build.",
    )
    return parser.parse_args()


def _already_built(out_dir: Path, label: str) -> bool:
    existing = sorted(out_dir.glob("chunk_*.pt"))
    if existing:
        logger.warning(
            "%s: %d chunk file(s) already exist in %s — skipping. "
            "Delete the directory manually to force a rebuild.",
            label,
            len(existing),
            out_dir,
        )
        return True
    return False


def build_graphs_step() -> None:
    out_dir = config.PATHS["GRAPHS_DIR"]
    if _already_built(out_dir, "graphs"):
        return

    logger.info("--- Building molecular graphs ---")
    logger.info("Cleaned dir : %s", config.PATHS["CLEANED_DIR"])
    logger.info("Output dir  : %s", out_dir)
    logger.info("Chunk size  : %d", config.GRAPH["CHUNK_SIZE"])

    total = build_graphs(
        cleaned_dir=config.PATHS["CLEANED_DIR"],
        output_dir=out_dir,
        chunk_size=config.GRAPH["CHUNK_SIZE"],
    )
    logger.info("Graphs done — %d total → %s", total, out_dir)


def build_fingerprints_step() -> None:
    out_dir = config.PATHS["FINGERPRINTS_DIR"]
    if _already_built(out_dir, "fingerprints"):
        return

    logger.info("--- Building Morgan fingerprints ---")
    logger.info("Cleaned dir : %s", config.PATHS["CLEANED_DIR"])
    logger.info("Output dir  : %s", out_dir)
    logger.info("Radius      : %d", config.FINGERPRINT["RADIUS"])
    logger.info("Bits        : %d", config.FINGERPRINT["N_BITS"])
    logger.info("Chunk size  : %d", config.FINGERPRINT["CHUNK_SIZE"])

    total = build_fingerprints(
        cleaned_dir=config.PATHS["CLEANED_DIR"],
        output_dir=out_dir,
        radius=config.FINGERPRINT["RADIUS"],
        n_bits=config.FINGERPRINT["N_BITS"],
        chunk_size=config.FINGERPRINT["CHUNK_SIZE"],
    )
    logger.info("Fingerprints done — %d total → %s", total, out_dir)


def main() -> None:
    args = parse_args()

    logger.info("=== Build Features [%s] ===", args.features)

    if args.features in (FeatureType.GRAPHS, FeatureType.ALL):
        build_graphs_step()

    if args.features in (FeatureType.FINGERPRINTS, FeatureType.ALL):
        build_fingerprints_step()

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
