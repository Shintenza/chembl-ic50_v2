#!/usr/bin/env python3
"""
03_split.py — Generate a train/val/test dataset split.

Usage
-----
    python scripts/03_split.py
    python scripts/03_split.py --split random
    python scripts/03_split.py --split scaffold --frac-train 0.7 --frac-val 0.15 \\
                               --frac-test 0.15 --seed 123 --output-name my_split

The split map is saved as a Parquet file in ``config.PATHS['SPLITS_DIR']``.
Running this script multiple times with different arguments creates multiple
named split files; training scripts select which one to use via ``--split-map``.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.enums import Split, SplitStrategy
from src.splitting import build_split_map, build_random_split_map

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a train/val/test split map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        type=SplitStrategy,
        default=SplitStrategy.SCAFFOLD,
        choices=list(SplitStrategy),
        help="Splitting strategy to use.",
    )
    parser.add_argument(
        "--frac-train",
        type=float,
        default=config.SPLIT["FRAC_TRAIN"],
        help="Fraction of molecules in the training set.",
    )
    parser.add_argument(
        "--frac-val",
        type=float,
        default=config.SPLIT["FRAC_VAL"],
        help="Fraction of molecules in the validation set.",
    )
    parser.add_argument(
        "--frac-test",
        type=float,
        default=config.SPLIT["FRAC_TEST"],
        help="Fraction of molecules in the test set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SPLIT["SEED"],
        help="Random seed.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Base name (without extension) for the output split-map file. "
            "Defaults to '<split>_split'. "
            "Saved as SPLITS_DIR/<output-name>.parquet"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate fractions
    total = args.frac_train + args.frac_val + args.frac_test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Fractions must sum to 1.0 (got {args.frac_train} + "
            f"{args.frac_val} + {args.frac_test} = {total:.6f})"
        )

    output_name = args.output_name or f"{args.split}_split"
    cleaned_dir: Path = config.PATHS["CLEANED_DIR"]
    splits_dir: Path = config.PATHS["SPLITS_DIR"]
    output_path: Path = splits_dir / f"{output_name}.parquet"

    logger.info("=== Dataset Split ===")
    logger.info("Strategy        : %s", args.split)
    logger.info("Input directory : %s", cleaned_dir)
    logger.info("Output file     : %s", output_path)
    logger.info("Fractions       : train=%.3f  val=%.3f  test=%.3f",
                args.frac_train, args.frac_val, args.frac_test)
    logger.info("Seed            : %d", args.seed)

    if args.split == SplitStrategy.SCAFFOLD:
        split_df = build_split_map(
            cleaned_dir=cleaned_dir,
            frac_train=args.frac_train,
            frac_val=args.frac_val,
            frac_test=args.frac_test,
            seed=args.seed,
        )
    else:
        split_df = build_random_split_map(
            cleaned_dir=cleaned_dir,
            frac_train=args.frac_train,
            frac_val=args.frac_val,
            frac_test=args.frac_test,
            seed=args.seed,
        )

    splits_dir.mkdir(parents=True, exist_ok=True)
    split_df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("Split map saved → %s (%d rows)", output_path, len(split_df))

    # Summary
    counts = split_df["split"].value_counts()
    total_n = len(split_df)
    print()
    print("=" * 50)
    print(f"Split Summary  [{args.split}]")
    print("=" * 50)
    for split_name in Split:
        n = counts.get(split_name, 0)
        pct = 100.0 * n / max(total_n, 1)
        print(f"  {split_name:<8}: {n:>8,}  ({pct:.1f}%)")
    print(f"  {'TOTAL':<8}: {total_n:>8,}")
    print("=" * 50)
    print(f"Saved → {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
