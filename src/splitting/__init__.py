"""Splitting sub-package: scaffold-based and random dataset splitting."""

from .scaffold_split import generate_scaffold, scaffold_split, build_split_map
from .random_split import build_random_split_map

__all__ = [
    "generate_scaffold",
    "scaffold_split",
    "build_split_map",
    "build_random_split_map",
]
