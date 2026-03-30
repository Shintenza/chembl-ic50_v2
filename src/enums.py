"""Shared enum types used across the pipeline."""

from enum import Enum


class _StrEnum(str, Enum):
    """Base class that makes str(member) return the value, not 'Class.MEMBER'.

    This ensures clean f-string interpolation and logging output without
    needing to call .value explicitly everywhere.
    """

    def __str__(self) -> str:
        return self.value


class Split(_StrEnum):
    """Dataset partition label."""
    TRAIN = "train"
    VAL   = "val"
    TEST  = "test"


class SplitStrategy(_StrEnum):
    """Splitting algorithm used to assign molecules to partitions."""
    SCAFFOLD = "scaffold"
    RANDOM   = "random"


class ModelType(_StrEnum):
    """Model architecture."""
    GCN = "gcn"
    MLP = "mlp"


class LossFunction(_StrEnum):
    """Training loss function."""
    MSE = "mse"
    MAE = "mae"


class FeatureType(_StrEnum):
    """Molecular feature representation to build."""
    GRAPHS        = "graphs"
    FINGERPRINTS  = "fingerprints"
    ALL           = "all"
