from enum import Enum


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Split(_StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class SplitStrategy(_StrEnum):
    SCAFFOLD = "scaffold"
    RANDOM = "random"


class ModelType(_StrEnum):
    GNN = "gnn"
    MLP = "mlp"


class LossFunction(_StrEnum):
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"


class FeatureType(_StrEnum):
    GRAPHS = "graphs"
    FINGERPRINTS = "fingerprints"
    ALL = "all"
