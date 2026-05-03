from src.features.fingerprints import prepare_tensor
from functools import partial
import logging
from pathlib import Path
from rdkit.Chem import rdFingerprintGenerator
from .build_features import build_features

logger = logging.getLogger(__name__)



def build_fingerprints(
    cleaned_dir: Path,
    output_dir: Path,
    radius: int,
    n_bits: int,
    chunk_size: int,
) -> int:
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )

    builder_fn = partial(prepare_tensor, mfpgen)

    return build_features(
        cleaned_dir=cleaned_dir,
        output_dir=output_dir,
        chunk_size=chunk_size,
        builder_fn=builder_fn,
    )
