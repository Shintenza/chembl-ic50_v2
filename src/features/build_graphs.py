from pathlib import Path

from .graphs import smiles_to_graph
from .build_features import build_features


def build_graphs(
    cleaned_dir: Path,
    output_dir: Path,
    chunk_size: int,
) -> int:
    return build_features(
        cleaned_dir=cleaned_dir,
        output_dir=output_dir,
        chunk_size=chunk_size,
        builder_fn=smiles_to_graph,
    )
