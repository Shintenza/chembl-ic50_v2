"""
ChEMBL database extractor.

Iterates over the entire IC50 dataset in batches, writing each batch as a
Parquet file so the process can be interrupted and resumed via
``start_offset``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extensions
from tqdm import tqdm

from .query import build_extraction_query

logger = logging.getLogger(__name__)


def get_connection(db_config: dict) -> psycopg2.extensions.connection:
    """Open and return a psycopg2 connection using *db_config*.

    Parameters
    ----------
    db_config:
        Dict with keys ``host``, ``dbname``, ``user``, ``password``, and
        optionally ``port``.

    Returns
    -------
    psycopg2.extensions.connection
        An open database connection.

    Raises
    ------
    psycopg2.OperationalError
        If the connection cannot be established.
    """
    conn = psycopg2.connect(
        host=db_config["host"],
        dbname=db_config["dbname"],
        user=db_config["user"],
        password=db_config["password"],
        port=db_config.get("port", 5432),
    )
    conn.set_session(readonly=True, autocommit=True)
    return conn


def extract_batches(
    db_config: dict,
    output_dir: Path,
    batch_size: int,
    start_offset: int = 0,
) -> int:
    """Extract all qualifying IC50 records in paginated batches.

    Each batch is written immediately as a Parquet file named
    ``batch_{n:04d}.parquet`` inside *output_dir*, so the process is
    restartable: pass ``start_offset`` to skip already-extracted rows and
    ``start_batch_n`` is inferred from the offset.

    Parameters
    ----------
    db_config:
        Connection parameters forwarded to :func:`get_connection`.
    output_dir:
        Directory in which batch Parquet files are saved.  Created if it
        does not exist.
    batch_size:
        Number of rows per batch.
    start_offset:
        Row offset at which to begin extraction (for resuming).

    Returns
    -------
    int
        Total number of rows extracted in this run.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive the starting batch number from the offset so filenames remain
    # consistent when resuming.
    start_batch_n: int = start_offset // batch_size

    total_rows: int = 0
    offset: int = start_offset
    batch_n: int = start_batch_n

    logger.info(
        "Starting extraction: batch_size=%d, start_offset=%d",
        batch_size,
        start_offset,
    )

    conn: psycopg2.extensions.connection | None = None
    try:
        conn = get_connection(db_config)
        pbar = tqdm(desc="Extracting batches", unit="batch")

        while True:
            query = build_extraction_query(offset=offset, batch_size=batch_size)

            try:
                df = pd.read_sql_query(query, conn)
            except Exception as exc:
                logger.error(
                    "Query failed at offset %d (batch %d): %s",
                    offset,
                    batch_n,
                    exc,
                )
                raise

            if df.empty:
                logger.info("No more rows at offset %d — extraction complete.", offset)
                break

            # Persist immediately to avoid memory build-up.
            out_path = output_dir / f"batch_{batch_n:04d}.parquet"
            df.to_parquet(out_path, index=False, engine="pyarrow")

            rows_in_batch = len(df)
            total_rows += rows_in_batch
            pbar.update(1)
            pbar.set_postfix(
                {
                    "batch": batch_n,
                    "rows_this_batch": rows_in_batch,
                    "total_rows": total_rows,
                }
            )
            logger.debug(
                "Batch %04d saved → %s (%d rows)", batch_n, out_path, rows_in_batch
            )

            # If the result was smaller than the batch size we've hit the end.
            if rows_in_batch < batch_size:
                break

            offset += batch_size
            batch_n += 1

        pbar.close()

    except psycopg2.OperationalError as exc:
        logger.error("Database connection error: %s", exc)
        raise
    finally:
        if conn is not None:
            conn.close()

    logger.info(
        "Extraction finished. Total rows extracted this run: %d", total_rows
    )
    return total_rows
