import logging
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extensions

from sqlalchemy import create_engine, Connection
from tqdm import tqdm

from .query import build_extraction_query

logger = logging.getLogger(__name__)

DEFAULT_DB_PORT = 5432


def get_connection(db_config: dict) -> Connection:
    user = db_config["user"]
    password = db_config["password"]
    host = db_config["host"]
    dbname = db_config["dbname"]
    port = db_config.get("port", DEFAULT_DB_PORT)

    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

    return engine.connect()


def extract_batches(
    db_config: dict,
    output_dir: Path,
    batch_size: int,
    start_offset: int = 0,
) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_batch_n = start_offset // batch_size

    total_rows = 0
    offset = start_offset
    batch_n = start_batch_n

    logger.info(
        "Starting extraction: batch_size=%d, start_offset=%d",
        batch_size,
        start_offset,
    )

    conn: Connection | None = None

    try:
        conn = get_connection(db_config)

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

            out_path = output_dir / f"batch_{batch_n:04d}.parquet"
            df.to_parquet(out_path, index=False, engine="pyarrow")

            rows_in_batch = len(df)
            total_rows += rows_in_batch
            logger.debug(
                "Batch %04d saved → %s (%d rows)", batch_n, out_path, rows_in_batch
            )

            if rows_in_batch < batch_size:
                break

            offset += batch_size
            batch_n += 1

    except psycopg2.OperationalError as exc:
        logger.error("Database connection error: %s", exc)
    finally:
        if conn is not None:
            conn.close()

    return total_rows
