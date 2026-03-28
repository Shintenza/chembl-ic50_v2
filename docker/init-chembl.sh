#!/usr/bin/env bash
set -euo pipefail

DUMP_DIR="/dumps"
DB_NAME="${CHEMBL_DB_NAME:-${POSTGRES_DB}}"
DB_USER="${POSTGRES_USER}"

log() { echo "[init-chembl] $*"; }

dump_file=$(find "$DUMP_DIR" -maxdepth 1 -name "*chembl*.dmp" | head -n1)

if [[ -z "$dump_file" ]]; then
    log "ERROR: No dump file matching *chembl*.dmp found in $DUMP_DIR"
    log "       Download the ChEMBL PostgreSQL dump and place it there."
    exit 1
fi

log "Found dump: $dump_file"
log "Restoring into database: $DB_NAME (this may take 30-60 minutes)..."

pg_restore \
    --username="$DB_USER" \
    --dbname="$DB_NAME" \
    --no-owner \
    --no-acl \
    --jobs="$(nproc)" \
    "$dump_file"

log "Running ANALYZE to update query planner statistics..."
psql --username="$DB_USER" --dbname="$DB_NAME" -c "ANALYZE;"

log "Done. ChEMBL database is ready."
