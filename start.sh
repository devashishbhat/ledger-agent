#!/bin/sh
set -e
if [ ! -f data/chunks.jsonl ]; then
  echo "No index found: building it (takes several minutes)..."
  uv run --no-sync python -m ledger.edgar
  uv run --no-sync python -m ledger.index
fi
exec uv run --no-sync uvicorn ledger.api:app --host 0.0.0.0 --port "${PORT:-8000}"
