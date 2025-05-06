#!/bin/bash
set -e

echo "[entrypoint] Container started at $(date)"
echo "[entrypoint] Checking .env..."
cat .env || echo "No .env file found"

echo "[entrypoint] Starting uvicorn..."
pipenv run uvicorn rag:app --host 0.0.0.0 --port 9000 || {
  echo "[entrypoint] ERROR: Failed to start uvicorn"
  exit 1
}
