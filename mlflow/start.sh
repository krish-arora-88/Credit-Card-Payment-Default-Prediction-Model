#!/bin/bash
# Fly Postgres gives postgres:// but MLflow requires postgresql://
DB_URI=$(echo "$DATABASE_URL" | sed 's|^postgres://|postgresql://|')

exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "$DB_URI" \
    --default-artifact-root /mlflow/artifacts \
    --serve-artifacts
