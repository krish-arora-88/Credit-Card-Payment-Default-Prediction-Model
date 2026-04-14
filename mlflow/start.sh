#!/bin/sh
# Fly Postgres gives postgres:// but MLflow requires postgresql://
DB_URI="${DATABASE_URL/postgres:\/\//postgresql://}"

exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "$DB_URI" \
    --default-artifact-root /mlflow/artifacts \
    --serve-artifacts
