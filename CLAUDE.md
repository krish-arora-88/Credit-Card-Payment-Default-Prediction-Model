# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_preprocess.py -v

# Run a single test
python -m pytest tests/test_api.py::test_health_endpoint -v

# Train locally (requires MLflow running)
MLFLOW_TRACKING_URI=http://localhost:5000 MODEL_NAME=CreditDefaultModel python -m src.train

# Promote a model version as champion
MLFLOW_TRACKING_URI=http://localhost:5000 python promote.py <version> <f1_score>

# Start full local stack (Postgres + MLflow + API + Prometheus + Grafana)
cp .env.example .env
docker compose up -d

# Deploy MLflow server to Fly.io
flyctl deploy --config fly-mlflow.toml

# Deploy FastAPI to Fly.io
flyctl deploy --config fly.toml
```

## Architecture

This is a production MLOps system wrapping a CatBoost credit card default classifier.

### Data flow

Raw CSV (`data/UCI_Credit_Card.csv`, 30k rows) → `src/preprocess.py:engineer_features()` → sklearn `ColumnTransformer` → `CatBoostClassifier` → MLflow Model Registry → FastAPI.

The preprocessing has two stages that must stay in sync:
1. **Pre-pipeline feature engineering** (`engineer_features()`): derives 7 columns (`has_delay`, `max_delay`, `avg_pay`, `avg_bill`, `avg_pay_amt`, `utilization`, `payment_ratio`) and remaps `EDUCATION` values before the sklearn transformer sees the data.
2. **sklearn ColumnTransformer** (`build_preprocessor()`): StandardScaler for numerics, median imputer for `EDUCATION`, most-frequent + OHE for `MARRIAGE` and `has_delay`. `ID` and `SEX` are dropped before the transformer (not passed in `FEATURE_COLS`).

Both `src/train.py` and `api/main.py` call the same `engineer_features()` + `FEATURE_COLS` from `src/preprocess.py`. If you change preprocessing, both paths update automatically.

### MLflow integration

- **Tracking URI**: `https://credit-mlflow-server.fly.dev` (production) or `http://localhost:5000` (local)
- **Backend store**: Fly Postgres (`mlflow-pg`)
- **Artifact store**: Fly volume at `/mlflow/artifacts`, proxied via `mlflow-artifacts:/` scheme — clients upload via REST, not direct filesystem access
- **Model registry name**: `CreditDefaultModel`, alias `champion` always points to the best model
- The pipeline is logged as a sklearn model (`mlflow.sklearn.log_model`), not a catboost model, so it includes the preprocessor

### Champion promotion

`promote.py` implements a strict F1-gate: a new version replaces `champion` only if `new_f1 > champion_f1` (ties do not promote). The GitHub Actions workflow (`retrain.yml`) runs this every Sunday at 02:00 UTC and is also manually dispatchable.

### FastAPI serving

The app loads the champion model at startup via the `lifespan` context manager. The model is held in the module-level `_pipeline` global. Four Prometheus metrics are exposed at `/metrics`:
- `modelops_predictions_total` (counter, labeled by class 0/1)
- `modelops_prediction_latency_seconds` (histogram)
- `modelops_model_version` (gauge)
- `modelops_prediction_probability` (histogram — drift indicator)

The `_get_or_create_*` helpers in `api/main.py` prevent duplicate metric registration when the module is reloaded during tests.

### Fly.io deployment

Two separate apps:
- `credit-mlflow-server` — MLflow server, 1GB RAM, `min_machines_running = 1` (always on, required by API and GitHub Actions)
- `credit-default-api` — FastAPI, 512MB RAM + 256MB swap, `min_machines_running = 0` (scales to zero when idle, ~15–30s cold start)

`fly-mlflow.toml` and `fly.toml` are the respective configs. The MLflow `Dockerfile` uses `mlflow/start.py` as entrypoint (not a shell script) because `python:3.11-slim` uses `dash` as `/bin/sh` which lacks bash parameter expansion. `start.py` rewrites `postgres://` → `postgresql://` since Fly Postgres provides the former but MLflow requires the latter.

### Local Docker Compose

Mirrors production: Postgres → MLflow → API → Prometheus → Grafana. MLflow uses the same Dockerfile but with a `DATABASE_URL` env var. The docker-compose `mlflow` service does **not** use `start.py`; it uses the Dockerfile's CMD which reads `POSTGRES_USER/PASSWORD/DB` env vars — this is a divergence from the Fly deploy path.

### Known Fly.io gotchas

- Fly Postgres connection strings use `postgres://`; MLflow requires `postgresql://` — handled in `mlflow/start.py`
- New MLflow experiments must be created after the server is configured with `--default-artifact-root mlflow-artifacts:/`; experiments created before that config have `/mlflow/artifacts` as artifact root and clients will fail to write artifacts. Fix by updating the `artifact_location` column in Postgres directly.
- `mlflow server --workers 1` is required to stay under 1GB RAM
