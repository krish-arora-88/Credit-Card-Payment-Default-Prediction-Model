"""
FastAPI inference server.
Loads the MLflow-registered champion model at startup and serves /predict.
"""
import os
import time
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram, REGISTRY
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel


def _get_or_create_counter(name, documentation, labelnames):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Counter(name, documentation, labelnames)


def _get_or_create_histogram(name, documentation, buckets):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Histogram(name, documentation, buckets=buckets)


def _get_or_create_gauge(name, documentation):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Gauge(name, documentation)


PREDICTION_COUNTER = _get_or_create_counter(
    'modelops_predictions_total',
    'Total prediction requests',
    ['prediction'],
)
PREDICTION_LATENCY = _get_or_create_histogram(
    'modelops_prediction_latency_seconds',
    'Time to compute a prediction',
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
MODEL_VERSION_GAUGE = _get_or_create_gauge(
    'modelops_model_version',
    'Current deployed model version number',
)
PREDICTION_PROBABILITY = _get_or_create_histogram(
    'modelops_prediction_probability',
    'Distribution of predicted default probabilities',
    buckets=[0.1 * i for i in range(11)],
)

TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MODEL_NAME   = os.getenv('MODEL_NAME', 'CreditDefaultModel')
MODEL_ALIAS  = os.getenv('MODEL_ALIAS', 'champion')

_pipeline = None
_model_version = 'unknown'


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _model_version
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    _pipeline = mlflow.sklearn.load_model(model_uri)
    try:
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        _model_version = mv.version
        MODEL_VERSION_GAUGE.set(float(_model_version))
    except Exception:
        pass
    yield


app = FastAPI(title='Credit Card Default Prediction API', lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


class PredictRequest(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str


@app.get('/health')
def health():
    return {'status': 'ok', 'model_version': _model_version}


@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    from src.preprocess import engineer_features, FEATURE_COLS

    row = pd.DataFrame([request.model_dump()])
    row = engineer_features(row)
    X = row[FEATURE_COLS]

    start = time.perf_counter()
    prob = float(_pipeline.predict_proba(X)[0][1])
    elapsed = time.perf_counter() - start

    prediction = int(prob > 0.5)

    PREDICTION_COUNTER.labels(prediction=str(prediction)).inc()
    PREDICTION_LATENCY.observe(elapsed)
    PREDICTION_PROBABILITY.observe(prob)

    return PredictResponse(
        prediction=prediction,
        probability=round(prob, 4),
        model_version=_model_version,
    )
