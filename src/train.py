"""
Training script: trains CatBoost on credit-card default data,
logs the run to MLflow, and registers the model.
"""
import os
import tempfile
from datetime import datetime
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import FEATURE_COLS, engineer_features, build_preprocessor

TARGET = 'default.payment.next.month'
MODEL_NAME = os.getenv('MODEL_NAME', 'CreditDefaultModel')

DEFAULT_PARAMS = {
    'depth': 4,
    'iterations': 100,
    'learning_rate': 0.075,
    'l2_leaf_reg': 4,
    'random_seed': 42,
    'auto_class_weights': 'Balanced',
    'verbose': 0,
}


def _load_data(data_path: str):
    df = pd.read_csv(data_path)
    df = engineer_features(df)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, val_df


def train_and_log(
    data_path: str = 'data/UCI_Credit_Card.csv',
    experiment_name: str = 'CreditCardDefault',
    params: dict | None = None,
    register: bool = True,
) -> dict[str, Any]:
    """Train model, log to MLflow, optionally register. Returns dict with f1_score + run_id + model_version."""
    if params is None:
        params = DEFAULT_PARAMS

    mlflow.set_experiment(experiment_name)
    run_name = f"catboost_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    train_df, val_df = _load_data(data_path)
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET]
    X_val   = val_df[FEATURE_COLS]
    y_val   = val_df[TARGET]

    preprocessor = build_preprocessor()
    model = CatBoostClassifier(**params)
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

    with mlflow.start_run(run_name=run_name) as run:
        pipeline.fit(X_train, y_train)

        y_pred  = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]

        metrics = {
            'f1_score':  float(f1_score(y_val, y_pred)),
            'precision': float(precision_score(y_val, y_pred)),
            'recall':    float(recall_score(y_val, y_pred)),
            'roc_auc':   float(roc_auc_score(y_val, y_proba)),
        }

        mlflow.log_params({k: v for k, v in params.items() if k != 'verbose'})
        mlflow.log_metrics(metrics)

        # SHAP summary artifact
        X_val_transformed = preprocessor.transform(X_val)
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_transformed)
        with tempfile.TemporaryDirectory() as tmp:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_values, X_val_transformed, show=False)
            path = f"{tmp}/shap_summary.png"
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(path)

        model_info = mlflow.sklearn.log_model(pipeline, "model")
        run_id = run.info.run_id

        model_version = None
        if register:
            mv = mlflow.register_model(model_info.model_uri, MODEL_NAME)
            model_version = mv.version

    return {
        'f1_score':      metrics['f1_score'],
        'roc_auc':       metrics['roc_auc'],
        'run_id':        run_id,
        'model_version': model_version,
        **metrics,
    }


if __name__ == '__main__':
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(tracking_uri)
    result = train_and_log()
    print(f"Training complete: F1={result['f1_score']:.4f}, version={result['model_version']}")
