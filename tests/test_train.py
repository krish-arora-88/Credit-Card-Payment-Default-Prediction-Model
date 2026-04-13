import os
import mlflow
import pytest
from src.train import train_and_log

@pytest.fixture(autouse=True)
def use_tmp_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    mlflow.set_experiment("test_experiment")
    yield

def test_train_and_log_returns_metrics():
    data_path = "data/UCI_Credit_Card.csv"
    if not os.path.exists(data_path):
        pytest.skip("data file not present")
    result = train_and_log(data_path=data_path, experiment_name="test_experiment", register=False)
    assert 'f1_score' in result
    assert 'run_id' in result
    assert 'model_version' in result
    assert 0.0 < result['f1_score'] < 1.0

def test_train_logs_params_and_metrics(tmp_path):
    data_path = "data/UCI_Credit_Card.csv"
    if not os.path.exists(data_path):
        pytest.skip("data file not present")
    result = train_and_log(data_path=data_path, experiment_name="test_experiment", register=False)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(result['run_id'])
    assert 'learning_rate' in run.data.params
    assert 'depth' in run.data.params
    assert 'f1_score' in run.data.metrics
    assert 'roc_auc' in run.data.metrics
