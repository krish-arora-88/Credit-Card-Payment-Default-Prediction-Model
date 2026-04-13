import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

mock_model = MagicMock()
mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

@pytest.fixture
def client():
    with patch('mlflow.sklearn.load_model', return_value=mock_model), \
         patch('mlflow.tracking.MlflowClient') as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_mv = MagicMock()
        mock_mv.version = '1'
        mock_client.get_model_version_by_alias.return_value = mock_mv
        with patch.dict(os.environ, {
            'MLFLOW_TRACKING_URI': 'http://localhost:5000',
            'MODEL_NAME': 'CreditDefaultModel',
            'MODEL_ALIAS': 'champion',
        }):
            # Import fresh each time
            import importlib
            import api.main as main_module
            importlib.reload(main_module)
            from fastapi.testclient import TestClient
            with TestClient(main_module.app) as test_client:
                yield test_client

VALID_PAYLOAD = {
    "LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 10000, "BILL_AMT2": 9000, "BILL_AMT3": 8000,
    "BILL_AMT4": 7000, "BILL_AMT5": 6000, "BILL_AMT6": 5000,
    "PAY_AMT1": 1000, "PAY_AMT2": 900, "PAY_AMT3": 800,
    "PAY_AMT4": 700, "PAY_AMT5": 600, "PAY_AMT6": 500,
}

def test_health_endpoint(client):
    resp = client.get('/health')
    assert resp.status_code == 200
    assert 'status' in resp.json()

def test_predict_returns_valid_response(client):
    resp = client.post('/predict', json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert 'prediction' in body
    assert 'probability' in body
    assert body['prediction'] in [0, 1]
    assert 0.0 <= body['probability'] <= 1.0

def test_predict_missing_field_returns_422(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != 'LIMIT_BAL'}
    resp = client.post('/predict', json=payload)
    assert resp.status_code == 422
