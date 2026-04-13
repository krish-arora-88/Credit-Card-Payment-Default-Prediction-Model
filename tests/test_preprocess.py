import pandas as pd
import numpy as np
import pytest
from src.preprocess import engineer_features, build_preprocessor, FEATURE_COLS

def _minimal_row():
    return {
        'ID': 1, 'LIMIT_BAL': 50000, 'SEX': 1, 'EDUCATION': 2,
        'MARRIAGE': 1, 'AGE': 30,
        'PAY_0': 0, 'PAY_2': 0, 'PAY_3': 0, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0,
        'BILL_AMT1': 10000, 'BILL_AMT2': 9000, 'BILL_AMT3': 8000,
        'BILL_AMT4': 7000, 'BILL_AMT5': 6000, 'BILL_AMT6': 5000,
        'PAY_AMT1': 1000, 'PAY_AMT2': 900, 'PAY_AMT3': 800,
        'PAY_AMT4': 700, 'PAY_AMT5': 600, 'PAY_AMT6': 500,
        'default.payment.next.month': 0,
    }

def test_engineer_features_creates_derived_cols():
    df = pd.DataFrame([_minimal_row()])
    out = engineer_features(df)
    assert 'has_delay' in out.columns
    assert 'max_delay' in out.columns
    assert 'utilization' in out.columns
    assert 'payment_ratio' in out.columns

def test_engineer_features_has_delay_false_when_no_delays():
    df = pd.DataFrame([_minimal_row()])  # all PAY_* = 0
    out = engineer_features(df)
    assert out['has_delay'].iloc[0] == 0

def test_engineer_features_education_nan_for_invalid():
    row = _minimal_row()
    row['EDUCATION'] = 5  # invalid, should become NaN
    df = pd.DataFrame([row])
    out = engineer_features(df)
    assert pd.isna(out['EDUCATION'].iloc[0])

def test_build_preprocessor_transforms_shape():
    rows = [_minimal_row() for _ in range(5)]
    df = pd.DataFrame(rows)
    df = engineer_features(df)
    X = df[FEATURE_COLS]
    preprocessor = build_preprocessor()
    Xt = preprocessor.fit_transform(X)
    assert Xt.shape[0] == 5
    assert Xt.shape[1] > 0
