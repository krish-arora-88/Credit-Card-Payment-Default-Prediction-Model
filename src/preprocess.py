import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PAY_COLS      = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
BILL_COLS     = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
PAY_AMT_COLS  = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

EDUCATION_MAP = {1: 4, 2: 3, 3: 2, 4: 1}

NUM_FEATS = [
    'LIMIT_BAL', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'max_delay', 'avg_pay', 'avg_bill', 'avg_pay_amt', 'utilization', 'payment_ratio',
]
CAT_FEATS     = ['MARRIAGE']
BINARY_FEATS  = ['has_delay']
ORDINAL_FEATS = ['EDUCATION']
DROP_FEATS    = ['ID', 'SEX']

FEATURE_COLS = NUM_FEATS + ORDINAL_FEATS + BINARY_FEATS + CAT_FEATS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering. Modifies a copy; safe to pass train or inference rows."""
    df = df.copy()
    # Clean EDUCATION
    df['EDUCATION'] = df['EDUCATION'].replace([5, 6, 0], np.nan)
    df['EDUCATION'] = df['EDUCATION'].map(EDUCATION_MAP)

    # Derived features
    df['has_delay']     = ((df[PAY_COLS] > 0).sum(axis=1) > 0).astype(int)
    df['max_delay']     = df[PAY_COLS].max(axis=1)
    df['avg_pay']       = df[PAY_COLS].mean(axis=1)
    df['avg_bill']      = df[BILL_COLS].mean(axis=1)
    df['avg_pay_amt']   = df[PAY_AMT_COLS].mean(axis=1)
    df['utilization']   = df['avg_bill'] / df['LIMIT_BAL']
    df['payment_ratio'] = df['avg_pay_amt'] / df['avg_bill'].apply(lambda x: max(x, 1))
    return df


def build_preprocessor():
    """Return a fitted-ready sklearn ColumnTransformer matching the notebook pipeline."""
    numeric_transformer  = StandardScaler()
    ordinal_transformer  = SimpleImputer(strategy='median')
    binary_transformer   = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(drop='if_binary', dtype=int),
    )
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore', sparse_output=False),
    )
    return make_column_transformer(
        (numeric_transformer,     NUM_FEATS),
        (ordinal_transformer,     ORDINAL_FEATS),
        (binary_transformer,      BINARY_FEATS),
        (categorical_transformer, CAT_FEATS),
    )
