"""
Champion promotion logic.
Compares new model F1 against current champion; promotes if strictly better.
"""
import os

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME     = os.getenv('MODEL_NAME', 'CreditDefaultModel')
CHAMPION_ALIAS = 'champion'


def should_promote(new_f1: float, champion_f1: float) -> bool:
    return new_f1 > champion_f1


def get_champion_f1(client: MlflowClient) -> float | None:
    """Return champion's F1 score, or None if no champion exists."""
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        run = client.get_run(champion.run_id)
        return float(run.data.metrics['f1_score'])
    except Exception:
        return None


def promote_if_better(new_version: str, new_f1: float) -> bool:
    """
    Promote new_version to champion alias if it beats current champion.
    Returns True if promoted.
    """
    client = MlflowClient()
    champion_f1 = get_champion_f1(client)

    if champion_f1 is None:
        print(f"No existing champion — promoting version {new_version} (F1={new_f1:.4f})")
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, new_version)
        return True

    if should_promote(new_f1, champion_f1):
        print(f"New champion: v{new_version} (F1={new_f1:.4f}) > v_old (F1={champion_f1:.4f})")
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, new_version)
        return True

    print(f"No promotion: new F1={new_f1:.4f} <= champion F1={champion_f1:.4f}")
    return False


if __name__ == '__main__':
    import sys
    new_version = sys.argv[1]
    new_f1 = float(sys.argv[2])
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(tracking_uri)
    promote_if_better(new_version, new_f1)
