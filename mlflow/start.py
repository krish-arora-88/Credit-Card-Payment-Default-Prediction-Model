"""
MLflow server entrypoint.
Fly Postgres provides postgres:// URIs; MLflow requires postgresql://.
"""
import os
import subprocess
import sys

db_url = os.environ["DATABASE_URL"].replace("postgres://", "postgresql://", 1)

cmd = [
    "mlflow", "server",
    "--host", "0.0.0.0",
    "--port", "5000",
    "--backend-store-uri", db_url,
    "--default-artifact-root", "/mlflow/artifacts",
    "--serve-artifacts",
    "--workers", "1",   # lean: 1 worker keeps RAM under 1GB on Fly free tier
]

sys.exit(subprocess.call(cmd))
