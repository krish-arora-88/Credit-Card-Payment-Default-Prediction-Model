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
    # --serve-artifacts proxies uploads through the REST API.
    # --artifacts-destination is where the SERVER stores files locally.
    # --default-artifact-root mlflow-artifacts:/ tells clients to go via proxy,
    # not write directly to the server's filesystem.
    "--serve-artifacts",
    "--artifacts-destination", "/mlflow/artifacts",
    "--default-artifact-root", "mlflow-artifacts:/",
    "--workers", "1",
]

sys.exit(subprocess.call(cmd))
