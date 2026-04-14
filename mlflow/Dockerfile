FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==2.13.0 psycopg2-binary boto3
# Shell form so DATABASE_URL env var is expanded at runtime
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "${DATABASE_URL}" \
    --default-artifact-root /mlflow/artifacts \
    --serve-artifacts
