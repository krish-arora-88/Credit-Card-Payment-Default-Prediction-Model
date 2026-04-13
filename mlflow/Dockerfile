FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==2.13.0 psycopg2-binary boto3
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--serve-artifacts"]
