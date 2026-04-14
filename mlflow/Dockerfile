FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==2.13.0 psycopg2-binary boto3

# Fly Postgres gives postgres:// but MLflow requires postgresql://
# Use Python (always present) to rewrite the URL and exec mlflow.
COPY mlflow/start.py /start.py
CMD ["python", "/start.py"]
