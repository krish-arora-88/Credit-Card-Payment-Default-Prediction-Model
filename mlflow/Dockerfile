FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==2.13.0 psycopg2-binary boto3

# Fly Postgres provides postgres:// but MLflow requires postgresql://
COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
