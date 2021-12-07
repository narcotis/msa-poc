FROM fastapi:python3.9-slim

RUN pip install nats-py
RUN pip install sqlalchemy psycopg2-binary sqlalchemy-utils

WORKDIR /app