FROM fastapi:python3.9-slim

RUN pip install nats-py

WORKDIR /app