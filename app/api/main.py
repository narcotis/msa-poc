import asyncio
from typing import Any, Optional, List
from uvicorn import Config, Server
from fastapi import FastAPI
from typing import AsyncGenerator, Dict
from nats.aio.client import Client as NATS
from nats.aio.client import Msg
import nats
import json
from . import models
from .database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Nats Client
nc = NATS()
js = nc.jetstream()
psub = None
acks = []
msgs = []

@app.on_event("startup")
async def nats_connect():
    global nc, js
    if not nc.is_connected:
        await nc.connect("nats://nats:4222")

    await js.add_stream(name="msa-test", subjects=["api.>"])


@app.on_event("shutdown")
async def nats_close():
    global nc
    await nc.close()


@app.get("/get")
async def get():
    global nc
    return {"status": "ok"}


@app.post("/post")
async def post(tmp: dict):
    global nc, js, acks
    data = json.dumps(tmp)
    ack = await js.publish(subject="api.data", payload=data.encode(), stream="msa-test")
    acks.append(ack)
    return acks
