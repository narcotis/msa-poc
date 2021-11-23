import asyncio
from typing import Any
from uvicorn import Config, Server
from fastapi import FastAPI
from typing import AsyncGenerator, Dict
from nats.aio.client import Client as NATS
from nats.aio.client import Msg
import nats
import json

class FastNATS(FastAPI):
    def __init__(self, **extra: Any):
        super().__init__(**extra)
        self.nats = NATS()


app = FastNATS()
nc = app.nats
js = nc.jetstream()

psub = None
acks = []
msgs = []

try:
    msg = await psub.fetch()
    for ms in msg:
        msgs.append(ms.data)
except:
    pass


@app.on_event("startup")
async def nats_connect():
    global nc, js, psub
    if not nc.is_connected:
        await nc.connect('nats://nats:4222')
    print("web1 connected!")
    await js.add_stream(name="msa-test", subjects=["api.>"])
    psub = await js.pull_subscribe("api.data", "psub")


@app.on_event("shutdown")
async def nats_close():
    global nc
    await nc.close()


@app.get("/get")
async def get():
    return msgs


@app.post("/post")
async def post(tmp: dict):
    global js
    data = json.dumps(tmp)
    ack = await js.publish(subject= "api.data", payload=data.encode(), stream="msa-test")
    acks.append(ack)
    return acks
