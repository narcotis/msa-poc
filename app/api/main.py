import asyncio
from fastapi import FastAPI
from typing import AsyncGenerator, Dict
from nats.aio.client import Client as NATS
from nats.aio.client import Msg
import nats
import json

app = FastAPI()
nc = NATS()
js = nc.jetstream()
psub = await js.pull_subscribe("api.data", "psub")

acks = []
msgs = []


@app.on_event("startup")
async def nats_connect():
    global nc, js
    if not nc.is_connected:
        await nc.connect('nats://nats:4222')
    print("web1 connected!")
    await js.add_stream(name="msa-test", subjects=["api.>"])
    await js.account_info()
    #psub = await js.pull_subscribe("api", "psub")

@app.on_event("shutdown")
async def nats_close():
    global nc, js
    await nc.close()

async def get_nats() -> NATS:
    global nc
    if not nc.is_connected:
        await nc.connect('nats://nats:4222')
    return nc

@app.get("/")
async def root():
    global nc
    if not nc.is_connected:
        await nc.connect('nats://nats:4222')
        print("web1 connected!")
    return "web1 connected!"


@app.get("/get")
async def get():
    global js
    psub = await js.pull_subscribe("api.data", "psub")
    msg = await psub.fetch()
    for ms in msg:
        msgs.append(ms.data)
    return msgs


@app.post("/post")
async def post(tmp: dict):
    global nc
    js = nc.jetstream()
    data = json.dumps(tmp)
    ack = await js.publish(subject= "api.data", payload=data.encode(), stream="msa-test")
    acks.append(ack)
    return acks

