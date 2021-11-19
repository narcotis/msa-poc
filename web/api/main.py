import asyncio
import uvicorn
from fastapi import FastAPI
from typing import AsyncGenerator, Dict
from nats.aio.client import Client as NATS
from nats.aio.client import Msg
from starlette.requests import Request
from starlette.responses import StreamingResponse
import nats
import json

app = FastAPI()
nc = NATS()
js = nc.jetstream()

acks = []
msgs = []

@app.on_event("startup")
async def nats_connect():
    global nc, js
    if not nc.is_connected:
        await nc.connect('nats://nats:4222')
    print("web2 connected!")
    await js.add_stream(name="msa-test", subjects=["api.>"])
    #await js.account_info()
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
    if not nc.is_connected:
        await nc.connect('nats://nats:4222')
        print("web2 connected!")
    return "web2 connected!"

@app.get("/get")
async def get():
    global js
    psub = await js.pull_subscribe("api.data", "psub")
    msg = await psub.fetch()
    #msgs.append(msg.data)
    for ms in msg:
        msgs.append(ms.data)
    return msgs

@app.post("/post")
async def post(tmp: dict):
    global js
    data = json.dumps(tmp)
    ack = await js.publish(subject= "api.data", payload=data.encode(), stream="msa-test")
    acks.append(ack)
    return acks
