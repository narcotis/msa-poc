import asyncio
import uvicorn
from fastapi import FastAPI
from typing import AsyncGenerator, Dict
from nats.aio.client import Client as NATS
from nats.aio.client import Msg
from starlette.requests import Request
from starlette.responses import StreamingResponse
import nats

app = FastAPI()
nc = NATS()
js = nc.jetstream()

@app.on_event("startup")
async def nats_connect():
    global nc, js
    if not nc.is_connected:
        await nc.connect('nats://nats:4222')
    await js.add_stream(name="msa-test", subjects=["api"])
    psub = await js.pull_subscribe("api", "psub")



@app.get("/")
async def root():
    return {"message": "Hello World"}
