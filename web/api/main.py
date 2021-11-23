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

class FastNATS(FastAPI):
    def __init__(self, nats_url: Optional[str] = None,
                 stream_name: Optional[str] = None,
                 subjects: Optional[List[str]] = None,
                 pull_subject: Optional[str] = None,
                 **extra: Any):
        super().__init__(**extra)
        self.nats = NATS()
        self.js = self.nats.jetstream()
        self.nats_url = nats_url
        self.stream_name = stream_name
        self.subjects = subjects
        self.pull_subject = pull_subject
        self.psub = None
        self.msgs = []
        self.acks = []
        asyncio.run(self.pull())

    async def pull(self):
        if self.nats_url is None:
            return

        while True:
            if not self.nats.is_connected:
                await self.nats.connect(self.nats_url)

            await self.js.add_stream(name=self.stream_name, subjects=self.subjects)
            self.psub = await self.js.pull_subscribe(subject=self.pull_subject, durable="psub")
            try:
                msg = await self.psub.fetch()
                for ms in msg:
                    self.msgs.append(ms.data)
            except:
                await asyncio.sleep(0.01)

app = FastNATS(nats_url="nats://nats:4222", stream_name="msa-test", subjects=["api.>"], pull_subject="api.data")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# nc = app.nats
# js = nc.jetstream()
#
# psub = None
# acks = []
# msgs = []
# try:
#     msg = await psub.fetch()
#     for ms in msg:
#         msgs.append(ms.data)
# except:
#     pass

@app.on_event("startup")
async def nats_connect():
    if app.nats_url is None:
        return

    if not app.nats.is_connected:
        await app.nats.connect(app.nats_url)

    await app.js.add_stream(name=app.stream_name, subjects=app.subjects)

    # if not nc.is_connected:
    #     await nc.connect('nats://nats:4222')
    # print("web1 connected!")
    # await js.add_stream(name="msa-test", subjects=["api.>"])
    # psub = await js.pull_subscribe("api.data", "psub")


@app.on_event("shutdown")
async def nats_close():
    app.loop.close()
    await app.nats.close()


@app.get("/get")
async def get():
    return app.msgs


@app.post("/post")
async def post(tmp: dict):
    data = json.dumps(tmp)
    ack = await app.js.publish(subject="api.data", payload=data.encode(), stream="msa-test")
    app.acks.append(ack)
    return app.acks
