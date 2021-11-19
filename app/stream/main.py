import asyncio
from uvicorn import Config, Server
from nats.aio.client import Client as NATS
from nats.aio.client import Msg
import nats

nc = NATS()


async def disconnected_cb():
    print("Got disconnected!")


async def reconnected_cb():
    print("Got reconnected to {url}".format(url=nc.connected_url.netloc))


async def run(loop):
    global nc
    await nc.connect(
        servers=['nats://nats:4222'],
        name='worker1',
        connect_timeout=10,
        ping_interval=10,
        max_outstanding_pings=5,
        max_reconnect_attempts=10,
        reconnect_time_wait=10,
        reconnected_cb=reconnected_cb,
        disconnected_cb=disconnected_cb

    )
    js = nc.jetstream()
    await js.add_stream(name="msa-test", subjects=["api.>"])
