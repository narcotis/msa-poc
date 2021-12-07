import asyncio
import nats
from nats.error import TimeoutError

async def main():
    nc = await nats.connect("nats://nats:4222")

    # jetstream
    js = nc.jetstream()

    # add subjects and stream
    # stream 이름과 subject 일치가 중요
    await js.add_stream(name='msa-test', subjects=["Optimizer.data.>", "Optimizer.train.>"])

    # 특정 subject에 대한 구독권 (pull_subscription) 선언
    # "psub" 은 durable (Counsumer Name). 이름을 설정해서 그 다음에 끊어졌다가 연결되도 이어서 sub
    # stream은 이력 안할경우, subject 기반으로 stream 자동 탐지
    # Config도 설정 가능
    psub = await js.pull_subscribe("Optimizer.train.>", "psub", stream="msa-test", config=None)







if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
        loop.run_forever()
        loop.close()
    except:
        pass
