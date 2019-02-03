from apscheduler.schedulers.blocking import BlockingScheduler
import asyncio
import sys

from async_v20 import OandaClient
from timeit import default_timer as timer

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler

def openClient():
    client = OandaClient(rest_timeout=200,max_simultaneous_connections=100)
    return client

def closeClient(client):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.close())


def get_candles_01(client, instrument_name: str = "EUR_USD",
                price: str = "MBA",
                granularity: str = "H1",
                count: int = 5000,
                from_time: pd.Timestamp = pd.to_datetime("2015-01-01")):
    a = client.get_candles(instrument_name,
                           price=price,
                           granularity=granularity,
                           count=count,
                           from_time=from_time)

    return a


def get_candles(client, instrument_name: str="EUR_USD",
                price: str="MBA",
                granularity: str="H1",
                count: int=5000,
                from_time: pd.Timestamp=pd.to_datetime("2015-01-01")):
    loop = asyncio.get_event_loop()
    a = client.get_candles(instrument_name,
                           price=price,
                           granularity=granularity,
                           count=count,
                           from_time=from_time)
    b = loop.run_until_complete(a)

    return b




async def myJobStream(client,instruments):
    async for price in await client.stream_pricing(instruments):
        print(price)

async def myJob(client,delay,tn,price,granularity,count,from_time):
    await asyncio.sleep(delay)
    nnow = pd.datetime.now()
    t1 = timer()
    aa = await get_candles_01(client,instrument_name=tn,price=price,granularity=granularity,count=count,from_time=from_time)
    bb = aa.candles.dataframe()
    t2 = timer()
    print(f"{tn}; start: {nnow}; length: {len(bb)}; dt = {t2-t1}")



def main(argv):
    t1 = timer()
    client = openClient()
    t2 = timer()
    print(f"opening Client : dt={t2 - t1}")

    loop = asyncio.get_event_loop()
    t3 = timer()
    print(f"getting loop: dt={t3 - t2}")

    response = loop.run_until_complete(client.initialize())
    t3 = timer()
    print(f"initialisation: dt={t3 - t2}")

    instrument_name = "EUR_USD"
    price = "MBA"
    granularity = "S5"
    count = 5000
    from_time = pd.datetime.utcnow() - pd.Timedelta(seconds=60)
    from_time = pd.to_datetime("2010-02-01 19:59")
    delay = 0.1

    t1 = timer()
    b = get_candles(client,
                    instrument_name=instrument_name,
                    price=price,
                    granularity=granularity,
                    count=count,
                    from_time=from_time)
    t2 = timer()
    print(f"getting candles: dt={t2 - t1}")
    df = b.candles.dataframe()
    print(len(df))

    t1 = timer()
    scheduler = AsyncIOScheduler()
    t2 = timer()
    print(f"creating scheduler: dt={t2-t1}")


    instrument_names = [
        'BCO_USD', 'CN50_USD', 'DE10YB_EUR', 'DE30_EUR',
        'EU50_EUR', 'EUR_CHF', 'EUR_GBP', 'EUR_JPY',
        'EUR_USD', 'FR40_EUR', 'HK33_HKD', 'JP225_USD',
        'NAS100_USD', 'SPX500_USD', 'UK100_GBP', 'UK10YB_GBP',
        'US2000_USD', 'US30_USD', 'USB02Y_USD', 'USB05Y_USD',
        'USB10Y_USD', 'USB30Y_USD', 'USD_CNH', 'XAU_EUR',
        ]

    t1 = timer()
    for tn in instrument_names:
        schedulerJob = scheduler.add_job(
            myJob,
                name=f"{tn}_name",
                id=f"{tn}_id",
                trigger='cron',
                hour='*',
                minute='*',
                second='*/20',
                max_instances=100,
                args=[client,delay,tn,price,granularity,count,from_time],
            )

    t2 = timer()
    print(f"addding job: dt={t2-t1}")

    scheduler.start()


    loop = asyncio.get_event_loop()
    # loop.run_until_complete(myJobStream(client, ['EUR_USD']))

    loop.run_forever()

    t1 = timer()
    closeClient(client)
    t2 = timer()
    print(f"closing Client: dt={t2 - t1}")

if __name__ == "__main__":
    sys.exit(main(sys.argv))
