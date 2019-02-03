# coding: utf-8

# import sqlite3
import os
# import shutil
# from async_v20 import DateTime
import numpy as np
import pandas as pd
import datetime
import sys
# import pprint
# import pathos
import time
# import pymysql
import sqlalchemy
# from pathos.pools import ProcessPool
import functools
# import re
from timeit import default_timer as timer
import multiprocessing as mp
# import random
import sqlalchemy.ext.automap
from sqlalchemy import func
# from numba import jit


class StringLiteral(sqlalchemy.sql.sqltypes.String):
    """Teach SA how to literalize various things."""
    def literal_processor(self, dialect):
        super_processor = super(StringLiteral, self).literal_processor(dialect)

        def process(value):
            if isinstance(value, int):
                return str(value)
            if not isinstance(value, str):
                value = str(value)
            result = super_processor(value)
            if isinstance(result, bytes):
                result = result.decode(dialect.encoding)
            return result
        return process
        pass
    pass


class LiteralDialect(sqlalchemy.engine.default.DefaultDialect):
    colspecs = {
        # prevent various encoding explosions
        sqlalchemy.sql.sqltypes.String: StringLiteral,
        # teach SA about how to literalize a datetime
        sqlalchemy.sql.sqltypes.DateTime: StringLiteral,
        # don't format py2 long integers to NULL
        sqlalchemy.sql.sqltypes.NullType: StringLiteral,
    }

def literalquery(statement):
    """NOTE: This is entirely insecure. DO NOT execute the resulting strings."""
    if isinstance(statement, sqlalchemy.orm.Query):
        statement = statement.statement
    return statement.compile(
        dialect=LiteralDialect(),
        compile_kwargs={'literal_binds': True},
    ).string

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def count_rows(tn):
    global meta, inspector, engine, Base, session_maker
    engine.dispose()

    tt = meta.tables.get(tn)
    ssn = session_maker()
    if 'time' in [ttt.name for ttt in tt.c]:
        n_rows = ssn.query(func.count(tt.c.time)).scalar()
    else:
        n_rows = ssn.query(tt).count()
    ssn.close()
    return n_rows



def get_tables(engine):

    inspector = sqlalchemy.inspect(engine)
    # print(type(inspector))
    a = inspector.get_table_names()

    return a


def doubler(x,engine):
    global N_doubler
    result = x * 2
    proc = os.getpid()
    for i in range(pow(2,N_doubler)):
        a = np.exp(2.3)

    tables = get_tables(engine)

    # print(tables)

    # print('{0} doubled to {1} by process id: {2}'.format(
    #     x, result, proc))
    return result

def doubler_global_engine(x):
    global engine
    global N_doubler

    engine.dispose()

    result = x * 2
    proc = os.getpid()
    for i in range(pow(2,N_doubler)):
        a = np.exp(2.3)

    tables = get_tables(engine)

    # print(tables)

    # print('{0} doubled to {1} by process id: {2}'.format(
    #     x, result, proc))
    return result


def get_time_sample(tn, N_replicates):
    global engine
    global N_samples
    global meta

    strSQL = f"SELECT time FROM {tn}"


    df = pd.read_sql(strSQL,con=engine,parse_dates=['time'])
    # print(df.info())
    df2s = []
    for repl in np.arange(start=1,stop=N_replicates+1):
        df2 = df.sample(N_samples)
        df2["Replicate"] = repl
        df2s.append(df2)
        pass
    df2 = functools.reduce(lambda x, y: x.append(y), df2s)

    # print(df2.info())
    df2["time_as_string"] = df2.time.dt.strftime('%Y-%m-%d %H:%M:%S.000000')
    # print(df2.info())
    df2["Instrument_Name"] = tn
    return(df2)



def get_sample(tn,times, engine_passed = None):
    global engine, meta, session_maker, df_training

    if engine_passed is None:
        engine_here = engine
    else:
        engine_here = engine
        pass

    engine_here.dispose()

    tt = meta.tables[tn]


    ssn = session_maker()
    # n_rows = count_rows(tn)
    # print(tn,n_rows)

    q = ssn.query(tt).filter(tt.c.time.in_(times.dt.to_pydatetime())).statement
    # q = ssn.query(tt).filter(tt.c.time.in_(times)).statement
    # print(type(q))
    # print(literalquery(q))

    # t1 = timer()
    ddf = pd.read_sql(q,con=engine_here,parse_dates=['time'])
    # print(ddf.info())

    # t2 = timer()
    # print(f"{tn}:{t2-t1}")

    # print(ddf.info())
    # if tn == "EUR_USD" :
    #     print(ddf.tail())
    #     print(len(ddf[ddf.volume>0]))

    ssn.close()

    result = ddf

    # print(f"{tn}: {len(result)}")

    # return result

def df_empty(columns, dtypes, index=None):
    assert len(columns)==len(dtypes)
    df = pd.DataFrame(index=index)
    for c,d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df


def main(argv):
    print('in main')
    strDB = 'M52'
    time_interval = pd.Timedelta(minutes=5)
    strDB = 'trading_OANDA_M5'
    time_interval = pd.Timedelta(minutes=5)
    strDB = 'trading_OANDA_M1'
    # time_interval = pd.Timedelta(minutes=1)
    # # # strDB = 'trading_OANDA_S5_sub_1'
    # strDB = 'trading_OANDA_S5'
    # time_interval = pd.Timedelta(seconds=5)


    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 300)

    engine_mysql = sqlalchemy.create_engine(f"mysql+mysqldb://bn@127.0.0.1/{strDB}", echo=False)
    engine_sqlite = sqlalchemy.create_engine(f"sqlite:///" + f"/home/bn/tradingData/{strDB}.sqlite", echo=False)

    global engine, meta, session_maker, N_doubler, N_samples, df_training, Base
    engine = engine_mysql

    inspector = sqlalchemy.inspect(engine)

    meta = sqlalchemy.MetaData()
    meta.reflect(bind=engine)

    Base = sqlalchemy.ext.automap.automap_base(metadata=meta)
    Base.prepare()

    session_maker = sqlalchemy.orm.sessionmaker(bind=engine)
    session = session_maker()
    assert(isinstance(session,sqlalchemy.orm.Session))
    session.close()

    N_doubler = 20

    N_samples = 500

    N_passes = 50

    df_training = df_empty(['time','mid_c'], dtypes=['datetime64[ns]','float32'])
    print(df_training.info())


    instrument_names = ['BCO_USD', 'CN50_USD', 'DE10YB_EUR', 'DE30_EUR',
                        'EU50_EUR', 'EUR_CHF', 'EUR_GBP', 'EUR_JPY',
                        'EUR_USD','FR40_EUR', 'HK33_HKD', 'JP225_USD',
                        'NAS100_USD', 'SPX500_USD', 'UK100_GBP', 'UK10YB_GBP',
                        'US2000_USD','US30_USD', 'USB02Y_USD', 'USB05Y_USD',
                        'USB10Y_USD', 'USB30Y_USD', 'USD_CNH', 'XAU_EUR']

    instrument_names_here = instrument_names[0:]

    tables = get_tables(engine)


    t1 = timer()

    min_time = datetime.datetime(2002, 5, 6, 20, 55)
    max_time = datetime.datetime(2019, 1, 4, 21, 55)

    times_all = pd.date_range(start=min_time,end=max_time,freq=time_interval)
    times0 = np.random.choice(times_all, N_samples)
    times1 = np.random.choice(times_all, N_samples)
    times2 = np.random.choice(times_all, N_samples)

    df_times0 = pd.DataFrame(data=times0,columns=['time'])
    times0 = df_times0['time']
    df_times1 = pd.DataFrame(data=times1,columns=['time'])
    times1 = df_times1['time']
    df_times2 = pd.DataFrame(data=times2, columns=['time'])
    times2 = df_times2['time']

    # times2 = times1 + time_interval * 100000

    # t1 = timer()
    # ssn = session_maker()
    # tt = meta.tables.get('all_times')
    # # n_rows = count_rows('all_times')
    # # print('all_times',n_rows)
    # q = ssn.query(tt).filter(tt.c.time.in_(times0.dt.to_pydatetime()))
    # # print(type(q))
    # # print(literalquery(q))
    # ddf = pd.read_sql(q.statement,con=engine,parse_dates=['time'])
    # # print(ddf.head())
    # ssn.close()
    # t2 = timer()
    # print(f'mapped times (2_0): {t2-t1}')

    instrument_names_here = [
        'BCO_USD',
        'CN50_USD', 'DE10YB_EUR', 'DE30_EUR',
        'EU50_EUR', 'EUR_CHF', 'EUR_GBP', 'EUR_JPY',
        'EUR_USD', 'FR40_EUR',
        'HK33_HKD', 'JP225_USD',
        'NAS100_USD', 'SPX500_USD', 'UK100_GBP', 'UK10YB_GBP',
        'US2000_USD', 'US30_USD', 'USB02Y_USD', 'USB05Y_USD',
        'USB10Y_USD', 'USB30Y_USD', 'USD_CNH', 'XAU_EUR'
    ]

    instrument_names_here = [t for t in instrument_names_here if t in Base.classes.keys()]

    times_loop_start = np.random.choice(times_all, N_samples)

    dts_loop = []
    for i in range(N_passes):

        times_loop = np.random.choice(times_all, N_samples)
        # times_loop = times_loop_start + i * time_interval

        df_times_loop = pd.DataFrame(data=times_loop, columns=['time'])
        times_loop = df_times_loop['time']


        t1 = timer()
        procs = []
        for index, instrument_name in enumerate(instrument_names_here):
            proc = mp.Process(target=get_sample, args=(instrument_name, times_loop))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        t2 = timer()
        dts_loop.append(t2-t1)
        print(f'parallel ({i}): start now: {datetime.datetime.now()}; start: {times_loop[0]}; dt: {t2-t1}',end="\r",flush=True)
        time.sleep(0.5)

        pass
    print()

    print(f"parallel: mean: {np.mean(dts_loop)}; std: {np.std(dts_loop)}")

    t1 = timer()
    for instrument_name in instrument_names_here:
        # print(instrument_name)
        result = get_sample(instrument_name, times1)
        # print(len(result))
    t2 = timer()
    print(f'serial (1_1): {t2-t1}')


    t1 = timer()
    for instrument_name in instrument_names_here:
        # print(instrument_name)
        result = get_sample(instrument_name, times1)
        # print(len(result))
    t2 = timer()
    print(f'serial (1_1): {t2-t1}')







    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))


