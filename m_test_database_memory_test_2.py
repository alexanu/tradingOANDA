# coding: utf-8

import sqlite3
import numpy as np
import pandas as pd
import datetime
import sys
import pprint
import pymysql
import sqlalchemy
from sqlalchemy import func
import sqlalchemy.orm
import sqlalchemy.ext.automap
import sqlalchemy.util._collections
from timeit import default_timer as timer
from sqlalchemy.dialects.mysql import insert as sqlalchemy_insert
import typing
import multiprocessing as mp
import gc
import psutil
import oursql

from sqlalchemy import event
from sqlalchemy.event import listen

def receive_before_cursor_execute(**kw):
    """listen for the 'before_cursor_execute' event"""
    print("event before_cursor_execute")
    conn = kw['conn']
    cursor = kw['cursor']

def receive_after_cursor_execute(**kw):
    """listen for the 'after_cursor_execute' event"""
    print("event after_cursor_execute")
    conn = kw['conn']
    cursor = kw['cursor']
    # pprint.pprint(kw)

def receive_close(**kw):
    """listen for the 'close' event"""
    print("event close")
    pprint.pprint(kw)

def receive_close_detached(**kw):
    """listen for the 'close_detached' event"""
    print("event close detached")
    pprint.pprint(kw)

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



def read_df_oursql(tn:str, i, n_days):
    global conn

    # print(tn)
    time_from = datetime.datetime(year=2012, month=1, day=1)
    time_to = time_from + pd.Timedelta(days=n_days)

    curs = conn.cursor(oursql.DictCursor)
    curs = conn.cursor(try_plain_query=False)
    curs.execute(
        f"SELECT * FROM `{tn}` WHERE time > '{time_from}' and time <= '{time_to}'")
    a = curs.fetchall()
    df = pd.DataFrame(a,columns=[t[0] for t in curs.description])
    curs.close()

    proc = psutil.Process()
    mem = (proc.memory_info().rss / 1024/1024)

    print(f"memory {proc.pid}: {tn}; {i}; {mem}; {len(df)}")

    pass

def read_df(tn: str,i,n_days):
    global meta, inspector, engine, Base, session_maker

    # print(tn)
    tt = meta.tables.get(tn)
    engine.dispose()

    ssn = session_maker()
    time_from = datetime.datetime(year=2012, month=1, day=1)
    time_to = time_from + pd.Timedelta(days=n_days)
    ssn.commit()
    ssn.close()

    where_part = sqlalchemy.and_(tt.c.time >= time_from, tt.c.time < time_to)
    order_part = tt.c.time

    q = ssn.query(tt).filter(where_part).order_by(where_part)
    # print(q)
    # print(literalquery(q))


    df = pd.read_sql(sql=q.statement, con=engine, parse_dates=["time"])
    conn = engine.raw_connection()
    curs = conn.cursor()

    curs.close()
    conn.close()


    # df = [0,1]


    proc = psutil.Process()
    mem = (proc.memory_info().rss / 1024/1024)

    print(f"memory {proc.pid}: {tn}; {i}; {mem}; {len(df)}")


def read_df_chain(tn,n_chain,n_days):
    for i in range(n_chain):
        read_df(tn,i,n_days)


def main(argv):
    print('in main')
    strDB = 'trading_OANDA_M5'
    time_interval = pd.Timedelta(minutes=5)
    all_times_table_name = 'all_times'

    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 300)

    global engine_mysql
    global engine_sqlite
    global engine
    global meta
    global Base
    global inspector
    global session_maker
    global conn

    engine_mysql = sqlalchemy.create_engine(f"mysql+pymysql://bn@127.0.0.1/{strDB}", echo=False)
    engine_sqlite = sqlalchemy.create_engine(f"sqlite:///" + f"/home/bn/tradingData/{strDB}.sqlite", echo=False)


    conn = oursql.connect(host='localhost', user='bn', passwd='',
                          db=strDB, port=3306)

    engine = engine_mysql

    inspector = sqlalchemy.inspect(engine)
    meta = sqlalchemy.MetaData()
    meta.reflect(bind=engine)

    Base = sqlalchemy.ext.automap.automap_base(metadata=meta)
    Base.prepare()

    print(Base.classes.keys())

    # listen(engine, 'before_cursor_execute', receive_before_cursor_execute, named=True)
    # listen(engine, 'after_cursor_execute', receive_after_cursor_execute, named=True)
    # listen(engine, 'close', receive_close(), named=True)
    # listen(engine, 'close_detached', receive_close_detached(), named=True)

    session_maker = sqlalchemy.orm.sessionmaker(bind=engine, autocommit=False)

    instrument_names = ['BCO_USD', 'CN50_USD', 'DE10YB_EUR', 'DE30_EUR',
                        'EU50_EUR', 'EUR_CHF', 'EUR_GBP', 'EUR_JPY',
                        'EUR_USD', 'FR40_EUR', 'HK33_HKD', 'JP225_USD',
                        'NAS100_USD', 'SPX500_USD', 'UK100_GBP', 'UK10YB_GBP',
                        'US2000_USD', 'US30_USD', 'USB02Y_USD', 'USB05Y_USD',
                        'USB10Y_USD', 'USB30Y_USD', 'USD_CNH', 'XAU_EUR']

    instrument_names_here = instrument_names[0:]
    n_days = 400
    n_chain = 5

    # for tn in instrument_names_here:
    #     read_df_chain(tn,n_chain, n_days)

    t1 = timer()
    procs = []
    for index, instrument_name in enumerate(instrument_names_here):
        proc = mp.Process(target=read_df_chain, args=(instrument_name, n_chain, n_days))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
        pass
    t2 = timer()
    print(f"parallel: {t2-t1}")




if __name__ == '__main__':
    sys.exit(main(sys.argv))
