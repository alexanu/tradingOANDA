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



def read_df_pymysql(tn: str,i):
    strDB = 'trading_OANDA_M5'
    conn = pymysql.connect(host='localhost', port=3306, user='bn', passwd='', db=strDB)
    cur = conn.cursor()

    t_start = datetime.datetime(year=2000,month=1,day=1)
    t_end = t_start + pd.Timedelta(days=4000) * (i+1)

    strSQL = f"SELECT * FROM {tn} WHERE time >= '{t_start}' and time < '{t_end}'"
    print(strSQL)
    a = cur.execute(strSQL)
    print(a)


    proc = psutil.Process()
    print(proc.memory_info().rss / 1.e6)

    cur.close()
    conn.close()

    df = None
    return df


def read_df(tn: str,i):
    global engine

    engine.dispose()

    t_start = datetime.datetime(year=2000,month=1,day=1)
    t_end = t_start + pd.Timedelta(days=4000) * (i+1)
    t_end = t_start + pd.Timedelta(days=4000)

    strSQL = f"SELECT * FROM {tn} WHERE time >= '{t_start}' and time < '{t_end}'"
    print(strSQL)
    a = engine.execute(strSQL)
    df = pd.DataFrame(iter(a))
    print(len(df))

    proc = psutil.Process()
    print(proc.memory_info().rss / 1.e6)

    return df



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
    # global meta
    # global Base
    # global inspector
    # global session_maker

    engine_mysql = sqlalchemy.create_engine(f"mysql+mysqlconnector://bn@127.0.0.1/{strDB}", echo=False)
    engine_sqlite = sqlalchemy.create_engine(f"sqlite:///" + f"/home/bn/tradingData/{strDB}.sqlite", echo=False)

    engine = engine_mysql

    inspector = sqlalchemy.inspect(engine)
    meta = sqlalchemy.MetaData()
    meta.reflect(bind=engine)

    Base = sqlalchemy.ext.automap.automap_base(metadata=meta)
    Base.prepare()

    print(Base.classes.keys())

    session_maker = sqlalchemy.orm.sessionmaker(bind=engine)

    instrument_names = ['BCO_USD', 'CN50_USD', 'DE10YB_EUR', 'DE30_EUR',
                        'EU50_EUR', 'EUR_CHF', 'EUR_GBP', 'EUR_JPY',
                        'EUR_USD', 'FR40_EUR', 'HK33_HKD', 'JP225_USD',
                        'NAS100_USD', 'SPX500_USD', 'UK100_GBP', 'UK10YB_GBP',
                        'US2000_USD', 'US30_USD', 'USB02Y_USD', 'USB05Y_USD',
                        'USB10Y_USD', 'USB30Y_USD', 'USD_CNH', 'XAU_EUR']

    instrument_names_here = instrument_names[0:10]
    for tn in instrument_names_here:
        print(tn)
        for i in range(10):
            read_df(tn,i)
            # read_df_pymysql(tn,i)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
