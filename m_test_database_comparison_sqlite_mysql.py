# coding: utf-8

# import sqlite3
# import os
# import shutil
# from async_v20 import DateTime
# import numpy as np
import pandas as pd
import datetime
import sys
# import pprint
# import pathos
# import time
# import pymysql
import sqlalchemy
# from pathos.pools import ProcessPool
# import functools
# import re
from timeit import default_timer as timer


def count_rows(engine, tn):
    conn = engine.connect()
    n_rows = conn.execute("SELECT count(*) FROM {0}".format(tn)).fetchone()[0]
    conn.close()
    return n_rows


def get_tables(engine):
    inspector = sqlalchemy.inspect(engine)
    print(type(inspector))
    a = inspector.get_table_names()
    return a

def compare_table_from_sqlite_to_mysql(engine_sqlite, engine_mysql, tn):

    engine_mysql.dispose()
    engine_sqlite.dispose()

    n_rows = count_rows (engine=engine_sqlite,tn=tn)
    print(tn, n_rows)

    strSQL = f"SELECT time FROM {tn}"
    df = pd.read_sql(strSQL,con=engine_sqlite,parse_dates=['time'])
    # print(df.info())
    df2 = df.sample(5000)
    # print(df2.info())
    df2.time = df2.time.dt.strftime('%Y-%m-%d %H:%M:%S.000000')
    # print(df2.info())

    a = df2.time.values
    b = "', '".join(a)

    t1 = timer()
    strSQL = f"SELECT ask_c,bid_h FROM {tn} WHERE time in ('{b}') ORDER BY time ASC"
    # df = pd.read_sql(strSQL,con=engine_sqlite,parse_dates=['time'])
    df = engine_sqlite.execute(strSQL).fetchall()
    print(len(df))
    t2 = timer()
    dt = (t2-t1)
    dt_sqlite = dt
    print("sqlite read selection: {0}".format(dt))
    print(df[-1])

    t1 = timer()
    strSQL = f"SELECT ask_c,bid_h FROM {tn} WHERE time in ('{b}') ORDER BY time ASC"
    # df = pd.read_sql(strSQL,con=engine_mysql,parse_dates=['time'])
    df = engine_mysql.execute(strSQL).fetchall()
    print(len(df))
    t2 = timer()
    dt = (t2-t1)
    dt_mysql = dt
    print("mysql read selection: {0}".format(dt))
    print(df[-1])

    # return [dt_sqlite, dt_mysql]





def compare_sqlite_mysql(engine_sqlite, engine_mysql):
    tables = get_tables(engine_sqlite)
    tables = ['BCO_USD','CN50_USD','DE10YB_EUR','EUR_JPY']

    for tn in [t for t in tables if t[0:7] != "sqlite_"]:
        print(tn)
        compare_table_from_sqlite_to_mysql(engine_sqlite,engine_mysql,tn)
        print()
        pass
    pass




def main(argv):
    print('in main')
    strDB = 'trading_OANDA_S5_sub_1'

    engine_mysql = sqlalchemy.create_engine(f"mysql+pymysql://bn:basket05@127.0.0.1/{strDB}", echo=False)
    engine_sqlite = sqlalchemy.create_engine(f"sqlite:///" + f"/home/bn/tradingData/{strDB}.sqlite", echo=False)



    t1 = datetime.datetime.utcnow()

    compare_sqlite_mysql(engine_sqlite,engine_mysql)

    t2 = datetime.datetime.utcnow()
    print("copy table(1): {0}".format(t2-t1))

    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))


