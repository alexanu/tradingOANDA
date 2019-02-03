# coding: utf-8

# import sqlite3
# import os
# import shutil
# from async_v20 import DateTime
# import numpy as np
import pandas as pd
# import datetime
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
# import multiprocessing as mp


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

def copy_table_from_sqlite_to_mysql(tn):
    global engine_sqlite, engine_mysql

    engine_mysql.dispose()
    engine_sqlite.dispose()

    n_rows = count_rows (engine=engine_sqlite,tn=tn)
    print(tn, n_rows)

    strSQL = f"SELECT * FROM {tn} "
    df = pd.read_sql(strSQL,con=engine_sqlite,parse_dates=['time'])

    # colNames = df.columns
    #
    # r = re.compile("ask|bid|mid_?")
    # newlist = list(filter(r.match, colNames))
    # for colName in newlist:
    #     # print(colName)
    #     df[colName] = df[colName].astype('float32')
    df = df[df.columns.difference(['ID'])]
    df['complete'] = df['complete'].astype('bool')



    # print(df.info())


    df.to_sql(tn,con=engine_mysql,if_exists='replace',index=False, chunksize=10000)


    strSQL = f"create unique index {tn}_i_time on {tn}(time)"
    print(strSQL)
    engine_mysql.execute(strSQL)

    strSQL = f"ALTER TABLE `{tn}` ADD COLUMN ID BIGINT PRIMARY KEY AUTO_INCREMENT FIRST"
    print(strSQL)
    engine_mysql.execute(strSQL)

    strSQL = f"ALTER TABLE `{tn}` PAGE_COMPRESSED=1"
    print(strSQL)
    engine_mysql.execute(strSQL)

    strSQL = f"OPTIMIZE TABLE {tn}"
    print(strSQL)
    engine_mysql.execute(strSQL)


def main(argv):
    print('in main')
    strDB = 'trading_OANDA_S5'

    global engine_mysql, engine_sqlite

    engine_mysql = sqlalchemy.create_engine(f"mysql+pymysql://bn@127.0.0.1/{strDB}", echo=False)
    engine_sqlite = sqlalchemy.create_engine(f"sqlite:///" + f"/home/bn/tradingData/{strDB}.sqlite", echo=False)


    instrument_names = ['BCO_USD', 'CN50_USD', 'DE10YB_EUR', 'DE30_EUR',
                        'EU50_EUR', 'EUR_CHF', 'EUR_GBP', 'EUR_JPY',
                        'EUR_USD', 'FR40_EUR', 'HK33_HKD', 'JP225_USD',
                        'NAS100_USD', 'SPX500_USD', 'UK100_GBP', 'UK10YB_GBP',
                        'US2000_USD', 'US30_USD', 'USB02Y_USD', 'USB05Y_USD',
                        'USB10Y_USD', 'USB30Y_USD', 'USD_CNH', 'XAU_EUR']

    instrument_names_here = sorted(instrument_names)
    instrument_names_here = instrument_names_here[0:]
    print(instrument_names_here)


    t1 = timer()
    for tn in instrument_names_here:
        print(tn)
        copy_table_from_sqlite_to_mysql(tn)
        pass
    t2 = timer()
    print("copy table(1): {0}".format(t2-t1))

    # this consumes too much memory for the smaller intervals like 1 Minute or 5 seconds; then we have to use the serial method above
    # procs = []
    # for index, tn in enumerate(tables):
    #     proc = mp.Process(target=copy_table_from_sqlite_to_mysql, args=(engine_sqlite, engine_mysql, tn))
    #     procs.append(proc)
    #     proc.start()
    # for proc in procs:
    #     proc.join()




    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))


