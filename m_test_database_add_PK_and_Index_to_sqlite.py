# coding: utf-8

# import sqlite3
import os
# import shutil
# from async_v20 import DateTime
# import pandas as pd
# import numpy as np
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




def main(argv):
    print('in main')
    strDB_sqlite = 'testDB13.sqlite'
    strDB_mysql, _ = os.path.splitext(strDB_sqlite)
    engine_sqlite = sqlalchemy.create_engine('sqlite:///' + "testDB13.sqlite", echo=False)


    t1 = datetime.datetime.utcnow()

    tables = get_tables(engine_sqlite)

    for tn in [tn2 for tn2 in tables if tn2.endswith("_tmp")]:
        strSQL = f"DROP TABLE IF EXISTS {tn}"
        print(strSQL)
        engine_sqlite.execute(strSQL)

    tables = get_tables(engine_sqlite)

    for tn in tables:
        n_rows = count_rows(engine_sqlite, tn)
        print(tn,n_rows)

        tmp_table = f"{tn}_tmp"

        strSQL = f"""CREATE TABLE {tmp_table}\
                (
                    ID INTEGER not null constraint {tn}_pk primary key autoincrement,
                    ask_c REAL, 
                    ask_h REAL,
                    ask_l REAL,
                    ask_o REAL,
                    bid_c REAL,
                    bid_h REAL,
                    bid_l REAL,
                    bid_o REAL,
                    complete INT,
                    mid_c REAL,
                    mid_h REAL,
                    mid_l REAL,
                    mid_o REAL,
                    time DATETIME not null,
                    volume INT
                )
                """
        print(strSQL)
        engine_sqlite.execute(strSQL)

        strSQL = f"""
                INSERT INTO {tmp_table} (ask_c, ask_h, ask_l, ask_o, bid_c, bid_h, bid_l, bid_o, complete, mid_c, mid_h, mid_l, mid_o, time, volume)
                SELECT 
                ask_c, ask_h, ask_l, ask_o, bid_c, bid_h, bid_l, bid_o, complete, mid_c, mid_h, mid_l, mid_o, time, volume
                FROM {tn}
                """
        print(strSQL)
        engine_sqlite.execute(strSQL)

        strSQL = f"DROP TABLE {tn}"
        print(strSQL)
        engine_sqlite.execute(strSQL)

        strSQL = f"ALTER TABLE {tmp_table} RENAME TO {tn}"
        print(strSQL)
        engine_sqlite.execute(strSQL)

        strSQL = f"CREATE INDEX {tn}_i_time ON {tn}(time)"
        print(strSQL)
        engine_sqlite.execute(strSQL)


    t2 = datetime.datetime.utcnow()
    print("copy table(1): {0}".format(t2-t1))

    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))


