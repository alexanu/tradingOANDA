# coding: utf-8

# import sqlite3
# import os
# import shutil
# from async_v20 import DateTime
# import numpy as np
# import pandas as pd
import datetime
import sys
# import pprint
# import pathos
# import time
import pymysql
# import sqlalchemy
# from pathos.pools import ProcessPool
# import functools
# import re



def get_tables(con, db_src):
    cur = con.cursor()
    strSQL = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA='{db_src}'"
    cur.execute(strSQL)
    tables = [t[0] for t in cur.fetchall()]
    cur.close()
    return tables

def move_table(con, db_src, db_dst, tn):

    cur = con.cursor()

    cur.execute(f"use {db_src}")

    strSQL = f"DROP TABLE IF EXISTS {db_dst}.{tn}"
    print(strSQL)
    cur.execute(strSQL)

    strSQL = f"RENAME TABLE {db_src}.{tn} TO {db_dst}.{tn}"
    print(strSQL)
    cur.execute(strSQL)

    cur.close()

    pass



def move_tables(con, db_src, db_dst):
    tables = get_tables(con, db_src)


    for tn in tables:
        print(tn)
        move_table(con, db_src, db_dst, tn)
        pass
    pass
    pass


def main(argv):
    print('in main')
    con = pymysql.connect('localhost','bn','basket05')
    db_src = "testDB13"
    db_dst = "trading_OANDA_S5"

    t1 = datetime.datetime.utcnow()

    move_tables(con, db_src, db_dst)

    t2 = datetime.datetime.utcnow()
    print("move table(1): {0}".format(t2-t1))

    con.close()

    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))


