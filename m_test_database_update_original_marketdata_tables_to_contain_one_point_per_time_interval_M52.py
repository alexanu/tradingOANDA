# coding: utf-8

# import sqlite3
# import os
# import shutil
# from async_v20 import DateTime
import numpy as np
import pandas as pd
import datetime
import sys
# import pprint
# import pathos
# import time
# import pymysql
import sqlalchemy
from sqlalchemy import func, select
# from pathos.pools import ProcessPool
# import functools
# import re
import sqlalchemy.orm
import sqlalchemy.ext.automap
import sqlalchemy.util._collections
from timeit import default_timer as timer
# from sqlalchemy.dialects.mysql import insert as sqlalchemy_insert
from trading import upsert as MyUpsert
# import typing
# import multiprocessing as mp
# import psutil
from enum import Enum


class enum_upsert_method(Enum):
    MEMORY = "memory"
    TMPTABLE = "TMPTABLE"


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


def bulk_insert_2(engine, df, tn):
    df.to_sql(tn, con=engine, if_exists='append', index=False, chunksize=10000)
    # strSQL = f"ALTER TABLE {tn} ADD ID SERIAL PRIMARY KEY"
    # engine.execute(strSQL)


def bulk_insert_3(engine, df, tn):
    meta = sqlalchemy.MetaData()
    meta.reflect(bind=engine)

    Base = sqlalchemy.ext.automap.automap_base(metadata=meta)
    Base.prepare()

    t_ORM = Base.classes.get(tn)

    lists = df.to_dict(orient='records')

    conn = engine.connect()
    conn.execute(t_ORM.__table__.insert(), lists)
    conn.close()


def update_metadata():
    global meta, inspector, engine, Base
    engine.dispose()

    meta.clear()
    meta.reflect(bind=engine)
    inspector = sqlalchemy.inspect(engine)
    Base = sqlalchemy.ext.automap.automap_base(metadata=meta)
    Base.prepare()


def upsert_wrapper(df: pd.DataFrame,
                   tn: str,
                   cols_to_ignore: list = None,
                   method: enum_upsert_method = enum_upsert_method.MEMORY,
                   chunksize: int = 5000):
    """upserts the dataframe df to the table tn"""
    if cols_to_ignore is None:
        cols_to_ignore = []

    global engine, meta, Base, session_maker, myupsert
    engine.dispose()

    tt = meta.tables.get(tn)
    cols_table = [c.name for c in tt.c]

    # subset the dataframe such that it only contains columns that are in the table
    col_choice = df.columns.intersection(cols_table)

    # ignore further columnns; e.g. useful when we have two unique keys such as
    # an autoinc primary key and another unique indexed key;
    # then we best ignore the primary key
    col_choice = col_choice.difference(cols_to_ignore)

    df2 = df[col_choice].copy()

    if method == enum_upsert_method.MEMORY:
        upsert_bulk_memory(df=df2, tt=tt, chunksize=chunksize)
    elif method == enum_upsert_method.TMPTABLE:
        upsert_bulk_with_tmp_table(df=df2, tt=tt, chunksize=chunksize)


def upsert_bulk_memory(df: pd.DataFrame, tt: sqlalchemy.Table, chunksize: int = 5000):
    global engine, meta, Base, session_maker, myupsert
    engine.dispose()

    # upsert this dataframe in chunks
    for k, g in df.groupby(np.arange(len(df)) // chunksize):
        # create the dictionary for upsert
        print (f"upserting chunk {k} into {tt.name}: {g.time.iloc[0]}, length: {len(g)}")
        d = g.to_dict(orient='records')
        engine.execute(myupsert(tt, d))

        # for performance comparison: do this one row at at a time
        # for dd in d:
        #     engine.execute(myupsert(tt,dd))


def upsert_bulk_with_tmp_table(df: pd.DataFrame, tt: sqlalchemy.Table, chunksize: int = 5000):
    # upsert dataframe using a temporary table
    global meta, inspector, engine, Base, session_maker, myupsert
    engine.dispose()

    tn = tt.name
    tn_tmp = f"{tn}_tmp"

    df.to_sql(f"{tn_tmp}", con=engine, if_exists='replace', index=False, chunksize=chunksize)

    pk = tt.primary_key
    all_columns = [c.name for c in tt.c]
    columns_to_copy = list(
        all_columns)  # now columns_to_copy and all_columns are not the same object; I can modify the first without modifying the second

    pk_name_if_single_column_and_autoincrement = None
    if (len(pk.columns) == 1 and
            isinstance(pk.columns.values()[0].type, sqlalchemy.Integer) and
            pk.columns.values()[0].autoincrement):
        pk_name_if_single_column_and_autoincrement = pk.columns.keys()[0]
        if pk_name_if_single_column_and_autoincrement in all_columns:
            columns_to_copy.remove(pk_name_if_single_column_and_autoincrement)

    # columns_to_copy now contains all columns except a single PK column if that is autoincremented
    # however, we might pass a dataframe that does not have all columns in the target table
    # in this case, we want to choose the intersection of column names in the target table and in the dataframe
    columns_to_copy = list(set(columns_to_copy).intersection(set(df.columns)))

    if engine.dialect.name.lower() in ['mysql', 'mariadb']:
        # case mysql: use on duplicate key
        # a) copy the candidate rows to a temp table

        strSQL = "INSERT INTO {0}(`{1}`) ".format(tn, "`, `".join(columns_to_copy))
        strSQL = strSQL + "\n SELECT "
        strList = [f"`t`.`{c}` as `{c}`" for c in columns_to_copy]
        strSQL = strSQL + ", ".join(strList)
        strSQL = strSQL + f"\n FROM `{tn_tmp}` as `t` "
        strSQL = strSQL + f"\n ON DUPLICATE KEY UPDATE"
        strList = [f"`{c}`=`t`.`{c}`" for c in columns_to_copy]
        strSQL = strSQL + ", ".join(strList)
        engine.execute(strSQL)

    elif engine.dialect.name.lower() in ['sqlite']:
        # case sqlite: use insert or replace
        strSQL = "INSERT OR REPLACE INTO {0}(`{1}`) ".format(tn, "`, `".join(columns_to_copy))
        strSQL = strSQL + "\n SELECT "
        strList = [f"`t`.`{c}` as `{c}`" for c in columns_to_copy]
        strSQL = strSQL + ", ".join(strList)
        strSQL = strSQL + f"\n FROM `{tn_tmp}` as `t` "
        engine.execute(strSQL)

        pass

    strSQL = f"DROP TABLE IF EXISTS {tn_tmp}"
    engine.execute(strSQL)


def get_min_max_time_single_instrument_volume_lt_0(instrument_name):
    global meta, inspector, engine, Base, session_maker
    engine.dispose()


    ssn = session_maker()
    tt = meta.tables[instrument_name]

    q = select([func.min(tt.c.time).label('min_time'), func.max(tt.c.time).label('max_time')]).where(
        tt.c.volume > 0)
    min_time, max_time = ssn.execute(q).fetchone()

    ssn.close()
    return [min_time, max_time]


def get_min_max_time_single_instrument(instrument_name):
    global meta, inspector, engine, Base, session_maker
    engine.dispose()

    ssn = session_maker()
    tt = meta.tables[instrument_name]
    min_time, max_time = \
        ssn.query(func.min(tt.c.time).label("min_time"), func.max(tt.c.time).label("max_time")).all()[0]
    return [min_time, max_time]


def get_min_max_time(instrument_names):
    global meta, inspector, engine, Base, session_maker
    engine.dispose()

    ssn = session_maker()
    min_times = []
    max_times = []
    for instrument_name in instrument_names:
        min_time, max_time = get_min_max_time_single_instrument(instrument_name)
        min_times.append(min_time)
        max_times.append(max_time)
        pass
    min_time = min(min_times)
    max_time = max(max_times)
    ssn.close()
    return [min_time, max_time]


def fillna_one_df(df, time_interval):
    """fill missing rows in the dataframe using forward fill"""
    t1 = timer()
    min_time = df['time'].iloc[0]
    max_time = df['time'].iloc[-1]

    # in contrast to numpy, the date_range function includes the "end" value
    times_without_gaps = pd.date_range(start=min_time, end=max_time, freq=time_interval)
    times_with_gaps = df['time']
    times_to_append = times_without_gaps.difference(times_with_gaps)
    t2 = timer()
    print(f"missing times calculation: {t2 - t1}")

    df_to_append = pd.DataFrame(data=None, columns=df.columns)
    df_to_append['time'] = times_to_append
    df_to_append['volume'] = 0
    t3 = timer()
    print(f"creation of dataframe to append: length: {len(df_to_append)}; dt: {t3 - t2}")

    df_without_gaps = df.append(df_to_append, sort=False)
    df_without_gaps.sort_values(by='time', inplace=True)

    t4 = timer()
    print(f"append dataframe with missing values to original dataframe: length: {len(df_without_gaps)}; dt: {t4 - t3}")

    df_without_gaps.fillna(method='ffill', inplace=True)
    t5 = timer()
    print(f"apply ffill : {t5 - t4}")

    print(f"duration total operation: {len(times_without_gaps)}; dt: {t5 - t1}")

    return df_without_gaps.copy()


def update_original_table_driver(tn: str,
                                 time_interval: pd.DateOffset,
                                 min_time: datetime.datetime,
                                 max_time: datetime.datetime,
                                 upsert_method: enum_upsert_method=enum_upsert_method.MEMORY,
                                 chunksize: int = 5000):
    # we take a market data table and fill the missing rows (the table should have one entry per time_interval)
    # we first take chunksize entries with existing data (volume > 0) and read that into a dataframe
    # We then add missing times such that the resulting dataframe has times that are exactly one time_interval apart
    # This results into a dataframe with missing times where each row with missing data the "time" column is None.
    # we then set the "volume" column of this dataframe to 0 and the "complete" column to None.
    # finally, we append this dataframe with missing timepoints to the original one and apply forwardfill
    # the result is a dataframe that has forwardfilled prices that are time_interval apart.
    # we use forwardfill because we want that prices each time point corresponds to prices that have been known in the
    # past of that dataframe; if you want, older prices are propagated to the future where prices are unavailable
    # if the dataframe extends to earlier times than the first valid price, we use backfill (because this is the next
    # best thing); of course, this means that early rows have prices that are known from the future, not the past;
    # but that is a minor inconvenience considering the small number of rows this is going to happen to.

    global meta, inspector, engine, Base, session_maker
    engine.dispose()

    tt = meta.tables.get(tn)
    min_time_table, max_time_table = get_min_max_time_single_instrument_volume_lt_0(tn)

    time_from = min_time_table
    while True:
        where_part = sqlalchemy.and_(sqlalchemy.and_(tt.c.time >= time_from, tt.c.volume > 0))
        order_part = tt.c.time.asc()

        ssn = session_maker()
        # we read chunksize rows of valid data from the table
        q = ssn.query(tt).filter(where_part).order_by(order_part).limit(chunksize)
        # print(literalquery(q))
        ssn.close()

        df = pd.read_sql(sql=q.statement, con=engine, parse_dates=["time"])
        print(time_from, len(df))

        # now we are specifying the start time of the next iteration
        # by taking the last row of this iteration, we make sure that
        # the next iteration starts with a valid time (which actually already exists in the database
        # becuse we have written the dataframe of this iteration to disk; but that is ok as we do an
        # upsert, so inserting the same line twice will not lead to corruption.
        # on the other hand, we need a valid time point to start the dataframe to make sure
        # that we do not create holes in the dataframe; we want the final data on disk to be spaced
        # evenly by time_interval. And if we start the next iteration with a time_from that is larger
        # than the latest time of this iteration, there might no time point in the original data on disk
        # corresponding to this time_from; so that would mean we retrieve a dataframe that starts later
        # than time_interval after the end of this iteration's dataframe; and that would create a hole)
        time_from = df['time'].iloc[-1]

        # forward fill the dataframe
        df_fillna = fillna_one_df(df, time_interval=time_interval)

        t1 = timer()
        # we do not want nor need to insert the values that already exist.
        # we therefore now eliminate the rows where volume > 0 which indicate a row with non-interpolated pricings
        df_fillna.reset_index(inplace=True)
        rows_to_drop_index = df_fillna[df_fillna['volume'] > 0].index
        df_fillna.drop(rows_to_drop_index, inplace=True)

        # now we take care that the "complete" entry, which has been filled using fillna before, is reset to "None" for all rows in the datframe to be upserted
        df_fillna["complete"] = None
        t2 = timer()
        print(f"drop rows that already exist: length of remaing dataframe: {len(df_fillna)}; dt: {t2 - t1}")

        # now do the actual upserting
        upsert_wrapper(df=df_fillna, tn=tn, cols_to_ignore=['ID'], method=upsert_method,
                       chunksize=chunksize)
        t3 = timer()
        print(f"upserting {tn}; {len(df_fillna)}: {t3 - t2}")
        print()

        # we stop the process when the whole table contents have been read.
        # this is the case when less than chunksize elements have been read from the bable
        if len(df) < chunksize:
            break

    # now extend the table to earlier times up to reaching min_time
    def tile_df(df_in, time_start, N, time_interval):
        times = pd.date_range(start=time_start, periods=N, freq=time_interval)
        # extract first row as array
        a = df_in.iloc[0].values
        # tile the first row for a total of N rows
        b = np.tile(a, (N, 1))
        # create a dataframe with these rows and the columns of the original dataframe
        df2 = pd.DataFrame(data=b, columns=df.columns)
        df2['time'] = times
        return df2

    if min_time < min_time_table:
        print(f"minmin {tn}, {min_time}, {min_time_table}")

        # retrieve the earliest time for which pricing was available
        where_part = sqlalchemy.and_(tt.c.time >= min_time, tt.c.volume > 0)
        order_part = tt.c.time.asc()

        ssn = session_maker()
        q = ssn.query(tt).filter(where_part).order_by(order_part).limit(1)
        # print(literalquery(q))
        ssn.close()

        df = pd.read_sql(sql=q.statement, con=engine, parse_dates=["time"])
        df['volume'] = 0
        df['complete'] = None

        # backfill the table until min_time
        time_earliest = min_time_table
        while True:
            print(f"extending towards earliest times: {tn}, {time_earliest}")
            # we have defined time_earliest below such that is updated in the loop
            df2 = tile_df(df, time_earliest - time_interval, chunksize, -time_interval)

            time_earliest = df2['time'].min()

            break_condition = False

            # eliminate the rows beyond min_time from the dataframe
            if time_earliest < min_time:
                # we have prepared the last dataframe which now extends beyound min_time
                df2.drop(df2.loc[df2['time'] < min_time].index, inplace=True)
                break_condition = True

            # now do the actual upserting
            upsert_wrapper(df=df2, tn=tn, cols_to_ignore=["ID"], method=upsert_method, chunksize=chunksize)

            if break_condition:
                break
                pass
            pass
        pass

    # we have filled the table in the direction of the earliest time
    # now fill the table in the direction of the latest time
    if max_time > max_time_table:
        # we are forward filling
        # dataframe just needs one row
        df = df.iloc[[0]].copy()
        # fill this row with None; and not with NaN!
        df = df.where(df.isnull(), None)
        df['volume'] = 0
        print(f"maxmax{tn}, {max_time}, {max_time_table}")
        time_latest = max_time_table
        while True:
            print(f"extending towards latesttimes: {tn}, {time_latest}")
            # we have defined time_latest below such that is updated in the loop
            df2 = tile_df(df, time_latest + time_interval, chunksize, time_interval)

            time_latest = df2['time'].max()

            break_condition = False

            # eliminate the rows beyond max_time from the dataframe
            if time_latest > max_time:
                # we have prepared the last dataframe which now extends beyound max_time
                df2.drop(df2.loc[df2['time'] > max_time].index, inplace=True)
                break_condition = True

            # now do the actual upserting
            upsert_wrapper(df=df2, tn=tn, cols_to_ignore=['ID'], method=upsert_method, chunksize=chunksize)

            if break_condition:
                break
                pass
            pass
        pass


def main(argv):
    print('in main')
    strDB = 'trading_OANDA_M5'
    time_interval = pd.Timedelta(minutes=5)
    # strDB = 'trading_OANDA_M1'
    # time_interval = pd.Timedelta(minutes=1)
    # strDB = 'trading_OANDA_S5'
    # time_interval = pd.Timedelta(seconds=5)
    all_times_table_name = 'all_times'

    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 300)

    global engine_mysql, engine_sqlite, engine, meta, Base, inspector, session_maker, myupsert

    engine_mysql = sqlalchemy.create_engine(f"mysql+mysqldb://bn@127.0.0.1/{strDB}", echo=False)
    engine_sqlite = sqlalchemy.create_engine(f"sqlite:///" + f"/home/bn/tradingData/{strDB}.sqlite", echo=False)

    engine = engine_mysql

    inspector = sqlalchemy.inspect(engine)
    meta = sqlalchemy.MetaData()
    meta.reflect(bind=engine)

    Base = sqlalchemy.ext.automap.automap_base(metadata=meta)
    Base.prepare()

    print(Base.classes.keys())

    session_maker = sqlalchemy.orm.sessionmaker(bind=engine)
    dialect_name = engine.dialect.name

    myupsert = MyUpsert.UpsertSQLite
    if dialect_name.lower() in ['sqlite']:
        myupsert = MyUpsert.UpsertSQLite
    elif dialect_name.lower() in ['mysql', 'mariadb']:
        myupsert = MyUpsert.UpsertMySQL


    instrument_names = ['BCO_USD', 'CN50_USD', 'DE10YB_EUR', 'DE30_EUR',
                        'EU50_EUR', 'EUR_CHF', 'EUR_GBP', 'EUR_JPY',
                        'EUR_USD', 'FR40_EUR', 'HK33_HKD', 'JP225_USD',
                        'NAS100_USD', 'SPX500_USD', 'UK100_GBP', 'UK10YB_GBP',
                        'US2000_USD', 'US30_USD', 'USB02Y_USD', 'USB05Y_USD',
                        'USB10Y_USD', 'USB30Y_USD', 'USD_CNH', 'XAU_EUR']

    instrument_names_here = sorted(instrument_names)

    instrument_names_here = instrument_names[0:]
    print(instrument_names_here)

    # t1 = timer()
    # min_time, max_time = get_min_max_time(meta=meta, session_maker=session_maker,
    #                                       instrument_names=instrument_names_here)
    # t2 = timer()
    # print(f"times: {min_time}, {max_time}: {t2 - t1}")
    #
    # session = session_maker()
    #
    # all_times_ORM = Base.classes.get(all_times_table_name)
    #
    # n_rows = session.query(func.count(all_times_ORM.ID)).scalar()
    # print(f"#rows in all_times: {n_rows}")

    # fill the missing time steps in the table

    t1 = timer()
    min_time, max_time = get_min_max_time(instrument_names=instrument_names_here)
    t2 = timer()
    print(f"times: {min_time}, {max_time}: {t2 - t1}")
    # N = 20000
    # times = pd.date_range(start="2017-01-01",periods=N,freq="5min")
    # ask_c = np.random.rand(N)
    # ask_c = [None for _ in ask_c]
    # df = pd.DataFrame({
    #     'time': times,
    #     'ask_c':ask_c
    # })
    # print(df)
    #
    # upsert_wrapper(df=df,tn="BCO_USD_bis",cols_to_ignore=[],method=enum_upsert_method.TMPTABLE,chunksize=5000)

    # instrument_names_here = [
    # 'BCO_USD', 'CN50_USD', 'DE10YB_EUR',
    # 'DE30_EUR',
    # 'EU50_EUR', 'EUR_CHF', 'EUR_GBP', 'EUR_JPY',
    # 'EUR_USD', 'FR40_EUR', 'HK33_HKD', 'JP225_USD',
    # 'NAS100_USD', 'SPX500_USD', 'UK100_GBP', 'UK10YB_GBP',
    # 'US2000_USD', 'US30_USD', 'USB02Y_USD', 'USB05Y_USD',
    # 'USB10Y_USD', 'USB30Y_USD', 'USD_CNH', 'XAU_EUR'
    # ]
    # min_time = pd.to_datetime("2003-01-13 21:25:00")


    # instrument_names_here = ['BCO_USD_bis']
    for tn in instrument_names_here:
        print(tn)
        # # create a table with one row per time_interval
        # driver = update_original_table_driver(tn=tn, time_interval=time_interval, min_time=min_time,
        #                                       max_time=max_time, upsert_method=enum_upsert_method.TMPTABLE, chunksize=20000)
        # strSQL = f"optimize table {tn}"
        # engine.execute(strSQL)

        t1 = timer()
        tt = meta.tables.get(tn)
        # update the timeslot
        stmt = tt.update().values(timeslot=func.cast(func.date_format(tt.c.time,'%Y-%m-%d %H:%i:00'),sqlalchemy.DATETIME))
        # print(literalquery(stmt))
        engine.execute(stmt)
        t2 = timer()
        print (f"creating timeslot data: dt:{t2-t1}")

        # t1 = timer()
        # # too slow
        # strSQL = f"update {tn} AS t1 SET t1.volume_in_timeslot = (SELECT SUM(t2.volume) FROM {tn} AS t2 WHERE t2.timeslot=t1.timeslot)"
        # print(strSQL)
        # engine.execute(strSQL)
        # t2 = timer()
        # print(f"alterrnative update: dt:{t2 - t1}")

        t1 = timer()
        t01 = timer()
        tmpTable = "tmpTable"
        strSQL = f"DROP TABLE IF EXISTS {tmpTable}"
        # print(strSQL)
        engine.execute(strSQL)
        t2 = timer()
        print(f"dropping tmpTable: dt:{t2 - t1}")

        t1 = timer()
        strSQL = f"CREATE TABLE {tmpTable} AS SELECT timeslot, MIN(complete) AS min_of_complete, SUM(volume) AS bb FROM {tn} GROUP BY timeslot"
        # print(strSQL)
        engine.execute(strSQL)
        t2 = timer()
        print(f"filling tmpTable with volume per timeslot data: dt:{t2 - t1}")

        t1 = timer()
        strSQL = f"CREATE INDEX timeslot ON {tmpTable}(timeslot)"
        # print(strSQL)
        engine.execute(strSQL)
        t2 = timer()
        print(f"creating index on tmpTable: dt:{t2 - t1}")

        t1 = timer()
        strSQL = f"UPDATE {tn} AS t1 LEFT JOIN {tmpTable} AS t2 ON (t1.timeslot = t2.timeslot) "
        strSQL = strSQL + "\n" + f"SET t1.volume_in_timeslot = t2.bb"
        # print(strSQL)
        engine.execute(strSQL)
        t2 = timer()
        print(f"updating by joining: dt:{t2 - t1}")

        t1 = timer()
        strSQL = f"DROP TABLE IF EXISTS {tmpTable}"
        # print(strSQL)
        engine.execute(strSQL)
        t2 = timer()
        print(f"dropping tmpTable: dt:{t2 - t1}")

        t02 = timer()
        print(f"total time tmpTable: {t02-t01}")

        t1 = timer()
        strSQL = f"UPDATE {tn} SET tradeability_volume = CASE WHEN volume_in_timeslot > 0 THEN 1 WHEN volume_in_timeslot = 0 THEN 0 END"
        # print(strSQL)
        engine.execute(strSQL)
        t2 = timer()
        print(f"setting tradeability_volume: dt:{t2 - t1}")
        print()

        t1 = timer()
        strSQL = f"OPTIMIZE TABLE {tn}"
        # print(strSQL)
        engine.execute(strSQL)
        t2 = timer()
        print(f"optimize table: dt:{t2 - t1}")
        print()

        pass

    # print("checking")
    # for tn in instrument_names_here:
    #     print(tn)
    #     tt = meta.tables.get(tn)
    #     ssn = session_maker()
    #     q = ssn.query(tt.c.ID, tt.c.time)
    #     ssn.close()
    #     df = pd.read_sql(q.statement, con=engine, parse_dates=["time"])
    #     df['diff'] = df['time'].diff()
    #     a = np.where(df['diff'] > time_interval)
    #     print(a)

    # t1 = timer()
    # for llist in chunks(instrument_names_here, 4):
    #     procs = []
    #     for index, instrument_name in enumerate(llist):
    #         proc = mp.Process(target=update_original_table_driver, args=(instrument_name, time_interval, min_time, max_time))
    #         procs.append(proc)
    #         proc.start()
    #     for proc in procs:
    #         proc.join()
    #     pass
    # t2 = timer()
    # print(f"parallel: {t2-t1}")


if __name__ == '__main__':
    sys.exit(main(sys.argv))
