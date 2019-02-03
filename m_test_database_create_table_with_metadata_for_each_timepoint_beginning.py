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
from sqlalchemy import func
# from pathos.pools import ProcessPool
# import functools
# import re
import sqlalchemy.orm
import sqlalchemy.ext.automap
import sqlalchemy.util._collections
from timeit import default_timer as timer
# from sqlalchemy.dialects.mysql import insert as sqlalchemy_insert
from trading import upsert as MyUpsert
import typing
# import multiprocessing as mp


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


def bulk_insert_4(engine, df, tn):
    meta = sqlalchemy.MetaData()
    meta.reflect(bind=engine)

    Base = sqlalchemy.ext.automap.automap_base(metadata=meta)
    Base.prepare()
    session_maker = sqlalchemy.orm.sessionmaker(bind=engine)
    dialect_name = engine.dialect.name

    myupsert = MyUpsert.UpsertSQLite
    if dialect_name.lower() in ['sqlite']:
        myupsert = MyUpsert.UpsertSQLite
    elif dialect_name.lower() in ['mysql', 'mariadb']:
        myupsert = MyUpsert.UpsertMySQL

    ssn = session_maker()

    t_ORM = Base.classes.get(tn)

    lists = df.to_dict(orient='records')

    llists = []
    for llist in chunks(lists, int(max(1, len(df) / int(1000)))):
        ssn.execute(myupsert(t_ORM, llist))

    ssn.commit()
    ssn.close()


def upsert_bulk(engine, df, tn):
    # fastest version of upsert
    meta = sqlalchemy.MetaData()
    meta.reflect(bind=engine)

    Base = sqlalchemy.ext.automap.automap_base(metadata=meta)
    Base.prepare()

    t_ORM = Base.classes.get(tn)

    dialect_name = engine.dialect.name

    df.to_sql(f"{tn}_tmp", con=engine, if_exists='replace', index=False, chunksize=10000)

    pk = t_ORM.__table__.primary_key
    all_columns = [c.name for c in t_ORM.__table__.c]
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

    if dialect_name.lower() in ['mysql', 'mariadb']:
        # case mysql: use on duplicate key
        # a) copy the candidate rows to a temp table

        strSQL = "INSERT INTO {0}(`{1}`) ".format(tn, "`, `".join(columns_to_copy))
        strSQL = strSQL + "\n SELECT "
        strList = [f"`t`.`{c}` as `{c}`" for c in columns_to_copy]
        strSQL = strSQL + ", ".join(strList)
        strSQL = strSQL + f"\n FROM `{tn}_tmp` as `t` "
        strSQL = strSQL + f"\n ON DUPLICATE KEY UPDATE"
        strList = [f"`{c}`=`t`.`{c}`" for c in columns_to_copy]
        strSQL = strSQL + ", ".join(strList)
        engine.execute(strSQL)

    elif dialect_name.lower() in ['sqlite']:
        # case sqlite: use insert or replace
        strSQL = "INSERT OR REPLACE INTO {0}(`{1}`) ".format(tn, "`, `".join(columns_to_copy))
        strSQL = strSQL + "\n SELECT "
        strList = [f"`t`.`{c}` as `{c}`" for c in columns_to_copy]
        strSQL = strSQL + ", ".join(strList)
        strSQL = strSQL + f"\n FROM `{tn}_tmp` as `t` "
        engine.execute(strSQL)

        pass

    engine.execute(f"DROP TABLE IF EXISTS {tn}_tmp")


def get_min_max_time(meta, session_maker, instrument_names):
    session = session_maker()
    min_times = []
    max_times = []
    for instrument_name in instrument_names:
        tt = meta.tables[instrument_name]
        min_time, max_time = \
        session.query(func.min(tt.c.time).label("min_time"), func.max(tt.c.time).label("max_time")).all()[0]
        min_times.append(min_time)
        max_times.append(max_time)
        pass
    min_time = min(min_times)
    max_time = max(max_times)
    session.close()
    return [min_time, max_time]


def create_all_times_only_time_and_instrument_name(min_time: pd.datetime,
                                                   max_time: pd.datetime,
                                                   time_interval: pd.DateOffset,
                                                   instrument_names: typing.List[str],
                                                   all_times_table_name: str,
                                                   ):
    # recreates the all_times table
    # only fills the time and instrument_name axis
    # the rest of the columns is set to NULL
    # this operation takes a long time, so we create a specific function only dedicated to this task

    # create an array with all times between min_time and max_time spaced by time_interval
    times = pd.date_range(start=min_time, end=max_time, freq=time_interval).values
    times = times[0:50000]

    # we cannot allocate a dataframe that spans all time points and all instruments.
    # this would be too large.
    # We therefore loop over the instruments, and for each instrument we create a dataframe
    # containing all time points, yielding about 4 GB (for 2002 to 2019)
    for instrument_name in instrument_names:
        print(instrument_name)

        df = pd.DataFrame(data=times, columns=['time'])
        df['instrument_name'] = instrument_name
        # print(df.info())

        upsert_bulk(engine, df, all_times_table_name)
        pass


def fill_columns(time_interval: pd.DateOffset,
                 instrument_names: typing.List[str],
                 all_times_table_name: str,
                 ):
    # fills the all_times table
    # based on information in the instrument_names tables
    # assumes that the columns `mapped_time`, `tradeability`, and `tradeability_time_slot` are set to NULL

    global engine
    engine.dispose()

    meta = sqlalchemy.MetaData()
    meta.reflect(bind=engine)

    Base = sqlalchemy.ext.automap.automap_base(metadata=meta)
    Base.prepare()

    all_times_ORM = Base.classes.get(all_times_table_name)
    all_times_table = all_times_ORM.__table__

    dialect_name = engine.dialect.name

    from sqlalchemy.sql import select
    # we cannot allocate a dataframe that spans all time points and all instruments.
    # this would be too large.
    # We therefore loop over the instruments, and for each instrument we create a dataframe
    # containing all time points, yielding about 4 GB (for 2002 to 2019)
    for instrument_name in instrument_names:
        print(instrument_name)
        t_ORM = Base.classes.get(instrument_name)
        # now update the mapped_time columns for those times that are available

        t_table = t_ORM.__table__

        # strSQL = f"UPDATE {all_times_table_name} AS t1 INNER JOIN {instrument_name} AS t2 ON t1.time = t2.time and t1.instrument_name = '{instrument_name}' SET t1.mapped_time = t2.time "
        # print(strSQL)
        #
        # conn = engine.connect()
        #
        # conn.execute(strSQL)
        #
        # conn.close()

        q = all_times_table.update().values(mapped_time=all_times_table.c.time).where(
            sqlalchemy.and_(all_times_table.c.time == select([t_table.c.time]).where(
                t_table.c.time == all_times_table.c.time).as_scalar(),
                            all_times_table.c.instrument_name == instrument_name)
        )

        print(type(select))
        print(str(q))
        print(literalquery(q))

        engine.execute(q)

        pass


def update_original_table(tn:str,
                          time_interval: pd.DateOffset,
                    min_time: datetime.datetime,
                    max_time: datetime.datetime,
                    ):
    global meta, inspector, engine, Base

    print(tn)

    tbl = meta.tables.get(tn)

    strSQL = f"SELECT * FROM {tn} ORDER BY time"
    t1 = timer()

    for i,df_orig in enumerate(pd.read_sql(strSQL, con=engine, parse_dates=["time"],chunksize=100000)):
        t2 = timer()
        print(f"dataframe extraction: {t2-t1}")

        max_time = df_orig.time.iloc[-1]
        t3 = timer()
        print(f"max: {t3-t2}")


        times_without_gaps = pd.date_range(start=min_time, end=max_time,freq=time_interval)
        t4 = timer()
        times_with_gaps = df_orig['time']
        times_to_append = times_without_gaps.difference(times_with_gaps)

        t4 = timer()
        print(f"missing times calculation: {t4 - t3}")


        df_to_append = pd.DataFrame(data=None, columns=df_orig.columns, index=np.arange(len(df_orig),len(df_orig)+len(times_to_append)))
        df_to_append['volume']=0
        df_to_append['complete']= None
        df_to_append['time'] = times_to_append
        t5 = timer()
        print(f"creation of dataframe to append: {t5 - t4}")

        df_without_gaps = df_orig.append(df_to_append,sort=False)
        df_without_gaps.sort_values(by='time',inplace=True)

        t6 = timer()
        print(f"append dataframe with missing values : {t6 - t5}")

        if i == 0:
            # this is the first chunk for this instrument name
            # we do a forward fill for all data (best guess for a price is the last available price)
            df_without_gaps.fillna(method='ffill',inplace=True)
            # as this is the first chunk, there might be records at the beginning of the dataset
            # for which we have no data.
            # we can do a backfill in this case, as the best information we have
            # is the next valid price.
            df_without_gaps.fillna(method='bfill', inplace=True)
        else:
            # this is not the first chunk, so we cannot do backfill to fill possible NaNs at the beginning.
            # instead, if the first row is NaN (indicated by volumne == 0), then we can copy the values of the
            # last row of the previous chunk as this corresponds to forward fill over the chunk boundaries.
            # because each chunk that we are pulling is guaranteed to end with max_time corresponding to a row
            # for which we have data, we can really take the last row of the last chunk
            my_tester = df_without_gaps['volume'].iloc[0]
            if my_tester <= 0:
                df_last_row["volume"] = 0
                df_last_row["complete"] = None
                df_without_gaps.iloc[0] = df_last_row.iloc[0]
                # print(df_without_gaps.head())
                pass
            # now the first row is filled, and the last row is filled by definition.
            # we can therefore perform ffill
            df_without_gaps.fillna(method='ffill', inplace=True)
            pass

        t7 = timer()
        print(f"apply ffill : {t7 - t6}")

        # min_t_gap_from = times_to_append[np.where(times_to_append > df_orig['time'][0])[0][0]] - time_interval
        # min_t_gap_from_index = np.where(df_orig['time']==min_t_gap_from)[0][0]
        # min_t_gap_to = df_orig['time'].iloc[min_t_gap_from_index+1]
        # time_range_with_gaps_for_display = pd.date_range(start=min_t_gap_from,end=min_t_gap_to,freq=time_interval)
        # print(df_orig.loc[df_orig['time'].isin(time_range_with_gaps_for_display)])
        # print(df_without_gaps.loc[df_without_gaps['time'].isin(time_range_with_gaps_for_display)])
        # print(df_without_gaps.head())
        # print("...")
        # print(df_without_gaps.tail())

        t8 = timer()
        # print(f"info example for missing times calculation: {t8 - t7}")




        # for the next iteration, the min_time is definied to be the last time after the last row of the last chunk
        # we do this because we want to have all times, and we know that the last row of the last chunk was a row
        # for which pricing data exists
        min_time = max_time + time_interval
        # save the last row of the last chunk to replace the first row of the next chunk if no pricing data is available
        df_last_row = df_without_gaps.iloc[-1:,:].copy()

        upsert_bulk(engine=engine, df=df_without_gaps, tn=tn)

        t1 = t8

    pass


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

    instrument_names_here = instrument_names[0:]

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
    # instrument_names_here = ["BCO_USD_bis"]
    t1 = timer()
    min_time, max_time = get_min_max_time(meta=meta, session_maker=session_maker,
                                          instrument_names=instrument_names_here)
    t2 = timer()
    print(f"times: {min_time}, {max_time}: {t2 - t1}")

    for instrument_name in instrument_names_here:
        print(instrument_name)
        update_original_table(tn=instrument_name, time_interval=time_interval, min_time=min_time-time_interval, max_time=max_time)
        pass

    # t1 = timer()
    # procs = []
    # for index, instrument_name in enumerate(instrument_names_here):
    #     proc = mp.Process(target=update_original_table, args=(instrument_name, time_interval, min_time, max_time))
    #     procs.append(proc)
    #     proc.start()
    # for proc in procs:
    #     proc.join()
    # t2 = timer()
    # print(f"parallel: {t2-t1}")

    # fill the all_times table by upserting the columns time and instrument_name
    # this command therefore only creates the bare structure of the table
    # the other columns have to be filled lateron
    # create_all_times_only_time_and_instrument_name(min_time=min_time,max_time=max_time,time_interval=time_interval,
    #                                                instrument_names=instrument_names_here,
    #                                                all_times_table_name=all_times_table_name,
    #                                                )

    # # loop over all instrument_names and fill the column
    # t1 = timer()
    # fill_columns(time_interval=time_interval, instrument_names=instrument_names_here, all_times_table_name=all_times_table_name)
    # t2 = timer()
    # print(f"fillingg {t2-t1}")

    # ##### bulk inserts
    # N  = 2**15
    # tn = "aa"
    # a = np.random.rand(N,2)
    # df = pd.DataFrame(data=a,columns=['bla','blubb'])
    # df["t1"] = np.arange(1*2, N*2+1,step=2)
    # df["t2"] = np.arange(1,N*2,step=2)
    #
    # engine.execute(f"DELETE FROM {tn}")
    #
    # t1 = timer()
    # upsert_bulk(engine,df,tn)
    # t2 = timer()
    # print(f"upsert_bulk insert: {t2-t1}")

    #######

    # ################################
    # # update a column from a datetime column such that it emulates the postgresql date_trunc function (truncating to minumtes)
    # # first, create a table on which we want to truncate a date colunn
    # q = session.query(all_times_ORM.time.label("time_1"),all_times_ORM.time.label("time_2"),func.cast(func.date_format(all_times_ORM.time,'%Y-%m-%d %H:%i:00'),sqlalchemy.DATETIME).label("max_time")).limit(100).statement
    # df = pd.read_sql(q,con=engine)
    #
    # df.to_sql('all_times_2',con=engine,if_exists='replace',index=False)
    # # add a primary key such that the reflect system sees the table
    # session.execute('alter table all_times_2 add id serial primary key')
    #
    # # this is the table
    # all_times_2_ORM = Base.classes.get('all_times_2')
    # # this is the update statement
    # stmt = all_times_2_ORM.__table__.update().values(time_2=func.cast(func.date_format(all_times_2_ORM.time_1,'%Y-%m-%d %H:%i:00'),sqlalchemy.DATETIME))
    # conn = session.connection()
    # conn.execute(stmt)
    # session.commit()
    # session.close()
    # ########################################

    # t1 = timer()
    #
    # session = session_maker()
    # conn = session.connection()
    # t_i = min_time
    # status = True
    # all_times_ORM = Base.classes.get("all_times")
    # while status:
    #     if (t_i.minute == 0 and t_i.second==0):
    #         print(t_i)
    #         session.commit()
    #         conn = session.connection()
    #     t_i_minute = t_i.replace(second=0, microsecond=0)
    #     for tn in instrument_names_here:
    #         tt_ORM = Base.classes.get(tn)
    #         tt = tt_ORM.__table__
    #         q = session.query(func.max(tt.c.time).label("min_time")).filter(tt.c.time<=t_i)
    #         mapped_time = q.first()[0]
    #         if mapped_time is None:
    #             q = session.query(func.min(tt.c.time).label("max_time")).filter(tt.c.time > t_i)
    #             mapped_time = q.first()[0]
    #             pass
    #         # print(f"{tn}: {mapped_time}")
    #
    #         tradeability = -1
    #         tradeability_time_slot = None
    #         if t_i == mapped_time:
    #             tradeability = True
    #             tradeability_time_slot = t_i_minute
    #             pass
    #
    #
    #         insert_stmt = sqlalchemy_insert(all_times_ORM).values(
    #             time=t_i,
    #             Instrument_Name = tn,
    #             mapped_time=mapped_time,
    #             tradeability=tradeability,
    #             tradeability_time_slot=tradeability_time_slot
    #         )
    #
    #         on_duplicate_key_stmt = insert_stmt.on_duplicate_key_update(
    #             mapped_time=mapped_time,
    #             tradeability=tradeability,
    #             tradeability_time_slot=tradeability_time_slot,
    #         )
    #
    #         conn.execute(on_duplicate_key_stmt)
    #
    #
    #         pass
    #
    #     if t_i == min_time + pd.Timedelta(days=5):
    #         status = False
    #         pass
    #     t_i = t_i + pd.Timedelta(seconds=5)
    #     pass
    #
    # t2 = timer()
    # print("?: {0}".format(t2-t1))
    # session.commit()
    # session.close()
    # pass



if __name__ == '__main__':
    sys.exit(main(sys.argv))
