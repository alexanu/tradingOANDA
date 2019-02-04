"""
defines a command-line utility for the creation of the database
"""

import sys
import argparse
from configparser import ConfigParser, ExtendedInterpolation
import os
import pandas as pd
import typing
from tradingOANDA.tradingDB import database as trading_oanda_database
from sqlalchemy import inspect
from sqlalchemy.engine.reflection import Inspector
from tradingOANDA.utils import enum_upsert_method, enum_extending_direction
from timeit import default_timer as timer
import sqlalchemy as sa
from tradingOANDA.tradingDB.utils import extend_towards_extremal_times
from tradingOANDA.tradingDB.database import MarketDataBaseORM, TradingDB
# import operator
import logging
from tradingOANDA.tradingDB.database import literalquery as lq
import multiprocessing as mp



def copy_table(tradingDB: TradingDB, tn_from: str, tn_to: str):
    strSQL = f"delete from {tn_to}"
    print(strSQL)
    tradingDB.engine.execute(strSQL)

    insp: Inspector = inspect(tradingDB.engine)
    columns_from = [c['name'] for c in insp.get_columns(tn_from)]
    columns_to = [c['name'] for c in insp.get_columns(tn_to)]
    columns_to_copy = list(set(columns_from).intersection(set(columns_to)))
    columns_to_copy_as_string = ", ".join(columns_to_copy)

    strSQL = f"INSERT INTO {tn_to} ({columns_to_copy_as_string}) SELECT {columns_to_copy_as_string} FROM {tn_from} WHERE volume>0 ORDER BY time"
    print(strSQL)
    tradingDB.engine.execute(strSQL)

    # strSQL = f"OPTIMIZE TABLE {tn_to}"
    # print(strSQL)
    # tradingDB.engine.execute(strSQL)

    strSQL = f"DROP TABLE IF EXISTS {tn_from}"
    print(strSQL)
    tradingDB.engine.execute(strSQL)

    strSQL = f"RENAME TABLE {tn_to} TO {tn_from}"
    print(strSQL)
    tradingDB.engine.execute(strSQL)

    pass



def fill_table(tradingDB: trading_oanda_database.TradingDB,
               tn: str,
               min_time: pd.Timestamp,
               max_time: pd.Timestamp,
               chunksize: int,
               upsert_method: enum_upsert_method,
               n_rows_cutoff_for_memory: int):
    """fill all rows of a given table such that it contains one row per time_interval
    This function will not be used in the main program; we only need to fill rows in dataframes and upsert
    these afterwards"""

    print(tn)

    min_time_table_valid_rows, max_time_table_valid_rows = tradingDB.get_min_max_time_single_instrument(tn,
                                                                                                        choose_only_rows_with_volume_gt_0=True)
    min_time_table_total, max_time_table_total = tradingDB.get_min_max_time_single_instrument(tn,
                                                                                              choose_only_rows_with_volume_gt_0=False)

    to: MarketDataBaseORM = tradingDB.get_market_data_orm_table(tn)

    # start at the beginning fo the table where volume > 0
    time_from = min_time_table_valid_rows
    while True:
        where_part = sa.and_(to.time >= time_from, to.volume > 0)
        order_part = to.time.asc()

        ssn = tradingDB.sessionmaker()
        q = ssn.query(to).filter(where_part).order_by(order_part).limit(chunksize)
        # print(lq(q))
        ssn.close()

        df = pd.read_sql(sql=q.statement, con=tradingDB.engine)
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

        df_fillna = tradingDB.fillna_one_df(df=df)


        t1 = timer()
        # previously, we did not want nor need to insert the values that already existed.
        # now, we add the distance to previous rows (valid and not-valid); this data is not
        # present in the original data.
        # we therefore do not drop valid rows anymore
        # instead, we want to make sure that the "complete" column is set to None for non-valid rows because
        # "complete" is data coming from OANDA, and in non-valid rows we have no reason to set "complete" to either False or True
        df_fillna.reset_index(inplace=True)
        rows_invalid_index = df_fillna[df_fillna['volume'] == 0].index
        df_fillna.loc[rows_invalid_index,["complete"]] = None

        t2 = timer()
        tradingDB.logger.info(
            f"set complete column to None for invalid rows: {len(df_fillna)}; dt: {t2 - t1}")

        tradingDB.upsert_df_to_table(df=df_fillna, tn=tn, upsert_method=upsert_method, logger=tradingDB.logger,
                                     n_rows_cutoff_for_memory=n_rows_cutoff_for_memory,
                                     chunksize=chunksize)
        t3 = timer()
        tradingDB.logger.info(f"upserting {tn}; {len(df_fillna)}: {t3 - t2}")

        # we stop the process when the whole table contents have been read.
        # this is the case when less than chunksize elements have been read from the table
        if len(df) < chunksize:
            break
        pass

    # not necessary; for training, we will simply extend towards extremal times
    # based on the earliest/most recent valid rows
    # # now we are going to fill the table towards the earliest point in time requested
    # extend_towards_extremal_times(tradingDB=tradingDB, tn=tn,
    #                               direction=enum_extending_direction.EARLIER_TIMES,
    #                               extremal_time=min_time,
    #                               upsert_method=upsert_method,
    #                               n_rows_cutoff_for_memory=n_rows_cutoff_for_memory,
    #                               chunksize=chunksize)
    # # now we are going to fill the table towards the most recent point in time requested
    # extend_towards_extremal_times(tradingDB=tradingDB, tn=tn,
    #                               direction=enum_extending_direction.MORE_RECENT_TIMES,
    #                               extremal_time=max_time,
    #                               upsert_method=upsert_method,
    #                               n_rows_cutoff_for_memory=n_rows_cutoff_for_memory,
    #                               chunksize=chunksize)
    pass


def load_database(args):
    # load the config file
    config_file = args.configFile
    config = ConfigParser(interpolation=ExtendedInterpolation(), defaults=os.environ)
    config.read(config_file)

    # load DB-specific information
    DBUsername = config.get('DataBase', 'DB_username')
    DBHost = config.get('DataBase', 'DB_host')
    DBName = config.get('DataBase', 'DB_name')
    timeInterval = pd.Timedelta(config.get('DataBase', 'time_interval'))
    InstrumentNames: typing.List[str] = eval(config.get('MarketData', 'instrument_names'))

    tradingDB = trading_oanda_database.TradingDB(DBUsername=DBUsername,
                                                 DBHost=DBHost,
                                                 DBName=DBName,
                                                 timeInterval=timeInterval,
                                                 InstrumentNames=InstrumentNames)

    return tradingDB


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--configFile', help='Config File Name', required=True, type=str)
parser.add_argument('--createNewTables', help='create all tables given in config file',
                    action='store_true')
parser.add_argument('--copyBisTables',
                    help='copy from non-bis to bis tables; then delete non-bis tables; then rename bis tables to non-bis tables',
                    action='store_true')
parser.add_argument('--fillAllTables', help='fill all existing tables',
                    action='store_true')


def main(argv):
    args = parser.parse_args(argv[1:])
    print(args)
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 300)

    # load DB; this will only create the ORM classes for tables that are on disk
    tradingDB = load_database(args)

    # the tables on disk
    tables = tradingDB.base.classes.keys()
    print(tables)
    tradingDB.logger.setLevel(logging.INFO)

    # create additional tables specified in the config file (InstrumentNames)
    if args.createNewTables:
        tradingDB.createORMClasses()
        tradingDB.createAllTables()

        tradingDB.updateMetadata()
        tables = tradingDB.base.classes.keys()
        pass

    # copy tables
    if args.copyBisTables:
        t: str
        tables_to = [t for t in tables if t.endswith("_bis")]
        tables_from = [t[:-4] for t in tables_to]

        for t_from, t_to in zip(tables_from, tables_to):
            copy_table(tradingDB, t_from, t_to)
            # tradingDB.engine.execute(f"DROP TABLE IF EXISTS {t_to}")
        tradingDB.updateMetadata()
        tables = tradingDB.base.classes.keys()

    # fill tables
    if args.fillAllTables:
        chunksize = 5000
        upsert_method = enum_upsert_method.TMPTABLE
        n_rows_cutoff_for_memory = chunksize

        min_time, max_time = tradingDB.get_min_max_time_multiple_instruments(tns=tables,
                                                                             choose_only_rows_with_volume_gt_0=True)
        # # serial:
        # t1 = timer()
        # for tn in tables:
        #     print(f"{tn}")
        #     fill_table(tradingDB=tradingDB, tn=tn, min_time=min_time, max_time=max_time, chunksize=chunksize,
        #                upsert_method=upsert_method, n_rows_cutoff_for_memory=n_rows_cutoff_for_memory)
        # t2 = timer()
        # print(f'serial: dt: {t2-t1}')


        # parallel:

        t1 = timer()
        procs = []
        for i,tn in enumerate(tables):
            proc = mp.Process(target=fill_table, args=(tradingDB, tn, min_time, max_time, chunksize, upsert_method, n_rows_cutoff_for_memory, ))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        t2 = timer()
        print(f'parallel: dt: {t2-t1}')




    # tables_bis = [t for t in tables if t.endswith("_bis")]
    # for tn in tables_bis:
    #     print(f"{tn}")
    #     tradingDB.engine.execute(f"DROP TABLE IF EXISTS {tn}")

    # for tn in tables:
    #     print(tn)
    #     tradingDB.engine.execute(f"DELETE FROM {tn} WHERE volume=0")


if __name__ == '__main__':
    sys.exit(main(sys.argv))
