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
# import multiprocessing as mp
# import functools
# import re
from timeit import default_timer as timer
import sqlalchemy.ext.automap
from sqlalchemy import func
# import migrate
# import migrate.changeset
# from sqlalchemy.sql import select
import typing

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


def get_tables():
    global meta, inspector, engine, Base, session_maker
    engine.dispose()

    a = inspector.get_table_names()
    return a

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

def make_index_unique(tn):
    global meta, inspector, engine, Base, session_maker
    engine.dispose()

    pass

def optimize_table(tn):
    global meta, inspector, engine, Base, session_maker
    engine.dispose()


    print(tn)

    n_rows = count_rows (tn)
    print(tn, n_rows)

    strSQL = f"OPTIMIZE TABLE {tn}"
    print(strSQL)
    engine_mysql.execute(strSQL)


def delete_where_volume_0(tn):
    global meta, inspector, engine, Base, session_maker
    engine.dispose()

    # t_cutoff = datetime.datetime(year=2019,month=1,day=4,hour=21,minute=55)

    print(tn)
    tt = meta.tables.get(tn)
    ssn = session_maker()
    q = tt.delete().where(tt.c.volume == 0)
    # q = tt.delete().where(sqlalchemy.and_(tt.c.volume == 0,tt.c.time > t_cutoff))
    # q = tt.delete().where(sqlalchemy.and_(tt.c.time > t_cutoff))
    ssn.execute(q)
    ssn.commit()
    ssn.close()



def change_index_on_time_to_unique(tn):
    global meta, inspector, engine, Base, session_maker
    engine.dispose()

    tt = meta.tables.get(tn)

    indexes_existing = list(tt.indexes)
    indexes_to_drop = []
    indexes_to_create = []
    for index in indexes_existing:
        print(index)
        if not index.unique:
            if index.columns.keys() == ['time']:
                indexes_to_drop.append(index)

                new_name = index.name
                new_column = index.columns.get('time')

                new_index = sqlalchemy.Index(new_name, new_column, unique=True)
                indexes_to_create.append(new_index)
                pass
            pass
        pass


    for index in indexes_to_drop:
        index.drop(bind=engine)

    for index in indexes_to_create:
        index.create(bind=engine)
    pass

def add_index_on_timeslot(tn):
    global meta, inspector, engine, Base, session_maker
    engine.dispose()

    tt = meta.tables.get(tn)

    indexes_existing = list(tt.indexes)
    indexes_to_drop = []
    indexes_to_create = []
    for index in indexes_existing:
        print(index)
        if index.columns.keys() == ['timeslot']:
            indexes_to_drop.append(index)

            pass
        pass


    for index in indexes_to_drop:
        print(f"dropping index: {index}")
        index.drop(bind=engine)

    new_name = 'timeslot'
    new_column = tt.c.get('timeslot')

    new_index = sqlalchemy.Index(new_name, new_column, unique=False)
    new_index.create(bind=engine)

def get_min_max_time_single_instrument_volume_lt_0(instrument_name):
    global meta, inspector, engine, Base, session_maker
    engine.dispose()

    from sqlalchemy.sql import select

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
    ssn.close()
    return [min_time, max_time]


def get_min_max_time( instrument_names):
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

def change_compressed_status(tn):
    global meta, inspector, engine, Base, session_maker
    engine.dispose()

    print(tn)

    # two compression systems:
    # needs to be optimized before one can see the effect.
    # ls -s is necessary to see the real size on disk
    # strSQL = f"alter table {tn} PAGE_COMPRESSED=1;"
    # classical compression
    # strSQL = f"alter table {tn} ROW_FORMAT = COMPRESSED;"

    # strSQL = f"alter table {tn} ROW_FORMAT= DYNAMIC;"
    # print(strSQL)
    # engine_mysql.execute(strSQL)

    strSQL = f"alter table {tn} PAGE_COMPRESSED = 1;"
    print(strSQL)
    engine_mysql.execute(strSQL)

def change_column_types(tn, from_type, to_type):
    global meta, inspector, engine, Base, session_maker
    engine.dispose()

    print(tn)
    tt = meta.tables.get(tn)
    for c in tt.c:
        if isinstance(c.type,from_type):
            print(tn,c.name,c.type, from_type,to_type)
            # c.alter(type=to_type, engine=engine)
            pass
        pass
    pass

def add_column(tn, col_names: typing.List[str],col_types: typing.List[str]):
    global meta, inspector, engine, Base, session_maker
    engine.dispose()

    print(tn)
    tt = meta.tables.get(tn)
    cns = [c.name for c in tt.c]

    # drop columns
    for col_name in col_names:
        cs_to_drop = []
        if col_name in cns:
            c = [c for c in tt.c if c.name==col_name][0]
            cs_to_drop.append(c)
            pass
        for c in cs_to_drop:
            strSQL = f"alter table {tn} drop column {c.name}"
            print(strSQL)
            engine.execute(strSQL)
        pass

    # col_names_to_add = list(set(col_names).difference(set(cns)))
    # if len(col_names_to_add) > 0:
    #     strSQL = f"alter table {tn} "
    #     for col_name, col_type in zip(col_names,col_types):
    #         if col_name not in cns:
    #             strSQL = strSQL + F"\nadd {col_name} {col_type} null, "
    #     strSQL = strSQL[:-2]
    #     print(strSQL)
    #     engine.execute(strSQL)
    #     pass
    # pass

def create_new_sub_1(tn,schema_from,schema_to):
    global meta, inspector, engine, Base, session_maker
    engine.dispose()

    print(tn)


    strSQL = f"delete from {schema_to}.{tn}"
    print(strSQL)
    engine.execute(strSQL)

    tt = meta.tables.get(tn)
    columns = [c.name for c in tt.c if c.name != "ID"]
    columns_as_string = ", ".join(columns)

    strSQL = f"INSERT INTO {schema_to}.{tn} ({columns_as_string}) SELECT {columns_as_string} FROM {schema_from}.{tn} ORDER BY time DESC LIMIT 1000000"
    print(strSQL)
    engine.execute(strSQL)

    strSQL = f"OPTIMIZE TABLE {schema_to}.{tn}"
    print(strSQL)
    engine.execute(strSQL)


def main(argv):
    print('in main')
    strDB = 'trading_OANDA_M5'
    time_interval = pd.Timedelta(minutes=5)
    strDB = 'trading_OANDA_M1'
    time_interval = pd.Timedelta(minutes=1)
    # strDB = 'trading_OANDA_S5'
    # time_interval = pd.Timedelta(seconds=5)
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

    instrument_names = ['BCO_USD', 'CN50_USD', 'DE10YB_EUR', 'DE30_EUR',
                        'EU50_EUR', 'EUR_CHF', 'EUR_GBP', 'EUR_JPY',
                        'EUR_USD', 'FR40_EUR', 'HK33_HKD', 'JP225_USD',
                        'NAS100_USD', 'SPX500_USD', 'UK100_GBP', 'UK10YB_GBP',
                        'US2000_USD', 'US30_USD', 'USB02Y_USD', 'USB05Y_USD',
                        'USB10Y_USD', 'USB30Y_USD', 'USD_CNH', 'XAU_EUR']

    instrument_names_here = sorted(instrument_names)
    instrument_names_here = instrument_names[0:]
    print(instrument_names_here)

    # tables = meta.tables.keys()
    # tables = sorted(tables)
    # _inh = list(set(tables)-set(instrument_names_here))
    # _inh = tables
    #
    # instrument_names_here = [f"{t}_bis" for t in instrument_names_here]
    # instrument_names_here = ['BCO_USD_bis']

    t1 = timer()
    for tn in instrument_names_here:
        print(tn)
        mit, mat = get_min_max_time_single_instrument(tn)
        print(mit,mat)

        # delete_where_volume_0(tn)
        # optimize_table(tn)

        # min_time, max_time = get_min_max_time_single_instrument(tn)
        # tt = meta.tables.get(tn)
        # ssn = session_maker()
        # last_mid_price = ssn.query(tt.c.mid_c).filter(tt.c.volume>0).order_by(tt.c.time.desc()).limit(1).all()[0][0]
        # min_time_volume_lt_0, max_time_volume_lt_0 = get_min_max_time_single_instrument_volume_lt_0(tn)
        # ssn.close()
        # print(tn, count_rows(tn), min_time, min_time_volume_lt_0,max_time_volume_lt_0,max_time,last_mid_price)

        add_index_on_timeslot(tn)

        # change_index_on_time_to_unique(tn)
        # change_compressed_status(tn)
        # change_column_types(tn,sqlalchemy.types.FLOAT,sqlalchemy.types.REAL)
        # change_column_types(tn, sqlalchemy.dialects.mysql.BIT, sqlalchemy.BOOLEAN)
        # add_column(tn,["timeslot","volume_in_timeslot","tradeability_volume","tradeability_online"],["DATETIME","BIGINT","SMALLINT","SMALLINT"])
        # change_column_types(tn,sqlalchemy.dialects.mysql.SMALLINT,sqlalchemy.BOOLEAN)
        # create_new_sub_1(tn,'trading_OANDA_S5','trading_OANDA_S5_sub_1')

        # n_rows = count_rows(tn)
        # min_time, max_time = get_min_max_time_single_instrument(tn)
        # print(f"{tn} \t {n_rows} \t {min_time} \t {max_time}")
        # # print (n_rows/((max_time - min_time) / time_interval + 1))
        # df = pd.read_sql(f"SELECT * FROM {tn} ORDER BY time DESC LIMIT 1",con=engine)
        # print(df)
    pass
    t2 = timer()
    print (f"serial: dt={t2-t1}")

    # t1 = timer()
    # for llist in chunks(instrument_names_here, 4):
    #     procs = []
    #     for index, instrument_name in enumerate(llist):
    #         # proc = mp.Process(target=delete_where_volume_0, args=(instrument_name,))
    #         # proc = mp.Process(target=optimize_table,args=(instrument_name,))
    #         proc = mp.Process(target=change_column_types,args=(instrument_name,sqlalchemy.types.FLOAT,sqlalchemy.types.REAL))
    #         procs.append(proc)
    #         proc.start()
    #     for proc in procs:
    #         proc.join()
    #     pass
    # t2 = timer()
    # print(f"parallel: {t2-t1}")






    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))


