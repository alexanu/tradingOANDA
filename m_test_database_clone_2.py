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
# from timeit import default_timer as timer
# import multiprocessing as mp
# import random
import sqlalchemy.ext.automap
# from sqlalchemy import func


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

def update_metadata():
    global meta, inspector, engine, Base
    engine.dispose()
    meta.clear()
    meta.reflect(bind=engine)
    inspector = sqlalchemy.inspect(engine)
    Base = sqlalchemy.ext.automap.automap_base(metadata=meta)
    Base.prepare()

def cloneTable(tn, tt):
    global meta, inspector, engine, Base
    engine.dispose()
    if tn in meta.tables.keys():
        tt_clone = meta.tables.get(tn)
        tt_clone.drop(bind=engine, checkfirst=True)
        update_metadata()
        pass

    tt_clone = tt._clone()
    tt_clone.name= tn

    return tt_clone


def main(argv):

    print('in main')
    strDB = 'trading_OANDA_S5'
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 300)



    engine_mysql = sqlalchemy.create_engine(f"mysql+mysqlconnector://bn@127.0.0.1/{strDB}", echo=False)
    engine_sqlite = sqlalchemy.create_engine(f"sqlite:///" + f"/home/bn/tradingData/{strDB}.sqlite", echo=False)

    global engine
    engine = engine_mysql

    global inspector
    inspector = sqlalchemy.inspect(engine)

    global meta
    meta = sqlalchemy.MetaData()
    meta.reflect(bind=engine)

    global Base
    Base = sqlalchemy.ext.automap.automap_base(metadata=meta)
    Base.prepare()

    global session_maker
    session_maker = sqlalchemy.orm.sessionmaker(bind=engine)

    instrument_names = ['BCO_USD', 'CN50_USD', 'DE10YB_EUR', 'DE30_EUR',
                        'EU50_EUR', 'EUR_CHF', 'EUR_GBP', 'EUR_JPY',
                        'EUR_USD','FR40_EUR', 'HK33_HKD', 'JP225_USD',
                        'NAS100_USD', 'SPX500_USD', 'UK100_GBP', 'UK10YB_GBP',
                        'US2000_USD','US30_USD', 'USB02Y_USD', 'USB05Y_USD',
                        'USB10Y_USD', 'USB30Y_USD', 'USD_CNH', 'XAU_EUR']

    instrument_names_here = instrument_names[0:]


    tables = meta.tables

    # tn_new = "a_mytable"
    # tn_new2 = "a_mytable2"
    # if tn_new in tables.keys():
    #     mytable = meta.tables.get(tn_new)
    #     print(mytable.primary_key)
    #     print(tables.get('all_times').primary_key)
    #     print ("df",[c.name for c in tables.get('all_times').constraints])
    #     mytable.drop(bind=engine)
    #     update_metadata()
    #     pass
    #
    # mytable = sqlalchemy.schema.Table(tn_new, meta,
    #                                       sqlalchemy.schema.Column('ID', sqlalchemy.BigInteger, primary_key=True, autoincrement=True, comment="oha"),
    #                                       sqlalchemy.schema.Column('time', sqlalchemy.DateTime(timezone=False)),
    #                                       sqlalchemy.schema.Column('instrument_name', sqlalchemy.BigInteger),
    #                                       sqlalchemy.schema.Column('someint', sqlalchemy.BigInteger,index=True),
    #                                       sqlalchemy.schema.Column('t_1', sqlalchemy.BigInteger),
    #                                       sqlalchemy.schema.Index(f"ix_{tn_new}_time_IN",*["time","instrument_name"])
    #                                       )
    # mytable.create(bind=engine)
    # update_metadata()
    #
    # mytable2 = cloneTable(tn_new2, mytable)
    # mytable2.create(bind=engine,checkfirst=True)
    # update_metadata()

    # instrument_names_here = ["EUR_USD"]
    # instrument_names_here = ["BCO)USD","CN50_USD"]
    # instrument_names_here = ["BCO_USD","CN50_USD"]

    # clone our tables, then fill them with the contents of the original table
    for instrument_name in instrument_names_here:
        print(instrument_name)
        tt = tables.get(f"{instrument_name}")
        tt_clone = cloneTable(f"{instrument_name}_bis", tt)
        tt_clone.create(bind=engine,checkfirst=True)
        update_metadata()
        # necessary for some reason; if we do not do that, the name of tt is that of the clone ...
        tt = tables.get(f"{instrument_name}")

        # print(inspector.get_table_options(tt.name))
        # print(inspector.get_table_options(tt_clone.name))

        strSQL = f"ALTER TABLE {instrument_name}_bis PAGE_COMPRESSED=1"
        print(strSQL)
        engine.execute(strSQL)

        sel = sqlalchemy.select([c for c in tt.c if c.name not in ["ID"]])
        # print(sel)

        q = tt_clone \
            .insert() \
            .from_select(names = [c.name for c in tt_clone.c if c.name not in ["ID"]], select=sel)
        print(literalquery(q))
        # engine.execute(q)


        # strSQL = f"OPTIMIZE TABLE {instrument_name}_bis"
        # print(strSQL)
        # engine.execute(strSQL)


        # print()

        pass





    # ssn = session_maker()
    # t1 = timer()
    # n_rows_instruments = 0
    # for instrument_name in instrument_names_here:
    #     tt = tables.get(instrument_name)
    #     n_rows = ssn.query(tt.c.time).count()
    #     print(instrument_name,n_rows)
    #     n_rows_instruments += n_rows
    #     pass
    # t2 = timer()
    # print(f"selection: {t2-t1}")
    # print(f"Total number of rows in all instruments: {n_rows_instruments}")
    # n_rows_all_times = ssn.query(tables.get('all_times').c.time).count()
    # print(f"Total number of rows in all_times: {n_rows_instruments}")
    # print(f"fraction: {n_rows_instruments/n_rows_all_times}")

    # ssn.close()



    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))


