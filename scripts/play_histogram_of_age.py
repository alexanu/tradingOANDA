"""
defines a command-line utility for the creation of the database
"""

import sys
import argparse
from configparser import ConfigParser, ExtendedInterpolation
import os
import pandas as pd
import typing

from tradingOANDA.tradingDB.database import TradingDB
from tradingOANDA.tradingDB.database import  literalquery

from sqlalchemy.orm import Session, Query
from sqlalchemy.sql import func
from sqlalchemy import literal_column
from sqlalchemy import text





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

    tradingDB = TradingDB(DBUsername=DBUsername,
                             DBHost=DBHost,
                             DBName=DBName,
                             timeInterval=timeInterval,
                             InstrumentNames=InstrumentNames)

    return tradingDB


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--configFile', help='Config File Name', required=True, type=str)


def draw_histogram(tradingDB:TradingDB, tn:str):

    to = tradingDB.get_market_data_orm_table(tn)


    ssn:Session = tradingDB.sessionmaker()
    q = ssn.query(to.time,to.age,to.volume).order_by(to.time)
    print(literalquery(q))
    ssn.close()

    df = pd.read_sql(q.statement,con=tradingDB.engine)
    df['diff_age'] = df['age'].diff()

    df2 = df[df['diff_age'].shift(-1) <0].copy()
    import matplotlib.pyplot as plt

    df2[df2['age']>100]['age'].hist()
    plt.show()

    pass


def main(argv):
    args = parser.parse_args(argv[1:])
    tradingDB = load_database(args)
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 300)
    draw_histogram(tradingDB=tradingDB,tn="BCO_USD")


if __name__ == '__main__':
    sys.exit(main(sys.argv))
