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


def create_database(args):
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
    tradingDB.createORMClasses()
    tradingDB.createAllTables()


    return tradingDB


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--configFile', help='Config File Name', required=True, type=str)


def main(argv):
    args = parser.parse_args(argv[1:])
    # create_database_test(args)
    tradingDB = create_database(args)
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 300)



if __name__ == '__main__':
    sys.exit(main(sys.argv))
