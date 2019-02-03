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
from tradingOANDA.tradingDB.database import  literalquery

from sqlalchemy.orm import Session, Query
from sqlalchemy.sql import func
from sqlalchemy import literal_column
from sqlalchemy import text


def sandbox_to_get_holidays(tradingDB: trading_oanda_database):

    # warnings.filterwarnings(action='ignore', message='W')
    # warnings.simplefilter("ignore")

    # tn = "NAS100_USD"; local_tz = "America/Chicago"; time_shift = "INTERVAL 7 HOUR"
    # tn = "US30_USD"; local_tz = "America/Chicago"; time_shift = "INTERVAL 7 HOUR"
    # tn = "SPX500_USD"; local_tz = "America/Chicago"; time_shift = "INTERVAL 7 HOUR"
    # tn = "US2000_USD"; local_tz = "America/Chicago"; time_shift = "INTERVAL 7 HOUR"
    # tn = "JP225_USD"; local_tz = "America/Chicago"; time_shift = "INTERVAL 7 HOUR"
    tn = "EUR_USD"; local_tz = "America/New_York"; time_shift = "INTERVAL 7 HOUR"
    # tn = "EUR_CHF"; local_tz = "America/New_York"; time_shift = "INTERVAL 7 HOUR"
    # tn = "EUR_JPY"; local_tz = "America/New_York"; time_shift = "INTERVAL 7 HOUR"
    # tn = "EUR_GBP"; local_tz = "America/New_York"; time_shift = "INTERVAL 7 HOUR"
    # tn = "DE30_EUR"; local_tz = "Europe/Berlin"; time_shift = "INTERVAL -1 HOUR"
    # tn = "UK100_GBP"; local_tz = "Europe/London"; time_shift = "INTERVAL 0 HOUR"
    # tn = "XAU_EUR"; local_tz = "America/New_York"; time_shift = "INTERVAL 6 HOUR"

    # Currencies: ['EUR_CHF', 'EUR_GBP', 'EUR_JPY','EUR_USD']: "America/New_York"; "INTERVAL 7 HOUR"; [[f"{t}",f"{t} (Observed)"] for t in ["New Year", "Christmas Day"]]
    # US Indices ["NAS100_USD","US30_USD","SPX500_USD","US2000_USD","JP225_USD"]; "America/Chicago"; "INTERVAL 7 HOUR";  [[f"{t}",f"{t} (Observed)"] for t in ["New Year", "Christmas Day"]] + "good friday"
    # Commodities ["XAU_EUR"]; "America/New_York"; "INTERVAL 6 HOUR";  [[f"{t}",f"{t} (Observed)"] for t in ["New Year", "Christmas Day"]] + "good friday"
    # German DAX ["DE30_EUR]: "Europe/Berlin"; "INTERVAL -1 HOUR"; ["New Year","Good Friday","Easter Monday","Labour Day","Ascension Thursday", "Day of German Unity","Christmas Day", "Second Christmas Day"] + 12-24 + 12-31
    # UK100  ["UK100_GBP]: "Europe/London"; "INTERVAL 0 HOUR"; ["New Year","Good Friday","Easter Monday","Early May Bank Holiday","Spring Bank Holiday", "Late Summer Bank Holiday","Christmas Day", "Boxing Day"]


    # Local_opening_hours_1: PD 17:00 - None; TD 00:00 - TD 15:14; TD 15:30 - TD 15:59 (NAS100_USD, US30_USD, SPX500_USD, JP225_USD, US2000_USD)
    # Local_opening_hours_2: TD 01:15 - TD 21:59 (DE30_EUR for local trading dates >= 2018-12-10)
    # Local_opening_hours_3: TD 08:00 - TD 21:59 (DE30_EUR for local trading dates < 2018-12-10)
    # Local_opening_hours_4: TD 01:00 - TD 20:59 (UK100_GBP)
    # Local_opening_hours_5: PD 18:00 - TD 16:59 (XAU_EUR)
    # Local_opening_hours_6: PD 17:00 - None; TD 00:00 - 16:59:59 (EUR_USD,EUR_CHF,EUR_GBP,EUR_JPY)

    to: trading_oanda_database.MarketDataBaseORM= tradingDB.getTable(tn=tn)

    ssn:Session = tradingDB.sessionmaker()
    c_local_time = func.convert_tz(to.time,'UTC',local_tz)
    c_local_time_shifted = func.date_add(c_local_time, text(time_shift))
    c_local_business_date = func.date(c_local_time_shifted)
    c_local_business_date_weekday = func.weekday(c_local_business_date)

    q: Query = ssn.query(
        c_local_business_date.label("local_business_date"),
        c_local_business_date_weekday.label("local_business_date_weekday"),
        func.min(c_local_time).label("min_time_local"),
        func.max(c_local_time).label("max_time_local"),
        func.sum(to.volume).label("sum_volume"),
        func.sum(to.complete).label("sum_complete"),

    )

    q = q.order_by(to.time)
    q = q.group_by(c_local_business_date)
    # q = q.filter(to.time >= "2018-01-01")
    q = q.filter(to.volume>0)
    q = q.having(literal_column("local_business_date")>="2018-11-29")
    q = q.having(literal_column("local_business_date")<="2019-01-02")
    q = q.having(literal_column("local_business_date_weekday").in_([0,1,2,3,4]))
    # q = q.having(literal_column("sum_volume").in_([0]))

    q = q.limit(30)
    print(literalquery(q))

    df = pd.read_sql(q.statement, con=tradingDB.engine,parse_dates=['min_time_local'])
    print(df)
    ssn.close()




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



def main(argv):
    args = parser.parse_args(argv[1:])
    tradingDB = load_database(args)
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 300)

    sandbox_to_get_holidays(tradingDB)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
