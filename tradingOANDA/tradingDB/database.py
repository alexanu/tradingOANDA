import sqlalchemy as sa
# import pandas as pd
# import numpy as np
import typing
from sqlalchemy import Column, DateTime, BIGINT, REAL, BOOLEAN, INTEGER, Integer, Index, Table, func
from sqlalchemy.ext import automap
from sqlalchemy.ext.declarative import as_declarative, declared_attr, declarative_base
from sqlalchemy.ext.declarative.api import DeclarativeMeta
from sqlalchemy.orm import Query, scoped_session, sessionmaker
from sqlalchemy.engine.base import Engine
from sqlalchemy.sql.schema import MetaData
from timeit import default_timer as timer
import pandas as pd
import numpy as np
import logging
from tradingOANDA.upsert import upsert_dataframe_to_table
from tradingOANDA.utils import enum_upsert_method, enum_extending_direction
from sqlalchemy.sql.sqltypes import String, NullType, DateTime
from sqlalchemy.engine.default import DefaultDialect
from tradingOANDA.utils import tile_df


class exception_db(Exception):
    """Base class for other exceptions"""


class exception_table_does_not_exist(exception_db):
    """Raised when an ORM table does not exist

    Attributes:
        tn -- name of the table_name that did not exist
    """

    def __init__(self, tn: str, db: str):
        self.tn = tn
        self.db = db
        self.message = f"Table {self.tn} does not exist in ORM List in db {self.db}"

    # def __repr__(self)->str:
    #     strReturn = f"{self.__class__}"
    #     return strReturn
    #
    def __str__(self) -> str:
        return self.message


class StringLiteral(String):
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


class LiteralDialect(DefaultDialect):
    colspecs = {
        # prevent various encoding explosions
        String: StringLiteral,
        # teach SA about how to literalize a datetime
        DateTime: StringLiteral,
        # don't format py2 long integers to NULL
        NullType: StringLiteral,
    }


def literalquery(statement):
    """NOTE: This is entirely insecure. DO NOT execute the resulting strings."""
    if isinstance(statement, Query):
        statement = statement.statement
    strReturn: str = statement.compile(
        dialect=LiteralDialect(),
        compile_kwargs={'literal_binds': True},
    ).string
    strReturn = strReturn.replace('"', '`')
    return strReturn


@as_declarative()
class MarketDataBaseORM(object):

    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    __abstract__ = True

    ID = Column(BIGINT, primary_key=True, comment="autoincrement primary key")
    time = Column(DateTime(timezone=False), unique=True, comment="Start of the timeslot for the candles")
    complete = Column(BOOLEAN,
                      comment="True if the timeslot was terminated when the candles request was made; else False")
    volume = Column(BIGINT, comment="number of prices observed in this timeslot")
    age = Column(BIGINT, comment=
                         "age (staleness) of the price in this row. \n"
                         "This field is 0 if a price has been sent by OANDA for this timeslot. \n"
                         "Otherwise, this field contains the number of time_intervals that the last valid price has been seen. "
                         )
    mid_c = Column(REAL, comment="mid price close")
    mid_h = Column(REAL, comment="mid price high")
    mid_l = Column(REAL, comment="mid price low")
    mid_o = Column(REAL, comment="mid price open")
    bid_c = Column(REAL, comment="bid price close")
    bid_h = Column(REAL, comment="bid price high")
    bid_l = Column(REAL, comment="bid price low")
    bid_o = Column(REAL, comment="bid price open")
    ask_c = Column(REAL, comment="ask price close")
    ask_h = Column(REAL, comment="ask price high")
    ask_l = Column(REAL, comment="ask price low")
    ask_o = Column(REAL, comment="ask price open")
    tdb_ol = Column(BOOLEAN,
                    comment="tradeability as given by OANDA's response to a v3/accounts/<accountID>/pricing request")
    to_req_t = Column(DateTime(timezone=False),
                      comment="time when sending the request to get tradeability_online")
    to_resp_t = Column(DateTime(timezone=False),
                       comment="time of server response when receiving response to get tradeability_online")
    to_price_t = Column(DateTime(timezone=False),
                        comment="time of last fixing of price when establishing tradeability_online")

    def __repr__(self):
        str_return = f"<{self.__class__.__name__}(time='{self.time}', complete='{self.complete}', volume='{self.volume}')>"
        return str_return


def MarketDataORMGenerator(tn, timeInterval: pd.Timedelta):
    class MyMappedClass(MarketDataBaseORM):
        __name__ = tn
        __qualname__ = tn
        __tablename__ = tn
        pass

        __table_args__ = (
            {
                'mysql_row_format': 'DYNAMIC',
                'mysql_page_compressed': '1',
                'mysql_comment': f"The table contains one row per timeslot. Timeslot length is {timeInterval}.\n"
                f"Timeslots for which we did not get data are forward filled starting from the most recent available data previous to this timeslot.\n"
                f"If the table extends prior to the first available data, pricing information is backfilled.",
            })

    return MyMappedClass


class TradingDB(object):
    """Class for database management"""
    __DBEngine: Engine
    __DBDeclarativeBase: DeclarativeMeta

    __slots__ = [
        '__DBUsername',
        '__DBHost',
        '__DBName',
        '__DBEngine',
        '__DBDeclarativeBase',
        '__timeInterval',
        '_logger',
        '_loggerSqlalchemy',
        'sessionmaker',
        'InstrumentNames',
        'df',
    ]

    def __init__(self,
                 DBUsername: str = None,
                 DBHost: str = None,
                 DBName: str = None,
                 timeInterval: pd.Timedelta = pd.Timedelta("5 minutes"),
                 InstrumentNames: typing.List[str] = None,
                 echo: bool = False):
        """Constructor for myDB"""
        if InstrumentNames is None:
            InstrumentNames = []
        if None in [DBUsername, DBHost, DBName]:
            return
        self.__DBUsername = DBUsername
        self.__DBHost = DBHost
        self.__DBName = DBName
        self.__timeInterval = timeInterval
        self.__DBEngine = self.__createDBEngine(echo=echo)
        self.__DBDeclarativeBase = self.__createDBDeclarativeBase()

        self.sessionmaker = self.__createSessionmaker()
        self.InstrumentNames = InstrumentNames

        # gets informations from the tables on disk; after that operation,
        # the ORM classes for all the tables are available
        self.updateMetadata()

        ### start logging
        l_frmt = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s'
        self._logger: logging.Logger = logging.getLogger('tradingOANDA')
        self._logger.setLevel(logging.DEBUG)

        # define a streaming handler
        lsh = logging.StreamHandler()
        l_frmt_stream = l_frmt
        lsh.setFormatter(logging.Formatter(l_frmt_stream))

        # add streaming handler to tradingOANDA
        self._logger.addHandler(lsh)

        # put tradingOANDA logging output into tradingOANDA.log
        l_frmt_stream = l_frmt
        lfh = logging.FileHandler('tradingOANDA.log')
        lfh.setFormatter(logging.Formatter(l_frmt_stream))  # use the same formatter as used for the streaming handler
        self._logger.addHandler(lfh)

        ## end logging

        pass

    def __createDBEngine(self, echo=False):
        retVal = sa.create_engine(f"mysql+mysqldb://{self.__DBUsername}@{self.__DBHost}/{self.__DBName}", echo=echo)
        return retVal

    def __createDBDeclarativeBase(self):
        """creates a declarative base that is bound to the engine"""
        retVal = None
        if isinstance(self.__DBEngine, sa.engine.base.Engine):
            retVal = declarative_base(bind=self.__DBEngine)
        return retVal

    def __createSessionmaker(self):
        """creates a Sessionmaker that is ready for multithreaded access to the database; bound to the engine"""
        session_factory = sessionmaker(bind=self.__DBEngine)
        return scoped_session(session_factory)

    def createORMClasses(self):
        """Creates all classes for the sqlalchemy ORM tables"""
        for tn in self.InstrumentNames:
            if not self.base.classes.has_key(tn):
                myClass = MarketDataORMGenerator(tn, timeInterval=self.timeInterval)
                # print(f"created orm class {tn}")
                # when this table is created, it is not bound to the base nor to the metadata
                # we have to add it to both things
                # add table to ORM collection for this engine
                self.base.classes.update({tn: myClass})
                # add table to metadata
                myClass.__table__._set_parent(self.meta)
                pass
            pass
        pass

    def createAllTables(self):
        """Creates tables on disk for all tables in metadata"""
        self.meta.create_all(bind=self.engine)

    def dropAllTables(self):
        self.meta.drop_all(bind=self.engine)

    def getTable(self, tn: str):
        retVal: sa.ext.declarative.api.DeclarativeMeta = self.base.classes.get(tn)
        return retVal

    def updateMetadata(self):
        self.meta.clear()
        self.meta.reflect(bind=self.engine)
        # inspector = sqlalchemy.inspect(self.engine)
        Base = automap.automap_base(metadata=self.meta)
        Base.prepare(engine=self.engine)
        self.base = Base

    def get_min_max_time_single_instrument(self, tn: str, choose_only_rows_with_volume_gt_0: bool = False):
        """get minimum/maximum time from a single instrument"""

        retVal = None
        to: MarketDataBaseORM = None
        try:
            to = self.get_market_data_orm_table(tn)
        except exception_table_does_not_exist as e:
            self.logger.error(e.message)
            pass

        if to is None:
            min_time, max_time = pd.NaT, pd.NaT
            retVal = [min_time, max_time]
            return retVal

        ssn = self.sessionmaker()
        q: Query = ssn.query(func.min(to.time).label("min_time"), func.max(to.time).label("max_time"))
        if choose_only_rows_with_volume_gt_0:
            q = q.filter(to.volume > 0)
        min_time, max_time = q.all()[0]
        ssn.close()

        retVal = [min_time, max_time]

        return retVal

    def fillna_one_df(self, df: pd.DataFrame):
        """forward fill a df such that the df contains a row for each time_interval

        we assume that this df only contains valid rows with volume > 0 and prices as fixed by OANDA
        """

        time_interval = self.timeInterval
        logger = self.logger
        # make sure there are only valid rows in the dataframe
        assert (len(df[df['volume'] == 0]) == 0)

        t1 = timer()
        df: pd.DataFrame = df.sort_values(by=['time']).copy()
        t2 = timer()
        logger.info(f"copying df to new sorted df: dt={t2 - t1}")

        # get min/max times of the dataframe
        min_time_df = df['time'].iloc[0]
        max_time_df = df['time'].iloc[-1]

        t3 = timer()
        logger.info(f"min/max time: {min_time_df} - {max_time_df}; dt={t3 - t2}")

        # pandas operation; the range includes the "end" value
        times_without_gaps = pd.date_range(start=min_time_df, end=max_time_df, freq=time_interval)
        times_with_gaps = df['time']
        times_to_append = times_without_gaps.difference(times_with_gaps)
        t4 = timer()
        logger.info(f"missing times calculation: dt={t4 - t3}")


        df_to_append = pd.DataFrame(data=None, columns=df.columns)
        df_to_append['time'] = times_to_append
        df_to_append['volume'] = 0
        df_to_append['age'] = None

        t5 = timer()
        logger.info(f"creation of dataframe to append: length: {len(df_to_append)}; dt={t5 - t4}")

        df_without_gaps: pd.DataFrame = df.append(df_to_append, sort=False)
        df_without_gaps.sort_values(by='time', inplace=True)

        t6 = timer()
        logger.info(
            f"append dataframe with missing values to original dataframe: new length: "
            f"{len(df_without_gaps)}; dt={t6 - t5}"
        )

        # apply ffill to the total dataframe with non-valid rows
        df_without_gaps.fillna(method='ffill', inplace=True)
        t7 = timer()
        logger.info(f"apply ffill : {t7 - t6}")

        # subset the dataframe to contain only time and volume
        # df2:pd.DataFrame = df_without_gaps[["time","volume"]].copy()
        # df_ is just a shorthand to df_without_gaps
        df_ = df_without_gaps
        t8 = timer()
        logger.info(f"create shortcut : {t8 - t7}")

        # calculate the indices where volume==0 (these are the invalid rows)
        df_.reset_index(inplace=True, drop=True)
        t9 = timer()
        logger.info(f"reset index: {t9 - t8}")

        # calculate indexes of invalid rows
        invalid_indices = df_[df_['volume'] == 0].index
        t10 = timer()
        logger.info(f"calculate indexes of invalid rows: {t10 - t9}")

        # set the last_valid_time column
        df_['last_valid_time'] = df_['time']
        df_['last_valid_time'].loc[invalid_indices] = None
        t11 = timer()
        logger.info(f"set last valid time column: {t11 - t10}")

        # ffill this column
        df_['last_valid_time'].fillna(method='ffill', inplace=True)
        t12 = timer()
        logger.info(f"fill last valid time column: {t12 - t11}")

        # calculate the distance to the last valid row
        df_['age'] = (df_['time'] - df_['last_valid_time']) / time_interval
        t13 = timer()
        logger.info(f"calculate distance to last valid time: {t13 - t12}")

        # drop helper column
        df_.drop(columns=['last_valid_time'], inplace=True)
        t14 = timer()
        logger.info(f"drop last valid time column: {t14 - t13}")


        logger.info(
            f"duration total operation: nRows: {len(times_without_gaps)}; "
            f"MB: {df_without_gaps.memory_usage(deep=True).sum() / 1024 / 1024}; dt: {t14 - t1}"
        )

        return df_without_gaps

    def get_min_max_time_multiple_instruments(self, tns: typing.List[str] = None,
                                              choose_only_rows_with_volume_gt_0: bool = False):

        min_times = []
        max_times = []

        if tns is None:
            tns = self.meta.tables.keys()
            pass

        for tn in tns:
            instrument = self.get_min_max_time_single_instrument(tn,
                                                                 choose_only_rows_with_volume_gt_0=choose_only_rows_with_volume_gt_0)
            min_time, max_time = instrument
            min_times.append(min_time)
            max_times.append(max_time)
            pass
        min_times = pd.Series(data=min_times)
        max_times = pd.Series(data=max_times)
        min_time = min_times.min()
        max_time = max_times.max()
        return [min_time, max_time]

    def upsert_df_to_table(self, df: pd.DataFrame, tn: str,
                           logger=logging.RootLogger(level=logging.DEBUG),
                           upsert_method: enum_upsert_method = enum_upsert_method.AUTO,
                           n_rows_cutoff_for_memory: int = 5000,
                           chunksize: int = 5000,
                           ):
        tt = self.meta.tables.get(tn)
        upsert_dataframe_to_table(df=df,
                                  engine=self.engine,
                                  tt=tt,
                                  logger=logger,
                                  upsert_method=upsert_method,
                                  n_rows_cutoff_for_memory=n_rows_cutoff_for_memory,
                                  chunksize=chunksize)

    def get_market_data_orm_table(self, tn: str) -> MarketDataBaseORM:
        """get a marketData orm table by name"""
        to: MarketDataBaseORM
        to = self.base.classes.get(tn)

        if to is None:
            e = exception_table_does_not_exist(tn=tn, db=self.__DBName)
            raise e

        return to

    @property
    def engine(self) -> Engine:
        return self.__DBEngine

    @property
    def base(self) -> DeclarativeMeta:
        return self.__DBDeclarativeBase

    @base.setter
    def base(self, value):
        self.__DBDeclarativeBase = value

    @property
    def meta(self) -> MetaData:
        return self.__DBDeclarativeBase.metadata

    @property
    def timeInterval(self) -> pd.Timedelta:
        return self.__timeInterval

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def __repr__(self):
        retVal = f"<{__name__}.{self.__class__.__qualname__}(name={self.__DBName})>"
        return retVal
