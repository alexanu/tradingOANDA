import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from tradingOANDA import upsert as MyUpsert
import pandas as pd
import numpy as np
import typing
from sqlalchemy import Column, DateTime, BIGINT, REAL, BOOLEAN
from sqlalchemy.ext import automap
import sys

class tORM1(declarative_base()):
    __tablename__ = "tn"
    __table_args__ = (
        {
            'mysql_row_format': 'DYNAMIC',
            'mysql_page_compressed': '1',
        }
    )

    ID = Column(BIGINT, primary_key=True)
    ask_c = Column(REAL)
    ask_h = Column(REAL)
    ask_l = Column(REAL)
    ask_o = Column(REAL)
    bid_c = Column(REAL)
    bid_h = Column(REAL)
    bid_l = Column(REAL)
    bid_o = Column(REAL)
    complete = Column(BOOLEAN)
    mid_h = Column(REAL)
    mid_c = Column(REAL)
    mid_l = Column(REAL)
    mid_o = Column(REAL)
    time = Column(DateTime(timezone=False), unique=True)
    volume = Column(BIGINT)
    timeslot = Column(DateTime)
    volume_in_timeslot = Column(BIGINT)
    tradeability_volume = Column(BOOLEAN)
    tradeability_online = Column(BOOLEAN)