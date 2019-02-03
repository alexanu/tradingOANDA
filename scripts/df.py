# coding: utf-8
from sqlalchemy import CheckConstraint, Column, DateTime, Float
from sqlalchemy.dialects.mysql import BIGINT, TINYINT
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class BCOUSD(Base):
    __tablename__ = 'BCO_USD'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class BCOUSDBi(Base):
    __tablename__ = 'BCO_USD_bis'

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))
    timeslot = Column(DateTime)
    volume_in_timeslot = Column(BIGINT(20))
    tradeability_volume = Column(TINYINT(1))
    tradeability_onine = Column(TINYINT(1))


class CN50USD(Base):
    __tablename__ = 'CN50_USD'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class DE10YBEUR(Base):
    __tablename__ = 'DE10YB_EUR'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class DE30EUR(Base):
    __tablename__ = 'DE30_EUR'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class EU50EUR(Base):
    __tablename__ = 'EU50_EUR'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class EURCHF(Base):
    __tablename__ = 'EUR_CHF'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class EURGBP(Base):
    __tablename__ = 'EUR_GBP'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class EURJPY(Base):
    __tablename__ = 'EUR_JPY'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class EURUSD(Base):
    __tablename__ = 'EUR_USD'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class FR40EUR(Base):
    __tablename__ = 'FR40_EUR'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class HK33HKD(Base):
    __tablename__ = 'HK33_HKD'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class JP225USD(Base):
    __tablename__ = 'JP225_USD'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class NAS100USD(Base):
    __tablename__ = 'NAS100_USD'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class SPX500USD(Base):
    __tablename__ = 'SPX500_USD'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class UK100GBP(Base):
    __tablename__ = 'UK100_GBP'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class UK10YBGBP(Base):
    __tablename__ = 'UK10YB_GBP'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class US2000USD(Base):
    __tablename__ = 'US2000_USD'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class US30USD(Base):
    __tablename__ = 'US30_USD'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class USB02YUSD(Base):
    __tablename__ = 'USB02Y_USD'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class USB05YUSD(Base):
    __tablename__ = 'USB05Y_USD'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class USB10YUSD(Base):
    __tablename__ = 'USB10Y_USD'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class USB30YUSD(Base):
    __tablename__ = 'USB30Y_USD'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class USDCNH(Base):
    __tablename__ = 'USD_CNH'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))


class XAUEUR(Base):
    __tablename__ = 'XAU_EUR'
    __table_args__ = (
        CheckConstraint('`complete` in (0,1)'),
    )

    ID = Column(BIGINT(20), primary_key=True)
    ask_c = Column(Float(asdecimal=True))
    ask_h = Column(Float(asdecimal=True))
    ask_l = Column(Float(asdecimal=True))
    ask_o = Column(Float(asdecimal=True))
    bid_c = Column(Float(asdecimal=True))
    bid_h = Column(Float(asdecimal=True))
    bid_l = Column(Float(asdecimal=True))
    bid_o = Column(Float(asdecimal=True))
    complete = Column(TINYINT(1))
    mid_c = Column(Float(asdecimal=True))
    mid_h = Column(Float(asdecimal=True))
    mid_l = Column(Float(asdecimal=True))
    mid_o = Column(Float(asdecimal=True))
    time = Column(DateTime, unique=True)
    volume = Column(BIGINT(20))
