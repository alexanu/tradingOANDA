#!/usr/bin/env python
# Copyright (c) 2012, Tim Henderson
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of this software nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This module provides an "Upsert" statement for SQL Alchemy which uses
ON DUPLICATE KEY UPDATE to implement the insert or update semantics. It
supports doing a bulk insert.
"""

import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles
import sqlalchemy.sql.expression as expr
import re
import numpy as np
from tradingOANDA.utils import enum_upsert_method
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
import logging



class UpsertMySQL(expr.Insert):
    pass

@compiles(UpsertMySQL, "mysql")
def compile_upsert(insert_stmt, compiler, **kwargs):
    if insert_stmt._has_multi_parameters:
        keys = insert_stmt.parameters[0].keys()
    else:
        keys = insert_stmt.parameters.keys()
    keys=list(keys) # we want to be able to remove keys lateron

    pk = insert_stmt.table.primary_key
    auto = None
    if (len(pk.columns) == 1 and
            isinstance(pk.columns.values()[0].type, sa.Integer) and
            pk.columns.values()[0].autoincrement):
        auto = pk.columns.keys()[0]
        if auto in keys:
            keys.remove(auto)
            pass
        pass

    insert = compiler.visit_insert(insert_stmt, **kwargs)
    ondup = 'ON DUPLICATE KEY UPDATE'
    updates = ', '.join(
        '%s = VALUES(%s)' % (c.name, c.name)
        for c in insert_stmt.table.columns
        if c.name in keys
    )

    if auto is not None:
        last_id = '%s = LAST_INSERT_ID(%s)' % (auto, auto)
        if updates:
            updates = ', '.join((last_id, updates))
        else:
            updates = last_id
    upsert = ' '.join((insert, ondup, updates))

    return upsert

class UpsertSQLite(expr.Insert):
    pass

@compiles(UpsertSQLite, "sqlite")
def compile_upsert(insert_stmt, compiler, **kwargs):
    insert = compiler.visit_insert(insert_stmt, **kwargs)
    upsert = re.sub(r"INSERT INTO", "INSERT OR REPLACE INTO", insert) # This is not upsert, but insert or replace. Might lead to loss of data in case of ON CASCADE DELETE
    return upsert


def upsertBulkInMemory(df: pd.DataFrame, engine: Engine, tt: Table, logger:logging.Logger, chunksize: int = 5000):
    """uses the bulk upserter in memory to upsert a dataframe to a table"""

    engine.dispose()

    myUpsert = None
    if engine.dialect.name.lower() in ['mysql', 'mariadb']:
        myUpsert = UpsertMySQL
    elif engine.dialect.name.lower() in ['sqlite']:
        myUpsert = UpsertSQLite

    # only take columns to copy if they exist both in the dataframe and in the table
    cols_table = [c.name for c in tt.c]
    cols_to_copy = list(set(df.columns).intersection(set(cols_table)))

    # remove an autoincrement primary key if that exists
    pk = tt.primary_key
    auto = None
    if (len(pk.columns) == 1 and
            isinstance(pk.columns.values()[0].type, sa.Integer) and
            pk.columns.values()[0].autoincrement):
        auto = pk.columns.keys()[0]
        if auto in cols_to_copy:
            cols_to_copy.remove(auto)
            pass
        pass

    df_to_upsert = df[cols_to_copy]


    # upsert this dataframe in chunks
    for k, g in df_to_upsert.groupby(np.arange(len(df_to_upsert)) // chunksize):
        # create the dictionary for upsert
        logger.log(logging.INFO,f"upserting chunk {k} into {tt.name}: {g.time.iloc[0]}, length: {len(g)}")
        d = g.to_dict(orient='records')
        engine.execute(myUpsert(tt,d))

def upsertBulkWithTmpTable(df: pd.DataFrame, engine: Engine, tt: Table,
                           logger:logging.Logger = logging.RootLogger(level=logging.DEBUG),
                           chunksize: int = 5000):
    """uses a temporary table to upsert a dataframe to a table"""

    engine.dispose()

    tn = tt.name
    tn_tmp = f"{tn}_tmp"

    df.to_sql(f"{tn_tmp}", con=engine, if_exists='replace', index=False, chunksize=chunksize)

    # pk is the primary key
    pk = tt.primary_key
    all_columns = [c.name for c in tt.c]
    columns_to_copy = list(
        all_columns)  # now columns_to_copy and all_columns are not the same object; I can modify the first without modifying the second

    if (len(pk.columns) == 1 and
            isinstance(pk.columns.values()[0].type, sa.Integer) and
            pk.columns.values()[0].autoincrement):
        pk_name_if_single_column_and_autoincrement = pk.columns.keys()[0]
        if pk_name_if_single_column_and_autoincrement in all_columns:
            columns_to_copy.remove(pk_name_if_single_column_and_autoincrement)

    # columns_to_copy now contains all columns except a single PK column if that is autoincremented
    # however, we might pass a dataframe that does not have all columns in the target table
    # in this case, we want to choose the intersection of column names in the target table and in the dataframe
    columns_to_copy = list(set(columns_to_copy).intersection(set(df.columns)))

    if engine.dialect.name.lower() in ['mysql', 'mariadb']:
        # case mysql: use on duplicate key
        # a) copy the candidate rows to a temp table

        strSQL: str = "INSERT INTO {0}(`{1}`) ".format(tn, "`, `".join(columns_to_copy))
        strSQL = strSQL + "\n SELECT "
        strList: list = [f"`t`.`{c}` as `{c}`" for c in columns_to_copy]
        strSQL = strSQL + ", ".join(strList)
        strSQL = strSQL + f"\n FROM `{tn_tmp}` as `t` "
        strSQL = strSQL + f"\n ON DUPLICATE KEY UPDATE"
        strList = [f"`{c}`=`t`.`{c}`" for c in columns_to_copy]
        strSQL = strSQL + ", ".join(strList)
        engine.execute(strSQL)

    elif engine.dialect.name.lower() in ['sqlite']:
        # case sqlite: use insert or replace
        strSQL: str = "INSERT OR REPLACE INTO {0}(`{1}`) ".format(tn, "`, `".join(columns_to_copy))
        strSQL = strSQL + "\n SELECT "
        strList: list = [f"`t`.`{c}` as `{c}`" for c in columns_to_copy]
        strSQL = strSQL + ", ".join(strList)
        strSQL = strSQL + f"\n FROM `{tn_tmp}` as `t` "
        engine.execute(strSQL)

        pass

    strSQL = f"DROP TABLE IF EXISTS {tn_tmp}"
    engine.execute(strSQL)



def upsert_dataframe_to_table(
        df: pd.DataFrame,
        engine: Engine,
        tt: Table,
        logger:logging.Logger = logging.RootLogger(level=logging.DEBUG),
        upsert_method: enum_upsert_method=enum_upsert_method.AUTO,
        n_rows_cutoff_for_memory: int=5000,
        chunksize: int = 5000):
    """upsert a dataframe to a table"""
    myUpsert = None

    assert(isinstance(upsert_method,enum_upsert_method))

    if upsert_method==enum_upsert_method.DO_NOT_UPSERT:
        return
    elif upsert_method==enum_upsert_method.MEMORY:
        myUpsert = upsertBulkInMemory
    elif upsert_method==enum_upsert_method.TMPTABLE:
        myUpsert = upsertBulkWithTmpTable
    elif upsert_method==enum_upsert_method.AUTO:
        N = len(df)
        if N <= n_rows_cutoff_for_memory:
            myUpsert = upsertBulkInMemory
        else:
            myUpsert = upsertBulkWithTmpTable
    else:
        return

    myUpsert(df=df,engine=engine,tt=tt,logger=logger,chunksize=chunksize)
