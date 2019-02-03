#!/usr/bin/env python
from enum import Enum
import pandas as pd

class enum_upsert_method(Enum):
    """enum for the choice of upload-method"""
    MEMORY = "memory"
    TMPTABLE = "tmpTable"
    DO_NOT_UPSERT = "do_not_upsert"
    AUTO = "auto"


class enum_extending_direction(Enum):
    """enum to determine in which direction we want to extend a dataframe with repetitions of the first/last row"""
    EARLIER_TIMES = "extending towards earlier times"
    MORE_RECENT_TIMES = "extending towards more recent times"


def tile_df(time_interval: pd.Timedelta,
            row_to_extend: pd.Series,
            time_start: pd.Timestamp,
            chunksize: int,
            ) -> pd.DataFrame:
    """construct a dataframe consisting of a repetition of rows

       We want to extend the rows of a dataframe both towards earlier and more recent times.
       If we want to extend the df towards earlier times than are present in the df,
       we construct a vector consisting of the needed (not yet present in the existing df) times.
       We then tile the rest of the columns
       we tile the provided row
       while extending the time column
        bfill (when extending towards earlier times) or
        ffill (when extending towards more recent times)

    :param time_interval:
    :type time_interval:
    :param row_to_extend: a pandas series representing the row that is going to be used as a template for tiling
    :type row_to_extend: pd.Series
    :param time_start:
    :type time_start:
    :param chunksize:
    :type chunksize:
    :return:
    :rtype:
    """


    times = pd.date_range(start=time_start, periods=chunksize, freq=time_interval)
    a = row_to_extend.values
    # tile the first row for a total of N rows
    b = pd.np.tile(a, (chunksize, 1))
    # create a dataframe with these rows and the columns of the original dataframe
    df2 = pd.DataFrame(data=b, columns=row_to_extend.index.values)
    df2['time'] = times

    return df2