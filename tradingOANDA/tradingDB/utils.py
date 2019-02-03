import pandas as pd
from tradingOANDA.utils import enum_extending_direction, enum_upsert_method, tile_df
from tradingOANDA.tradingDB.database import TradingDB
from tradingOANDA.tradingDB.database import exception_table_does_not_exist
import operator
from timeit import default_timer as timer




def extend_towards_extremal_times(tradingDB: TradingDB,
                                  tn: str,
                                  direction: enum_extending_direction,
                                  extremal_time: pd.Timestamp,
                                  upsert_method: enum_upsert_method,
                                  n_rows_cutoff_for_memory: int,
                                  chunksize: int):
    try:
        to = tradingDB.get_market_data_orm_table(tn)
    except exception_table_does_not_exist as e:
        tradingDB.logger.error(e.message)
        raise e
        pass



    min_time_table_valid_rows, max_time_table_valid_rows = \
        tradingDB.get_min_max_time_single_instrument(tn,
                                                     choose_only_rows_with_volume_gt_0=True)

    if direction == enum_extending_direction.EARLIER_TIMES:
        strDirection = "earlier"
        time_interval = -tradingDB.timeInterval
        myComparisonOperator = operator.lt
        myAddingOperator = operator.sub
        myExtremalTime = min_time_table_valid_rows
    elif direction == enum_extending_direction.MORE_RECENT_TIMES:
        strDirection = "more recent"
        time_interval = +tradingDB.timeInterval
        myComparisonOperator = operator.gt
        myAddingOperator = operator.add
        myExtremalTime = max_time_table_valid_rows
        pass

    ssn = tradingDB.sessionmaker()
    q = ssn.query(to).filter(to.time == myExtremalTime)
    # print(lq(q))
    ssn.close()

    df = pd.read_sql(sql=q.statement, con=tradingDB.engine)
    row_to_extend = df.iloc[0].copy()
    row_to_extend['volume'] = 0
    row_to_extend['complete'] = None
    row_to_extend['age'] = None

    time_start = myAddingOperator(df['time'].iloc[0], tradingDB.timeInterval)
    while True:

        t1 = timer()
        df2 = tile_df(time_interval= time_interval,
                      row_to_extend=row_to_extend,
                      time_start=time_start,
                      chunksize=chunksize,
                      )
        t2 = timer()
        tradingDB.logger.info(f"extending towards {strDirection} times: {tn}; start_time: {time_start}; dt={t2 - t1}")

        df2.reset_index(inplace=True)
        rows_to_drop_index = df2[myComparisonOperator(df2['time'], extremal_time)].index
        df2.drop(rows_to_drop_index, inplace=True)

        t3 = timer()
        tradingDB.logger.info(
            f"drop rows that are too {strDirection} times: {tn}; start_time: {time_start}; dt={t3 - t2}")

        tradingDB.upsert_df_to_table(df=df2, tn=tn, upsert_method=upsert_method, logger=tradingDB.logger,
                                     n_rows_cutoff_for_memory=n_rows_cutoff_for_memory,
                                     chunksize=chunksize)
        t4 = timer()
        tradingDB.logger.info(f"upserting: {tn}; start_time: {time_start}; length: {len(df2)}; dt={t4 - t3}")

        if len(df2) > 0:
            time_start = myAddingOperator(df2['time'].iloc[-1], tradingDB.timeInterval)

            if myComparisonOperator(time_start, extremal_time):
                break
                pass
            pass
        else:
            break
