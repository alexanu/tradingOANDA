import pandas as pd
from timeit import default_timer as timer
import sys
import numpy as np

def createIntervals(min_time_interval: pd.Timedelta = pd.Timedelta(seconds=5)) -> pd.DatetimeIndex:
    # create a list of integers representing the distance in time_interval to now
    N = 10


    data = [
        ['5 seconds', '5 minutes'],
        ['20 seconds', '30 minutes'],
        ['1 minutes', '2 hours'],
        ['5 minutes', '8 hours'],
        ['20 minutes', '1 days'],
        ['1 hours', '2 days'],
        ['8 hours', f"{7 * 4 * 1} days"],
        ['1 days', f"{7 * 4 * 3} days"],
        ['7 days', f"{7 * 4 * 13 * 3} days"],
    ]

    df = pd.DataFrame(data)
    df.columns = ['duration_step', 'start_next_interval']
    df['duration_step_pd'] = None
    for index, row in df.iterrows():
        duration_step_pd = pd.to_timedelta(row['duration_step'])
        df.at[index, 'duration_step_pd'] = duration_step_pd

    df = df[df['duration_step_pd']>=min_time_interval].copy()

    df['duration_interval'] = None
    df['periods'] = None
    df['start_pd'] = None
    df['stop_pd'] = None
    df['N'] = None

    start_pd = pd.Timedelta(seconds=0)
    for index, row in df.iterrows():
        duration_step_pd = df.at[index, 'duration_step_pd']
        df.at[index, 'start_pd'] = start_pd
        start_next_interval_pd = pd.to_timedelta(row['start_next_interval'])
        df.at[index, 'stop_pd'] = start_next_interval_pd
        n_periods_interval = (start_next_interval_pd - start_pd) / duration_step_pd
        df.at[index, 'periods'] = n_periods_interval
        df.at[index, 'duration_interval'] = n_periods_interval * duration_step_pd
        start_pd = start_next_interval_pd
        pass

    df['N'] = np.cumsum(df['periods'])

    return df

def create_time_range(df: pd.DataFrame, start_time:pd.datetime):
    datetimeSeries = []

    starts = start_time + df['start_pd']
    freqs = df['duration_step_pd']
    stops = start_time + df['stop_pd']

    for i, [start, freq, stop] in enumerate(zip(starts, freqs, stops)):
        a = pd.date_range(start=start,end=stop-freq,freq=freq)
        datetimeSeries.append(a.to_series())
        pass


    a: pd.Series = pd.concat(datetimeSeries)
    a.sort_values(inplace=True)

    return a

def main(argv):
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 300)

    start_time = pd.to_datetime("2019-01-01")
    time_interval = pd.Timedelta(seconds=5)

    t1 = timer()
    df = createIntervals(min_time_interval=time_interval)
    t2 = timer()
    print(f"create intervals; len: {len(df)}; dt: {t2-t1}")
    print(df)

    t1 = timer()
    time_range = create_time_range(df=df,start_time=start_time)
    t2 = timer()
    print(f"create time range; len: {len(time_range)}; dt: {t2 - t1}")


if __name__ == "__main__":
    sys.exit(main(sys.argv))