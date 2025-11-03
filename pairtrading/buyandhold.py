import numpy as np
import pandas as pd

def buyhold_benchmark_neutral(actual_df, pair_row):
    """
    Construct a static long-short (beta-neutral) buy-and-hold benchmark
    for a given stock pair. This represents a passive portfolio that
    goes long one unit of stock1 and short 'beta' units of stock2 and
    holds the position throughout the entire test period without any
    rebalancing or timing.

    Parameters
    ----------
    actual_df : pd.DataFrame
        DataFrame containing ['Date', stock1, stock2] with actual close prices.
    pair_row : dict or pd.Series
        A row containing at least:
        - 'stock1' : first stock symbol (long leg)
        - 'stock2' : second stock symbol (short leg)
        - 'beta'   : hedge ratio between the two stocks.

    Returns
    -------
    pd.DataFrame
        Benchmark daily and cumulative log returns:
        - 'strategy_ret' : daily log return of the static long-short portfolio
        - 'cumret'       : cumulative log return over time

    Notes
    -----
    - This benchmark serves as a non-timing, passive reference for the
      pairs trading strategy.
    - It isolates the timing benefit of the active strategy by comparing
      against a portfolio with identical exposure (long stock1, short beta·stock2)
      but no trading decisions.
    """
    s1, s2, beta = pair_row["stock1"], pair_row["stock2"], float(pair_row["beta"])
    tmp = actual_df[["Date", s1, s2]].dropna().copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"])
    tmp = tmp.set_index("Date").sort_index()

    r1 = np.log(tmp[s1]).diff()
    r2 = np.log(tmp[s2]).diff()

    bh_ret = (r1 - beta * r2).fillna(0.0)
    out = pd.DataFrame({
        "strategy_ret": bh_ret,
        "cumret": bh_ret.cumsum()
    }, index=tmp.index)
    return out
