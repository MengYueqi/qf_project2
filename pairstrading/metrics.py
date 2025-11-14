import numpy as np


def compute_pair_metrics(res):
    """
    Compute key performance metrics for a pairs trading backtest.

    Parameters
    ----------
    res : pd.DataFrame
        DataFrame containing at least two columns:
        - 'strategy_ret' : daily log returns of the strategy
        - 'cumret'       : cumulative log returns

    Returns
    -------
    dict
        {
            "CumulativeReturn": total geometric return,
            "AnnualizedReturn": annualized geometric return,
            "AnnualVolatility": annualized standard deviation of daily returns,
            "SharpeRatio": annualized Sharpe ratio (risk-free rate assumed 0),
            "MaxDrawdown": minimum drawdown (in decimal form),
            "WinRate": fraction of days with positive returns,
            "Days": number of trading days
        }

    Notes
    -----
    - All returns are computed in log space for consistency.
    - The Sharpe ratio uses a 252-day convention.
    - Max drawdown is derived from the equity curve (exp(cumret)).
    """
    daily_ret = res["strategy_ret"].dropna()
    n_days = len(daily_ret)

    if n_days == 0:
        return {
            "CumulativeReturn": np.nan,
            "AnnualizedReturn": np.nan,
            "AnnualVolatility": np.nan,
            "SharpeRatio": np.nan,
            "MaxDrawdown": np.nan,
            "WinRate": np.nan,
            "Days": 0,
        }

    final_cumlog = res["cumret"].iloc[-1]

    total_return = np.exp(final_cumlog) - 1

    ann_return = (1 + total_return) ** (252 / n_days) - 1

    daily_std = daily_ret.std()
    ann_vol = daily_std * np.sqrt(252)


    if daily_std == 0:
        sharpe = np.nan
    else:
        sharpe = daily_ret.mean() / daily_std * np.sqrt(252)

    equity = np.exp(res["cumret"])            
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()

   
    win_rate = (daily_ret > 0).mean()

    return {
        "CumulativeReturn": total_return,   
        "AnnualizedReturn": ann_return,
        "AnnualVolatility": ann_vol,
        "SharpeRatio": sharpe,
        "MaxDrawdown": max_dd,
        "WinRate": win_rate,
        "Days": n_days,
    }