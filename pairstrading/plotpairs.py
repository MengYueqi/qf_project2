import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import pandas as pd

def plot_strategy_vs_benchmark(res_strategy, res_bh, title="Strategy vs Buy&Hold (Neutral)"):
    """
    Plot cumulative log returns of the pairs trading strategy against
    a static long-short (beta-neutral) buy-and-hold benchmark.

    Parameters
    ----------
    res_strategy : pd.DataFrame
        Backtest result of the strategy. Must contain a 'cumret' column.
        'cumret' represents cumulative log return.
    res_bh : pd.DataFrame
        Benchmark return series with a 'cumret' column,
        representing cumulative log return of the static long-short portfolio.
    title : str, optional
        Plot title.

    Notes
    -----
    This comparison evaluates the timing advantage of the pairs trading
    strategy against a passive beta-neutral long/short position.
    """
    
    plt.figure(figsize=(10,5))
    idx = res_strategy.index.intersection(res_bh.index)
    plt.plot(idx, res_strategy.loc[idx, "cumret"], label="Pairs Trading")
    plt.plot(idx, res_bh.loc[idx, "cumret"], label="Buy&Hold Neutral", linestyle="--")
    plt.ylabel("Cumulative Log Return")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pair_performance(res, pair_row):

    s1, s2 = pair_row["stock1"], pair_row["stock2"]

    res = res.copy()
    if 'Date' in res.columns:
        res['Date'] = pd.to_datetime(res['Date'], errors='coerce')
        res = res.set_index('Date')
    else:
        if not pd.api.types.is_datetime64_any_dtype(res.index):
            try:
                res.index = pd.to_datetime(res.index, errors='raise')
            except Exception:
                if pd.api.types.is_integer_dtype(res.index):
                    res.index = pd.to_datetime(res.index, unit='D', origin='unix')
                else:
                    raise ValueError("Index is not datetime-like and no 'Date' column found.")
    res.index.name = 'Date'

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(res.index, res["cumret"], color="royalblue", linewidth=2, label="Cumulative Return")
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Cumulative Log Return", fontsize=12, color="royalblue")
    ax1.tick_params(axis='y', labelcolor='royalblue')

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.title(f"Pair Trading Performance: {s1} vs {s2}", fontsize=14, weight='bold')
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.7)
    ax1.legend(loc="upper left")

    plt.tight_layout()
    plt.show()
