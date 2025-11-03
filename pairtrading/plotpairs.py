import matplotlib.pyplot as plt 
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