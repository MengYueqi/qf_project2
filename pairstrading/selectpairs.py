from itertools import combinations
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

def build_pair_universe(close_df: pd.DataFrame,
                        train_end: str = "2023-01-01",
                        corr_threshold: float = 0.8,
                        min_len: int = 200,
                        pval_threshold: float = 0.05,
                        top_n: int | None = None) -> pd.DataFrame:
    """
    Construct a pairs universe from historical close prices using:
      1) correlation pre-filter on log-prices
      2) Engle-Granger cointegration test
      3) OLS y = alpha + beta * x to estimate hedge ratio and spread stats

    Parameters
    ----------
    close_df : pd.DataFrame
        Wide table with columns ['Date', <ticker1>, <ticker2>, ...].
    train_end : str
        Last date (exclusive) for training window, e.g. "2023-01-01".
    corr_threshold : float
        Minimum pairwise correlation on log-prices to pass the pre-filter.
    min_len : int
        Minimum overlapping length for a pair to be considered.
    pval_threshold : float
        Cointegration p-value threshold to accept a pair.
    top_n : int or None
        If set, return the top-N pairs by p-value.

    Returns
    -------
    pair_df : pd.DataFrame
        Columns: ['stock1','stock2','pvalue','alpha','beta','mu','sigma','n_obs']
        Sorted by 'pvalue' ascending.
    """
    df = close_df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df['Date'] < pd.to_datetime(train_end)].set_index('Date')
    elif not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("Provide a 'Date' column or a DatetimeIndex.")

    price_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(price_cols) < 2:
        raise ValueError("Need at least two numeric price columns.")
    train_prices = df[price_cols].dropna(how="all")
    log_prices = np.log(train_prices)

    corr_matrix = log_prices.corr()
    candidates = [(i, j) for i, j in combinations(log_prices.columns, 2)
                  if corr_matrix.loc[i, j] > corr_threshold]

    pairs = []
    for s1, s2 in candidates:
        pair_df = log_prices[[s1, s2]].dropna()
        if len(pair_df) < min_len:
            continue

        # Engle–Granger
        score, pval, _ = coint(pair_df[s1], pair_df[s2])
        if pval >= pval_threshold:
            continue

        # OLS: y = alpha + beta * x  (x=log p_s2, y=log p_s1)
        x = pair_df[s2].values
        y = pair_df[s1].values
        X = np.column_stack([np.ones(len(x)), x])
        alpha_hat, beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]

        spread = pair_df[s1] - beta_hat * pair_df[s2]
        mu = float(spread.mean())
        sigma = float(spread.std(ddof=1)) 

        pairs.append((s1, s2, float(pval), float(alpha_hat), float(beta_hat), mu, sigma, int(len(pair_df))))

    if not pairs:
        return pd.DataFrame(columns=['stock1','stock2','pvalue','alpha','beta','mu','sigma','n_obs'])

    pair_df_out = pd.DataFrame(pairs, columns=['stock1','stock2','pvalue','alpha','beta','mu','sigma','n_obs'])
    pair_df_out = pair_df_out.sort_values('pvalue', ascending=True).reset_index(drop=True)

    if top_n is not None and top_n > 0:
        pair_df_out = pair_df_out.head(top_n).reset_index(drop=True)

    return pair_df_out

