import os
import pandas as pd
import numpy as np
from real_env import RealTradingEnv

def load_real_trading_env(
    base_path="strategy/RL/data",
    tickers=None,
    obs_fillna=0.0,
    cost_coeff=0.001,
    alpha=100.0,
    leverage_cap=2.0,
    max_episode_steps=200,
    random_start=True,
    feature_dim=25,       # 想保留的总特征维度
    random_feature_select=True,  # 是否随机选因子列
    seed=42,                # 固定随机种子方便复现
):
    if tickers is None:
        tickers = ["AAPL","MSFT","GOOGL","AMZN","NVDA"]

    np.random.seed(seed)

    # -------- 1. 价格 --------
    price_dfs = []
    for t in tickers:
        fp = os.path.join(base_path, "close_price", f"{t}.csv")
        df = pd.read_csv(fp, parse_dates=["Date"])
        df = df[["Date", df.columns[1]]].rename(columns={df.columns[1]: t})
        price_dfs.append(df)

    price_df = price_dfs[0]
    for df in price_dfs[1:]:
        price_df = pd.merge(price_df, df, on="Date", how="outer")

    price_df = price_df.sort_values("Date").reset_index(drop=True)
    price_df = price_df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    # -------- 2. 收益率 --------
    rets_df = price_df.copy()
    rets_df[tickers] = price_df[tickers].pct_change().fillna(0.0)

    # -------- 3. 因子特征 --------
    factor_dfs = []
    for t in tickers:
        fp = os.path.join(base_path, "factors", f"{t}.csv")
        df = pd.read_csv(fp, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        df = df.fillna(obs_fillna)

        factor_cols = [c for c in df.columns if c != "Date"]
        rename_map = {c: f"{t}_{c}" for c in factor_cols}
        df = df.rename(columns=rename_map)
        factor_dfs.append(df)

    factors_df = factor_dfs[0]
    for df in factor_dfs[1:]:
        factors_df = pd.merge(factors_df, df, on="Date", how="outer")

    full_df = pd.merge(factors_df, price_df[["Date"]], on="Date", how="inner")
    full_df = full_df.sort_values("Date").reset_index(drop=True)
    full_df = full_df.fillna(obs_fillna)

    # -------- 4. 对齐 --------
    features_raw = full_df.drop(columns=["Date"]).iloc[:-1].to_numpy(dtype=np.float32)
    rets = rets_df[tickers].iloc[1:].to_numpy(dtype=np.float32)
    next_prices = price_df[tickers].iloc[1:].to_numpy(dtype=np.float32)

    aligned_dates = full_df["Date"].iloc[:-1].reset_index(drop=True)

    assert features_raw.shape[0] == rets.shape[0], "特征和收益行数必须对齐"

    # 拼接 next_prices 在最前面
    features_full = np.concatenate([next_prices, features_raw], axis=1)

    # -------- 5. 特征选择逻辑 --------
    if feature_dim is not None and feature_dim < features_full.shape[1]:
        keep_base = 5  # 前五列是预测股价
        num_extra = feature_dim - keep_base

        # 从剩余列（第6列之后）随机选 num_extra 个
        all_indices = np.arange(keep_base, features_full.shape[1])
        if random_feature_select:
            chosen_extra = np.random.choice(all_indices, size=num_extra, replace=False)
        else:
            chosen_extra = all_indices[:num_extra]  # 不随机

        selected_indices = np.concatenate([np.arange(keep_base), np.sort(chosen_extra)])
        features = features_full[:, selected_indices]
    else:
        features = features_full

    # -------- 6. 创建环境 --------
    env = RealTradingEnv(
        features=features,
        rets=rets,
        cost_coeff=cost_coeff,
        alpha=alpha,
        leverage_cap=leverage_cap,
        start_index=0,
        end_index=len(features) - 2,
        max_episode_steps=max_episode_steps,
        random_start=random_start,
    )

    return env, features, rets, aligned_dates


if __name__ == "__main__":
    env, features, rets, dates = load_real_trading_env(feature_dim=30)
    print("✅ RealTradingEnv loaded successfully.")
    print("features shape:", features.shape)
    print("前5列（价格）+ 随机选列:", features[0, :10])
