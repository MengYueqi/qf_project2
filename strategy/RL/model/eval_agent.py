import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

def _sharpe_like(pnl_series: np.ndarray, eps: float = 1e-8) -> float:
    if pnl_series.size == 0:
        return 0.0
    mu = pnl_series.mean()
    sd = pnl_series.std() + eps
    return (mu / sd) * np.sqrt(TRADING_DAYS_PER_YEAR)

def _max_drawdown_from_equity(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    running_max = -np.inf
    drawdowns = []
    for v in equity:
        running_max = max(running_max, v)
        drawdowns.append(running_max - v)
    return float(max(drawdowns) if drawdowns else 0.0)

def _load_benchmark_series(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    print(df.head())
    s = (
        df.assign(Date=pd.to_datetime(df["Date"]).dt.date.astype(str))
          .sort_values("Date")
          .set_index("Date")["Price"]
          .astype(float)
    )
    if s.empty:
        raise ValueError(f"{csv_path}: empty benchmark series")
    return s  # index=Date(str), values=Price(float)

def _align_benchmark_by_indices(bench: pd.Series, indices: list[int], initial_capital: float):
    """
    用 t_index（位置索引）对齐基准：
    - 取 bench.iloc[idx] 的价格序列
    - 用首个价格为基准构造 buy-and-hold 累计PnL 与 逐步PnL
    还返回映射出来的日期列表（来自 bench.index）
    """
    # 边界检查
    n = len(bench)
    bad = [i for i in indices if (i < 0 or i >= n)]
    if bad:
        raise IndexError(f"Some t_index out of range: {bad[:5]} ... (bench length={n})")

    # 取价格与对应日期
    prices = bench.iloc[indices].values
    mapped_dates = bench.index.take(indices).tolist()  # 日期字符串列表

    p0 = float(prices[0])
    if p0 <= 0:
        raise ValueError("Benchmark first price must be > 0")
    cum_bh = initial_capital * (prices / p0 - 1.0)
    pnl_bh = np.diff(np.r_[0.0, cum_bh])
    return cum_bh, pnl_bh, mapped_dates

def evaluate_agent_once(
    agent,
    env,
    max_steps: int = 1000,
    benchmark_csvs: list[str] | None = None,
    benchmark_names: list[str] | None = None,
    initial_capital: float = 1.0,
    date_key: str = "date",       # 若 env 返回真实日期可用；本实现优先用 t_index
    index_key: str = "t_index",   # 用 t_index 对齐
):
    obs_np = env.reset()
    import torch
    obs = torch.tensor(obs_np, dtype=torch.float32, device=agent.device)

    equity_curve = [0.0]
    pnl_history = []

    # 收集两种“时间轴”信息：优先使用 t_index；若没有再用 date
    t_indices_raw: list[int] = []
    dates_list: list[str] = []

    for _ in range(max_steps):
        action_np, _ = agent.choose_action_deterministic(obs)
        next_obs_np, reward, done, info = env.step(action_np)

        step_pnl = float(info["pnl"])
        pnl_history.append(step_pnl)
        equity_curve.append(equity_curve[-1] + step_pnl)

        # 优先收集 t_index（注意 off-by-one：pnl 对应的是 t_index-1）
        if isinstance(info, dict) and (index_key in info):
            idx_after = int(info[index_key])       # curr_idx after increment
            idx_for_pnl = max(0, idx_after - 1)    # 对应刚产生的这一步 pnl
            t_indices_raw.append(idx_for_pnl)
        elif isinstance(info, dict) and (date_key in info):
            dates_list.append(str(info[date_key]))

        obs = torch.tensor(next_obs_np, dtype=torch.float32, device=agent.device)
        if done:
            break

    strategy_equity_curve = np.array(equity_curve[1:])
    strategy_pnl_history = np.array(pnl_history)

    # ============ 指标（策略） ============
    total_return = float(strategy_equity_curve[-1]) if strategy_equity_curve.size else 0.0
    sharpe_like = _sharpe_like(strategy_pnl_history)
    max_dd = _max_drawdown_from_equity(strategy_equity_curve)

    # ============ 基准对齐（支持多个CSV） ============
    bh_rows = []
    if benchmark_csvs:
        for i, csv in enumerate(benchmark_csvs):
            bench = _load_benchmark_series(csv)

            if len(t_indices_raw) == len(strategy_pnl_history) and len(t_indices_raw) > 0:
                # 用 t_index 对齐（优先）
                bh_eq, bh_pnl, mapped_dates = _align_benchmark_by_indices(
                    bench, t_indices_raw, initial_capital
                )
                # 仅首次记录日期（给你调试用）
                if i == 0 and not dates_list:
                    dates_list = mapped_dates
            elif len(dates_list) == len(strategy_pnl_history) and len(dates_list) > 0:
                # 退路：有真实日期再按日期对齐
                # 用 reindex + ffill，确保完全对齐
                s = bench.reindex(dates_list).ffill()
                prices = s.values
                p0 = float(prices[0])
                bh_eq = initial_capital * (prices / p0 - 1.0)
                bh_pnl = np.diff(np.r_[0.0, bh_eq])
            else:
                # 两者都没有，就无法对齐
                print(f"[WARN] Cannot align benchmark for {csv} (no t_index and no dates). Skipped.")
                continue

            bh_total = float(bh_eq[-1]) if bh_eq.size else 0.0
            bh_sharpe = _sharpe_like(bh_pnl)
            bh_mdd = _max_drawdown_from_equity(bh_eq)

            name = (benchmark_names[i] if (benchmark_names and i < len(benchmark_names))
                    else csv)
            bh_rows.append((name, bh_total, bh_sharpe, bh_mdd))

    # ============ 打印结果 ============
    if bh_rows:
        print("=== EVAL RESULT (Strategy vs. Benchmarks) ===")
        print(f"Steps traded: {len(strategy_pnl_history)}")
        print(f"[Strategy]   CumPnL={total_return:.6f} | Sharpe*={sharpe_like:.3f} | MaxDD={max_dd:.6f}")
        for name, total, shp, mdd in bh_rows:
            print(f"[Buy&Hold] {name:>12} | CumPnL={total:.6f} | Sharpe*={shp:.3f} | MaxDD={mdd:.6f}")
    else:
        print("=== EVAL RESULT (Strategy only) ===")
        print(f"Steps traded   : {len(strategy_pnl_history)}")
        print(f"Cumulative PnL : {total_return:.6f}")
        print(f"Sharpe-like    : {sharpe_like:.3f}")
        print(f"Max Drawdown   : {max_dd:.6f}")

    return strategy_equity_curve, strategy_pnl_history, dates_list
