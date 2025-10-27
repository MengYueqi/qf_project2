import numpy as np
import torch

def evaluate_agent_once(agent, env, max_steps=1000):
    """
    不训练，只让当前 agent 在 env 上交易一段时间，
    返回资金曲线和一些绩效指标。
    假设 env 是和训练同样的 RealTradingEnv，但我们手动 reset 一次固定起点。
    """

    obs_np = env.reset()
    obs = torch.tensor(obs_np, dtype=torch.float32, device=agent.device)

    equity_curve = [0.0]   # 从0开始累加PnL
    pnl_history = []

    for t in range(max_steps):
        # 用当前策略下单（不做探索性rsample，而是用均值更稳也可以，我们先继续用 choose_action() 的采样版也行）
        action_np, value = agent.choose_action_deterministic(obs)

        next_obs_np, reward, done, info = env.step(action_np)

        # 关键：用真实pnl来衡量表现，而不是reward（reward包含alpha放大、交易成本缩放）
        step_pnl = info["pnl"]
        pnl_history.append(step_pnl)

        equity_curve.append(equity_curve[-1] + step_pnl)

        obs = torch.tensor(next_obs_np, dtype=torch.float32, device=agent.device)

        if done:
            break

    equity_curve = np.array(equity_curve[1:])  # drop the initial 0
    pnl_history = np.array(pnl_history)

    total_return = equity_curve[-1]
    avg_pnl = pnl_history.mean()
    vol_pnl = pnl_history.std() + 1e-8
    sharpe_like = (avg_pnl / vol_pnl) * np.sqrt(252)  # 假设一步=1交易日

    max_dd = 0.0
    peak = -1e9
    running_max = -1e9
    running_max = -np.inf
    drawdowns = []
    running_max = -np.inf
    for v in equity_curve:
        if v > running_max:
            running_max = v
        dd = running_max - v
        drawdowns.append(dd)
    max_dd = max(drawdowns) if drawdowns else 0.0

    print("=== EVAL RESULT ===")
    print(f"Steps traded: {len(pnl_history)}")
    print(f"Cumulative PnL: {total_return:.6f}")
    print(f"Mean daily pnl: {avg_pnl:.6f}")
    print(f"Vol daily pnl : {vol_pnl:.6f}")
    print(f"Sharpe-like   : {sharpe_like:.3f}")
    print(f"Max Drawdown  : {max_dd:.6f}")

    return equity_curve, pnl_history
