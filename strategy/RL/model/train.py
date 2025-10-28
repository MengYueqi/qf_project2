import numpy as np
import torch

from ppo_agent import PPOAgent, RolloutBuffer
from data_loader import load_real_trading_env  # <= 新增：真实环境加载器
from eval_agent import evaluate_agent_once  # <= 新增：评估函数
from real_env import RealTradingEnv


def run_training_loop(
    total_iterations=10,
    steps_per_rollout=512,
    ppo_epochs=5,
    batch_size=64,
    device="mps",
    lr=5e-5,
):
    """
    total_iterations:   外层训练循环的轮数
    steps_per_rollout:  每一轮从环境里收集多少 step 的数据
    ppo_epochs:         每一轮用这批数据更新几次 PPO
    batch_size:         PPO 的 minibatch 大小
    device:             "cpu" 或 "cuda"
    """

    # 0. 先把真实数据环境准备好
    # 你可以随时改 tickers 顺序 / 选择哪5只股票
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    

    env, features, rets, dates = load_real_trading_env(
        base_path="strategy/RL/data",  # 根据你运行脚本的位置，可能要改成绝对路径
        tickers=tickers,
        obs_fillna=0.0,
        cost_coeff=0.001,
        alpha=300.0,
        leverage_cap=1.0,
        max_episode_steps=200,
        random_start=True,
    )

    # 根据真实数据推导维度
    obs_dim = features.shape[1]      # 全部拼好的特征列数
    action_dim = len(tickers)        # 我们持仓的股票数 = 动作维度

    # 1. Agent
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        lr=lr,
    )

    for iteration in range(1, total_iterations + 1):
        # 2. rollout buffer
        buffer = RolloutBuffer(
            buffer_size=steps_per_rollout,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )

        # 重置环境
        obs_np = env.reset()  # np.array(float32, shape=(obs_dim,))
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device)

        ep_reward_sum = 0.0
        ep_step_count = 0

        # 3. rollout 收集 steps_per_rollout 步
        for step in range(steps_per_rollout):
            # 策略 → 动作
            action_np, logp, value_pred = agent.choose_action(obs)
            # action_np: np.array shape=(action_dim,)
            # logp: float or tensor scalar
            # value_pred: float (V(s))

            # 环境推进
            next_obs_np, reward, done, info = env.step(action_np)

            # 写入 buffer
            buffer.add(
                obs=obs.cpu(),  # buffer里会把这些拼起来
                action=torch.tensor(action_np, dtype=torch.float32),
                log_prob=torch.tensor(logp, dtype=torch.float32),
                reward=torch.tensor(reward, dtype=torch.float32),
                done=torch.tensor(float(done), dtype=torch.float32),
                value=torch.tensor(value_pred, dtype=torch.float32),
            )

            ep_reward_sum += reward
            ep_step_count += 1

            # 准备下一步
            obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)

            if done:
                # episode结束（比如走完max_episode_steps或走到区间尾部）
                # 我们开一条新的episode继续填buffer，直到装满steps_per_rollout步
                obs_np = env.reset()
                obs = torch.tensor(obs_np, dtype=torch.float32, device=device)

        # 4. 计算GAE, returns，用于 PPO 的目标
        with torch.no_grad():
            # 用 Critic 估计 rollout 最后一个状态的 V(s_T)，给 GAE bootstrap
            last_v = agent.critic(
                obs.unsqueeze(0).to(device)
            ).squeeze(-1)[0].detach()

        buffer.compute_returns_and_advantages(
            last_value=last_v,
            gamma=agent.gamma,
            gae_lambda=agent.gae_lambda,
        )

        avg_step_reward = ep_reward_sum / max(ep_step_count, 1)

        # 5. PPO更新
        stats = agent.update(
            rollout_buffer=buffer,
            epochs=ppo_epochs,
            batch_size=batch_size,
        )

        # 6. 打印训练指标
        print(
            f"[Iter {iteration:02d}] "
            f"avg_step_reward={avg_step_reward:8.4f}  "
            f"policy_loss={stats['policy_loss']:.4f}  "
            f"value_loss={stats['value_loss']:.4f}  "
            f"entropy={stats['entropy']:.4f}  "
            f"ret_mean={stats['ret_mean']:.4f}"
        )

     # 训练结束后评估
    eval_env, si, ei = make_eval_env(features, rets, lookback_days=250)
    evaluate_agent_once(agent, eval_env, max_steps=250)

def make_eval_env(features, rets, lookback_days=250):
    # 用最后 lookback_days 天窗口来评估
    T = len(features)
    start_index = max(0, T - lookback_days - 1)  # -1 给一步forward room
    end_index   = T - 2                          # -2 因为 env 里会用 idx+1

    eval_env = RealTradingEnv(
        features=features,
        rets=rets,
        cost_coeff=0.001,
        alpha=300.0,
        leverage_cap=1.0,
        start_index=start_index,
        end_index=end_index,
        max_episode_steps=lookback_days,
        random_start=False,   # <--- 评估时关键：不要随机起点！
    )

    return eval_env, start_index, end_index



if __name__ == "__main__":
    # 固定随机种子，增加可复现性
    torch.manual_seed(0)
    np.random.seed(0)

    run_training_loop(
        total_iterations=1000,   # PPO 外层循环
        steps_per_rollout=512,  # 每次收集多少步
        ppo_epochs=10,           # 用这批数据迭代几次
        batch_size=64,
        device="cpu",
    )