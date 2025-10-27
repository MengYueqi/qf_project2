import numpy as np
import torch

# 从你的已有文件导入
from ppo_agent import PPOAgent, RolloutBuffer


class DummyTradingEnv:
    """
    一个极简的模拟多资产交易环境:
    - obs: 随机因子向量 (类似多股票+市场特征拼一起)
    - action: 例如长度=5，代表5只股票的目标仓位 [-1,1]
    - reward: 组合收益 - 换仓成本 (再乘一个放大系数)
    """

    def __init__(
        self,
        obs_dim=40,
        action_dim=5,
        cost_coeff=0.001,
        alpha=100.0,
        max_steps=200,
        leverage_cap=2.0,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cost_coeff = cost_coeff
        self.alpha = alpha
        self.max_steps = max_steps
        self.leverage_cap = leverage_cap

        self.prev_w = np.zeros(action_dim, dtype=np.float32)
        self.step_count = 0
        self.state = self._rand_state()

    def _rand_state(self):
        # 模拟“市场状态”的特征向量；这里用N(0,1)
        return np.random.normal(0, 1, size=self.obs_dim).astype(np.float32)

    def reset(self):
        self.prev_w = np.zeros(self.action_dim, dtype=np.float32)
        self.step_count = 0
        self.state = self._rand_state()
        return self.state

    def step(self, action):
        """
        action: np.array(shape=(action_dim,), range大致在[-1,1])
        返回:
            next_obs (np.float32[obs_dim])
            reward (float)
            done (bool)
            info (dict)
        """

        # 1. 杠杆上限，控制总绝对仓位 ∑|w|
        gross_leverage = np.abs(action).sum()
        if gross_leverage > self.leverage_cap:
            scale = self.leverage_cap / (gross_leverage + 1e-8)
            action = action * scale
        print(action)

        # 2. 模拟明日真实收益率 (比如随机N(0,1%) )
        true_ret = np.random.normal(0, 0.01, size=self.action_dim).astype(np.float32)

        # 3. 组合收益 pnl = w · r
        pnl = float((action * true_ret).sum())

        # 4. 调仓成本 ~ 成交冲击/滑点
        trade_cost = self.cost_coeff * float(np.abs(action - self.prev_w).sum())

        # 5. 奖励，放大到更适合RL训练的尺度
        reward = self.alpha * (pnl - trade_cost)

        # 6. 状态推进
        self.prev_w = action.copy()
        self.state = self._rand_state()

        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {
            "pnl": pnl,
            "cost": trade_cost,
            "gross_leverage": float(np.abs(action).sum()),
        }

        return self.state, reward, done, info


def run_training_loop(
    total_iterations=10,
    steps_per_rollout=512,
    ppo_epochs=5,
    batch_size=64,
    obs_dim=40,
    action_dim=5,
    device="cpu",
):
    """
    total_iterations:   外层训练循环的轮数
    steps_per_rollout:  每一轮从环境里收集多少 step 的数据
    ppo_epochs:         每一轮用这批数据更新几次 PPO
    batch_size:         PPO 的 minibatch 大小
    obs_dim:            状态向量维度（模拟 factor+仓位信息拼接）
    action_dim:         动作维度（股票数量 = 我们要分配仓位的数量）
    """

    # 1. 环境和 Agent
    env = DummyTradingEnv(obs_dim=obs_dim, action_dim=action_dim)
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )

    for iteration in range(1, total_iterations + 1):
        # 2. 新建 rollout buffer
        buffer = RolloutBuffer(
            buffer_size=steps_per_rollout,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )

        # 重置环境，开始 rollout
        obs = torch.tensor(env.reset(), dtype=torch.float32, device=device)

        ep_reward_sum = 0.0
        ep_step_count = 0

        # 3. 用当前策略 roll 出 steps_per_rollout 步数据
        for step in range(steps_per_rollout):
            # 策略给一个动作
            action_np, logp, value_pred = agent.choose_action(obs)

            # 环境前进一步
            next_obs_np, reward, done, info = env.step(action_np)

            # 存到 buffer
            buffer.add(
                obs=obs.cpu(),  # buffer 内部会自己再 to(device) 时处理
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
                # 如果一条episode结束了，就reset继续采，直到buffer装满
                obs = torch.tensor(env.reset(), dtype=torch.float32, device=device)

        # 4. GAE + returns 计算 (Critic target / Advantage)
        with torch.no_grad():
            # 取最后一个obs估一下 V(s_T)，用于bootstrap
            last_v = agent.model.forward(obs.unsqueeze(0).to(device))[2][0].detach()

        buffer.compute_returns_and_advantages(
            last_value=last_v,
            gamma=agent.gamma,
            gae_lambda=agent.gae_lambda,
        )

        avg_step_reward = ep_reward_sum / max(ep_step_count, 1)

        # 5. 调用 PPO 更新
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


if __name__ == "__main__":
    # 固定随机种子，让结果更可复现一些
    torch.manual_seed(0)
    np.random.seed(0)

    # 直接跑一小段训练看输出
    run_training_loop(
        total_iterations=10,      # 训练多少轮
        steps_per_rollout=512,    # 每一轮roll多少步
        ppo_epochs=5,             # 每一轮做几次PPO update
        batch_size=64,            # PPO minibatch大小
        obs_dim=40,               # 状态向量长度(模拟市场特征+仓位特征)
        action_dim=5,             # 股票数量/仓位维度
        device="cpu",             # 没GPU就用"cpu"
    )
