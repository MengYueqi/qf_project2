import numpy as np

class RealTradingEnv:
    """
    基于真实历史数据的多资产交易环境。

    features[t] : shape (T, obs_dim) 的观测特征 (float32)
    rets[t]     : shape (T, action_dim) 在 t→t+1 的真实收益 (float32)

    时间推进:
        - idx_t 时刻看到 obs = features[idx_t]
        - agent 下 action_t (仓位向量)
        - reward 基于 rets[idx_t]
        - next_obs = features[idx_t+1]
        - idx_t += 1
    """

    def __init__(
        self,
        features: np.ndarray,       # (T, obs_dim), float32
        rets: np.ndarray,           # (T, action_dim), float32
        cost_coeff=0.001,
        alpha=100.0,
        leverage_cap=1.0,
        start_index=0,
        end_index=None,
        max_episode_steps=None,
        random_start=False,
    ):
        assert len(features) == len(rets), "features 和 rets 长度必须一致"
        assert features.dtype == np.float32
        assert rets.dtype == np.float32

        self.features_full = features
        self.rets_full = rets

        self.cost_coeff = cost_coeff
        self.alpha = alpha
        self.leverage_cap = leverage_cap

        self.start_index = start_index
        # 我们需要 curr_idx+1 存在，所以 end_index 不能指到最后一行
        if end_index is None:
            end_index = len(features) - 2
        self.end_index = min(end_index, len(features) - 2)

        self.max_episode_steps = max_episode_steps
        self.random_start = random_start  # 是否每次reset随机起点

        # 运行状态
        self.curr_idx = None
        self.episode_steps = None
        self.prev_w = None  # 上一步仓位，用于交易成本

        self.obs_dim = features.shape[1]
        self.action_dim = rets.shape[1]

    def _sample_start_index(self):
        """
        如果 random_start=True，我们随机挑一个合法起点；
        否则就用 self.start_index。
        """
        if not self.random_start:
            return self.start_index

        # 为了保证一条episode跑得下去，留一点安全buffer
        hi = self.end_index - 200 if self.max_episode_steps else self.end_index
        hi = max(self.start_index, hi)
        return np.random.randint(self.start_index, hi + 1)

    def reset(self):
        # 选择起点
        self.curr_idx = self._sample_start_index()
        self.episode_steps = 0
        self.prev_w = np.zeros(self.action_dim, dtype=np.float32)

        obs = self.features_full[self.curr_idx]  # shape=(obs_dim,)
        return obs.astype(np.float32)

    def step(self, action: np.ndarray):
        """
        action: shape=(action_dim,), 目标仓位（可以是[-1,1]之类）
        返回: (next_obs, reward, done, info)
        """

        # 1. 杠杆限制
        gross_leverage = np.abs(action).sum()
        if gross_leverage > self.leverage_cap:
            scale = self.leverage_cap / (gross_leverage + 1e-8)
            action = action * scale

        # 2. 用真实收益算 pnl
        true_ret_vec = self.rets_full[self.curr_idx]  # (action_dim,)
        pnl = float((action * true_ret_vec).sum())

        # 3. 成交成本
        trade_cost = self.cost_coeff * float(np.abs(action - self.prev_w).sum())

        # 4. RL reward 缩放
        reward = self.alpha * (pnl - trade_cost)

        # 5. 推进时间
        self.prev_w = action.copy()
        self.curr_idx += 1
        self.episode_steps += 1

        # 6. 下一个观测
        next_obs = self.features_full[self.curr_idx].astype(np.float32)

        # 7. 判断是否结束
        done_due_to_time = (self.curr_idx >= self.end_index)
        done_due_to_horizon = (
            self.max_episode_steps is not None
            and self.episode_steps >= self.max_episode_steps
        )
        done = bool(done_due_to_time or done_due_to_horizon)
        # print(f"DEBUG: start_index={self.start_index}, curr_idx={self.curr_idx}, gap_idx={self.curr_idx - self.start_index}, end_index={self.end_index}, done={done}")

        info = {
            "pnl": pnl,
            "cost": trade_cost,
            "gross_leverage": float(np.abs(action).sum()),
            "t_index": int(self.curr_idx - self.start_index),
        }


        return next_obs, reward, done, info
