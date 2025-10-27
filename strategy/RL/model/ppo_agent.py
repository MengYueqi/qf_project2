import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(32, 32)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        self.mu_head = nn.Linear(in_dim, action_dim)
        # self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=False)

        self.v_head = nn.Linear(in_dim, 1)

        nn.init.uniform_(self.mu_head.weight, -0.01, 0.01)
        nn.init.constant_(self.mu_head.bias, 0.0)
        nn.init.uniform_(self.v_head.weight, -0.01, 0.01)
        nn.init.constant_(self.v_head.bias, 0.0)

    def forward(self, obs):
        x = self.backbone(obs)

        mu = self.mu_head(x)
        std = torch.exp(self.log_std)

        v_raw = self.v_head(x)
        v = torch.tanh(v_raw) * 10.0   # 🔥 critic输出强行限制到 [-10,10]

        return mu, std, v


    def act(self, obs):
        """
        用于 roll 环节：从策略采样一个动作，并且返回 log_prob 和 value
        obs: (batch, obs_dim)
        return dict:
          action: (batch, action_dim) in [-1,1] after tanh
          log_prob: (batch,)
          value: (batch,1)
        """
        mu, std, v = self.forward(obs)  # mu/std here are pre-tanh
        dist = Normal(mu, std)

        raw_action = dist.rsample()     # rsample for reparam trick
        # squash to [-1,1]
        action = torch.tanh(raw_action)

        # log_prob needs to account for tanh squashing:
        # log_prob_raw - sum(log(1 - tanh(a)^2))  (change of variable)
        log_prob_raw = dist.log_prob(raw_action)  # (batch, action_dim)
        log_prob_raw = log_prob_raw.sum(dim=-1)   # sum over action dims

        # Tanh correction term:
        # For each dim: log(1 - tanh(x)^2) = log(1 - action^2)
        # (action = tanh(raw_action))
        correction = torch.log(1 - action.pow(2) + 1e-8).sum(dim=-1)
        log_prob = log_prob_raw - correction
        # print(action)

        return {
            "action": action,
            "log_prob": log_prob,
            "value": v,
        }

    def evaluate_actions(self, obs, actions):
        """
        用于 PPO 更新阶段：给定旧的 obs 和 旧的 actions，
        计算新的 log_prob, entropy, value。

        obs:     (batch, obs_dim)
        actions: (batch, action_dim) already in [-1,1]

        return:
          log_prob: (batch,)
          entropy:  (batch,)
          value:    (batch,1)
        """
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)

        # inverse tanh to recover pre-squash raw_action
        # raw = atanh(a) = 0.5 * ln((1+a)/(1-a))
        # clamp for numerical stability
        clipped_actions = torch.clamp(actions, -0.999999, 0.999999)
        raw_action = 0.5 * torch.log((1 + clipped_actions) / (1 - clipped_actions))

        log_prob_raw = dist.log_prob(raw_action).sum(dim=-1)

        correction = torch.log(1 - actions.pow(2) + 1e-8).sum(dim=-1)
        log_prob = log_prob_raw - correction

        # Entropy of tanh-squashed Normal is trickier.
        # A common simplification: use the entropy of the pre-squash Normal.
        # That's fine for PPO as an entropy bonus.
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, v


class RolloutBuffer:
    """
    用来暂存一段交互轨迹（on-policy），
    然后在 PPO 更新时计算 advantage / returns。

    我们会存:
    - obs
    - actions
    - log_probs
    - rewards
    - dones
    - values
    """

    def __init__(self, buffer_size, obs_dim, action_dim, device):
        self.buffer_size = buffer_size
        self.device = device

        self.obs          = torch.zeros((buffer_size, obs_dim),    dtype=torch.float32, device=device)
        self.actions      = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.log_probs    = torch.zeros((buffer_size,),            dtype=torch.float32, device=device)
        self.rewards      = torch.zeros((buffer_size,),            dtype=torch.float32, device=device)
        self.dones        = torch.zeros((buffer_size,),            dtype=torch.float32, device=device)
        self.values       = torch.zeros((buffer_size,),            dtype=torch.float32, device=device)

        # filled length so far
        self.ptr = 0
        self.full = False

        # to be computed after rollout:
        self.advantages   = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.returns      = torch.zeros((buffer_size,), dtype=torch.float32, device=device)

    def add(self, obs, action, log_prob, reward, done, value):
        """
        obs:       (obs_dim,) torch tensor
        action:    (action_dim,) torch tensor
        log_prob:  scalar tensor
        reward:    scalar float or tensor
        done:      scalar float/bool
        value:     scalar tensor (critic output)
        """
        self.obs[self.ptr]        = obs
        self.actions[self.ptr]    = action
        self.log_probs[self.ptr]  = log_prob
        self.rewards[self.ptr]    = reward
        self.dones[self.ptr]      = float(done)
        self.values[self.ptr]     = value.squeeze(-1)  # critic gives (1,), we store scalar

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = self.buffer_size  # Stop incrementing further

    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        使用 GAE(λ) 计算 advantage 和 return (a.k.a. value targets)
        - last_value: V(s_T) for the final state after the last step in buffer
          shape: (1,) tensor

        After this call:
        self.advantages[i]
        self.returns[i]  ( = advantages[i] + values[i] )
        """
        # We'll iterate backward
        gae = 0.0
        last_adv = 0.0
        buffer_size = self.ptr  # only filled part
        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step+1]
                next_value = self.values[step+1]

            delta = (
                self.rewards[step]
                + gamma * next_value * next_non_terminal
                - self.values[step]
            )

            gae = (
                delta
                + gamma * gae_lambda * next_non_terminal * gae
            )
            self.advantages[step] = gae

        self.returns[:buffer_size] = self.advantages[:buffer_size] + self.values[:buffer_size]

        # === advantage 标准化 + clamp 防数值爆炸 ===
        adv = self.advantages[:buffer_size]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = torch.clamp(adv, -100.0, 100.0)

        rets = self.returns[:buffer_size]
        rets = torch.clamp(rets, -1e3, 1e3)

        self.advantages[:buffer_size] = adv
        self.returns[:buffer_size] = rets


    def get(self, batch_size):
        """
        一个简单的mini-batch iterator
        打乱索引，yield小批数据给PPO更新
        """
        buffer_size = self.ptr
        idx = torch.randperm(buffer_size, device=self.device)

        for start in range(0, buffer_size, batch_size):
            end = start + batch_size
            mb_idx = idx[start:end]

            yield (
                self.obs[mb_idx],
                self.actions[mb_idx],
                self.log_probs[mb_idx],
                self.advantages[mb_idx],
                self.returns[mb_idx],
                self.values[mb_idx],
            )


class PPOAgent:
    """
    PPO 算法本体 (不包含环境交互循环)
    - 持有 ActorCritic
    - 提供 update() 来执行一次 PPO 参数更新
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        actor_critic_hidden=(16,16),
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        value_coef=0.1,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device="cpu",
    ):
        self.device = torch.device(device)

        self.model = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=actor_critic_hidden,
        ).to(self.device)

        actor_params = list(self.model.backbone.parameters()) + \
               list(self.model.mu_head.parameters()) + \
               [self.model.log_std]

        critic_params = list(self.model.v_head.parameters())

        self.optimizer = optim.Adam(
                [
                {"params": actor_params,  "lr": lr},         # e.g. lr = 3e-4
                {"params": critic_params, "lr": lr * 0.1},   # critic 学慢一点，比如 3e-5
            ]
        )


        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    @torch.no_grad()
    def choose_action(self, obs_tensor):
        """
        给环境时步用：
        obs_tensor: shape (obs_dim,) or (1, obs_dim)
        return:
            action (np.array in [-1,1]^action_dim)
            log_prob (float)
            value (float)
        """
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        obs_tensor = obs_tensor.to(self.device)

        out = self.model.act(obs_tensor)
        action      = out["action"][0].cpu().numpy()
        log_prob    = out["log_prob"][0].cpu().item()
        value       = out["value"][0].cpu().item()
        return action, log_prob, value
    
    @torch.no_grad()
    def choose_action_deterministic(agent, obs_tensor):
        # 用策略均值 mu 而不是随机采样
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        obs_tensor = obs_tensor.to(agent.device)

        mu, std, v = agent.model.forward(obs_tensor)
        action = torch.tanh(mu)  # [-1,1]
        return action[0].cpu().numpy(), v[0].cpu().item()

    def evaluate_agent_once(agent, eval_env, max_steps=1000):
        obs_np = eval_env.reset()
        obs = torch.tensor(obs_np, dtype=torch.float32, device=agent.device)

        equity_curve = [0.0]
        pnl_history = []

        for t in range(max_steps):
            action_np, value_est = choose_action_deterministic(agent, obs)

            next_obs_np, reward, done, info = eval_env.step(action_np)

            pnl = info["pnl"]       # 真实组合当期收益 (未放大, 未加alpha)
            pnl_history.append(pnl)
            equity_curve.append(equity_curve[-1] + pnl)

            obs = torch.tensor(next_obs_np, dtype=torch.float32, device=agent.device)

            if done:
                break

        equity_curve = np.array(equity_curve[1:])
        pnl_history = np.array(pnl_history)

        total_return = equity_curve[-1] if len(equity_curve) else 0.0
        avg_pnl = pnl_history.mean() if len(pnl_history) else 0.0
        vol_pnl = pnl_history.std() + 1e-8
        sharpe_like = (avg_pnl / vol_pnl) * np.sqrt(252)

        # 最大回撤
        running_max = -np.inf
        drawdowns = []
        for v in equity_curve:
            if v > running_max:
                running_max = v
            drawdowns.append(running_max - v)
        max_dd = max(drawdowns) if drawdowns else 0.0

        print("=== EVAL RESULT ===")
        print(f"Steps traded : {len(pnl_history)}")
        print(f"Cumulative PnL: {total_return:.6f}")
        print(f"Mean pnl/step: {avg_pnl:.6f}")
        print(f"Vol  pnl/step: {vol_pnl:.6f}")
        print(f"Sharpe-like  : {sharpe_like:.3f}")
        print(f"Max Drawdown : {max_dd:.6f}")

        return equity_curve, pnl_history


    def update(self, rollout_buffer, epochs=5, batch_size=64):
        """
        用 rollout_buffer 里的数据跑一次 PPO 更新
        返回一个 dict，包含这次更新的一些指标
        """
        stats_last = {}

        for _ in range(epochs):
            for (
                obs_b,
                act_b,
                old_logp_b,
                adv_b,
                ret_b,
                val_b
            ) in rollout_buffer.get(batch_size):

                obs_b      = obs_b.to(self.device)
                act_b      = act_b.to(self.device)
                old_logp_b = old_logp_b.to(self.device)
                adv_b      = adv_b.to(self.device)
                ret_b      = ret_b.to(self.device)

                # 标准化 advantage
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)

                new_logp, entropy, new_v = self.model.evaluate_actions(obs_b, act_b)

                ratio = torch.exp(new_logp - old_logp_b)

                unclipped_obj = ratio * adv_b
                clipped_obj   = torch.clamp(
                    ratio,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range
                ) * adv_b

                policy_loss = -torch.min(unclipped_obj, clipped_obj).mean()
                value_loss  = (new_v.squeeze(-1) - ret_b).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats_last = {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy.mean().item(),
                    "ret_mean": ret_b.mean().item(),
                }

        return stats_last