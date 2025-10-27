import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Shared backbone MLP, then:
    - policy_head: outputs mean for each asset
    - value_head : outputs scalar V(s)

    动作空间: 连续 [-1, 1]^n_assets
    我们这里让 Actor 输出高斯分布的 mean/std，然后会用 tanh 去 squash 成 [-1,1]
    """

    def __init__(self, obs_dim, action_dim, hidden_sizes=(256, 128)):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # Actor head: 输出动作均值
        self.mu_head = nn.Linear(in_dim, action_dim)

        # log_std 可以是全局可学习参数，也可以是按状态输出。
        # 这里用全局参数（更稳定一些）。
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head: 输出状态价值 V(s)
        self.v_head = nn.Linear(in_dim, 1)

        # 初始化可稍微缩小输出范围，避免一开始太激进
        nn.init.uniform_(self.mu_head.weight, -0.01, 0.01)
        nn.init.constant_(self.mu_head.bias, 0.0)
        nn.init.uniform_(self.v_head.weight, -0.01, 0.01)
        nn.init.constant_(self.v_head.bias, 0.0)

    def forward(self, obs):
        """
        obs: (batch, obs_dim) float32 tensor
        return:
            mu:      (batch, action_dim)
            std:     (batch, action_dim)
            value:   (batch, 1)
        """
        x = self.backbone(obs)
        mu = self.mu_head(x)          # unrestricted, will tanh later at sample time
        std = torch.exp(self.log_std) # broadcastable
        v = self.v_head(x)
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
        actor_critic_hidden=(256,128),
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        value_coef=0.5,
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

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

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