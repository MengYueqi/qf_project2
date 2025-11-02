import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(256,128)):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.body = nn.Sequential(*layers)

        # actor 输出均值
        self.mu_head = nn.Linear(in_dim, action_dim)

        # actor 输出 log_std（可学习的全局参数 or 线性层）
        # 版本1：全局可学习参数（跟你现在的一样）
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # 初始化稍微保守一点
        nn.init.uniform_(self.mu_head.weight, -0.01, 0.01)
        nn.init.constant_(self.mu_head.bias, 0.0)

    def forward(self, obs):
        """
        obs: (batch, obs_dim)
        return:
            mu:  (batch, action_dim)
            std: (batch, action_dim)
        """
        x = self.body(obs)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)  # broadcast
        return mu, std

    def act(self, obs):
        """
        采样带tanh squash的动作 + log_prob
        obs: (batch, obs_dim)
        """
        mu, std = self.forward(obs)
        dist = Normal(mu, std)

        raw_action = dist.rsample()          # (batch, action_dim)
        action = torch.tanh(raw_action)      # squash到[-1,1]

        # log_prob 校正tanh
        log_prob_raw = dist.log_prob(raw_action).sum(dim=-1)
        correction = torch.log(1 - action.pow(2) + 1e-8).sum(dim=-1)
        log_prob = log_prob_raw - correction  # (batch,)

        # 熵（用pre-squash的Normal的熵）
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy

    def evaluate_actions(self, obs, actions):
        """
        用于 PPO 更新阶段：
        给历史 (obs, actions) 算新的 log_prob / entropy
        obs: (batch, obs_dim)
        actions: (batch, action_dim) in [-1,1]
        """
        mu, std = self.forward(obs)
        dist = Normal(mu, std)

        # atanh 反推 raw_action
        clipped_actions = torch.clamp(actions, -0.999999, 0.999999)
        raw_action = 0.5 * torch.log((1 + clipped_actions) / (1 - clipped_actions))

        log_prob_raw = dist.log_prob(raw_action).sum(dim=-1)
        correction = torch.log(1 - actions.pow(2) + 1e-8).sum(dim=-1)
        log_prob = log_prob_raw - correction

        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class CriticNet(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(256,128)):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.body = nn.Sequential(*layers)

        self.v_head = nn.Linear(in_dim, 1)

        nn.init.uniform_(self.v_head.weight, -0.01, 0.01)
        nn.init.constant_(self.v_head.bias, 0.0)

    def forward(self, obs):
        """
        obs: (batch, obs_dim)
        return:
            value: (batch,1)
        """
        x = self.body(obs)
        v = self.v_head(x)
        return v



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
        # adv = torch.clamp(adv, -100.0, 100.0)

        rets = self.returns[:buffer_size]
        # rets = torch.clamp(rets, -1e3, 1e3)

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
    def __init__(
        self,
        obs_dim,
        action_dim,
        actor_hidden=(256,128),
        critic_hidden=(256,256,32),
        lr=5e-5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.001,
        max_grad_norm=0.5,
        device="cpu",
    ):
        self.device = torch.device(device)

        # 分离的 Actor / Critic
        self.actor = ActorNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=actor_hidden,
        ).to(self.device)

        self.critic = CriticNet(
            obs_dim=obs_dim,
            hidden_sizes=critic_hidden,
        ).to(self.device)

        # 一个 optimizer 管两个网络的参数（最简单的做法）
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
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
        与你原来的接口保持一致，供环境 rollout 使用
        return: action (np), log_prob (float), value (float)
        """
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        obs_tensor = obs_tensor.to(self.device)

        # actor 采样动作
        action_t, logp_t, _entropy_t = self.actor.act(obs_tensor)
        # critic 估值
        value_t = self.critic(obs_tensor)

        action_np = action_t[0].cpu().numpy()
        logp = logp_t[0].cpu().item()
        value = value_t[0].cpu().item()

        return action_np, logp, value

    @torch.no_grad()
    def choose_action_deterministic(self, obs_tensor):
        """
        用于 eval：直接用 actor 的均值的tanh作为动作，而不是采样
        """
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        obs_tensor = obs_tensor.to(self.device)

        mu, std = self.actor.forward(obs_tensor)
        action_det = torch.tanh(mu)      # 均值经过tanh
        value_t = self.critic(obs_tensor)

        return (
            action_det[0].cpu().numpy(),
            value_t[0].cpu().item()
        )

    def update(self, rollout_buffer, epochs=5, batch_size=64):
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

                # 标准化 advantage（稳定 PPO）
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)

                # 1. 重新算 actor 的 log_prob, entropy
                new_logp, entropy = self.actor.evaluate_actions(obs_b, act_b)

                # 2. critic 新的 value 估计
                new_v = self.critic(obs_b).squeeze(-1)  # (batch,)

                # PPO ratio
                ratio = torch.exp(new_logp - old_logp_b)

                unclipped_obj = ratio * adv_b
                clipped_obj   = torch.clamp(
                    ratio,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range
                ) * adv_b

                policy_loss = -torch.min(unclipped_obj, clipped_obj).mean()

                # value loss
                value_loss  = (new_v - ret_b).pow(2).mean()

                # entropy bonus (encourage exploration)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                stats_last = {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy.mean().item(),
                    "ret_mean": ret_b.mean().item(),
                }

        return stats_last
