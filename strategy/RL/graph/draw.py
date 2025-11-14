#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接读取指定日志文件，提取 avg_step_reward 并显示折线图。
"""

import re
import matplotlib.pyplot as plt

# === 修改为你自己的日志路径 ===
LOG_PATH = "./strategy/RL/log/zjh.log"
# =================================


def parse_avg_step_reward(log_path):
    iters, rewards = [], []
    iter_auto = 1
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("["):
                continue
            m_iter = re.search(r'^\[(?:Iter|Epoch)?\s*(\d+)\]', line)
            idx = int(m_iter.group(1)) if m_iter else iter_auto
            m_reward = re.search(r'avg_step_reward\s*=\s*([+-]?\d+(?:\.\d+)?)', line)
            if not m_reward:
                continue
            reward = float(m_reward.group(1))
            iters.append(idx)
            rewards.append(reward)
            iter_auto += 1
    return iters, rewards


def main():
    iters, rewards = parse_avg_step_reward(LOG_PATH)
    if not iters:
        print("没有找到包含 avg_step_reward 的行")
        return

    plt.figure(figsize=(8, 4.8))
    plt.plot(iters, rewards, linestyle='-')
    plt.title("avg_step_reward per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("avg_step_reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
