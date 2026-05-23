"""Small PPO implementation for CrowdSim robot navigation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass
class RobotPPOConfig:
    obs_dim: int
    action_dim: int = 2
    hidden_dim: int = 128
    lr: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4
    minibatch_size: int = 256


class RobotActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std, value

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self(obs)
        dist = Normal(mean, log_std.exp())
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = tanh_normal_log_prob(dist, raw_action, action)
        return action, raw_action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        raw_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self(obs)
        dist = Normal(mean, log_std.exp())
        actions = torch.tanh(raw_actions)
        log_prob = tanh_normal_log_prob(dist, raw_actions, actions)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


def tanh_normal_log_prob(
    dist: Normal,
    raw_action: torch.Tensor,
    action: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    log_prob = dist.log_prob(raw_action).sum(dim=-1)
    correction = torch.log(1.0 - action.pow(2) + eps).sum(dim=-1)
    return log_prob - correction


class RobotRolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.device = device
        self.obs = torch.zeros(rollout_steps, num_envs, obs_dim, device=device)
        self.raw_actions = torch.zeros(rollout_steps, num_envs, action_dim, device=device)
        self.log_probs = torch.zeros(rollout_steps, num_envs, device=device)
        self.rewards = torch.zeros(rollout_steps, num_envs, device=device)
        self.dones = torch.zeros(rollout_steps, num_envs, device=device)
        self.values = torch.zeros(rollout_steps, num_envs, device=device)
        self.advantages = torch.zeros(rollout_steps, num_envs, device=device)
        self.returns = torch.zeros(rollout_steps, num_envs, device=device)
        self.step = 0

    def add(
        self,
        obs: torch.Tensor,
        raw_actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        if self.step >= self.rollout_steps:
            raise RuntimeError("RolloutBuffer is full.")
        self.obs[self.step].copy_(obs)
        self.raw_actions[self.step].copy_(raw_actions)
        self.log_probs[self.step].copy_(log_probs)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step].copy_(dones.float())
        self.values[self.step].copy_(values)
        self.step += 1

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        advantage = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.rollout_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.rollout_steps - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantage = delta + gamma * gae_lambda * next_non_terminal * advantage
            self.advantages[t] = advantage
        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std(unbiased=False) + 1.0e-8
        )

    def batches(self, minibatch_size: int):
        total = self.rollout_steps * self.num_envs
        indices = torch.randperm(total, device=self.device)
        flat = {
            "obs": self.obs.reshape(total, -1),
            "raw_actions": self.raw_actions.reshape(total, -1),
            "log_probs": self.log_probs.reshape(total),
            "advantages": self.advantages.reshape(total),
            "returns": self.returns.reshape(total),
        }
        for start in range(0, total, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            yield {key: value[batch_idx] for key, value in flat.items()}

    def reset(self) -> None:
        self.step = 0


class RobotPPOTrainer:
    def __init__(self, config: RobotPPOConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.model = RobotActorCritic(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.act(obs)

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)[2]

    def update(self, buffer: RobotRolloutBuffer) -> dict[str, float]:
        cfg = self.config
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}
        for _ in range(cfg.ppo_epochs):
            for batch in buffer.batches(cfg.minibatch_size):
                new_log_probs, entropy, values = self.model.evaluate_actions(
                    batch["obs"], batch["raw_actions"]
                )
                ratio = torch.exp(new_log_probs - batch["log_probs"])
                unclipped = ratio * batch["advantages"]
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                policy_loss = -torch.min(unclipped, clipped * batch["advantages"]).mean()
                value_loss = F.mse_loss(values, batch["returns"])
                entropy_loss = entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += float(policy_loss.detach().cpu())
                stats["value_loss"] += float(value_loss.detach().cpu())
                stats["entropy"] += float(entropy_loss.detach().cpu())
                stats["updates"] += 1

        count = max(stats["updates"], 1)
        return {
            "policy_loss": stats["policy_loss"] / count,
            "value_loss": stats["value_loss"] / count,
            "entropy": stats["entropy"] / count,
        }

    def save(self, path: Path, step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load(self, path: Path) -> int:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        return int(payload.get("step", 0))
