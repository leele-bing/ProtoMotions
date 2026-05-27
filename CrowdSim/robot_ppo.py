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
    hidden_dims: tuple[int, ...] = (128,)
    vector_hidden_dims: tuple[int, ...] | None = None
    map_projection_hidden_dims: tuple[int, ...] | None = None
    actor_hidden_dims: tuple[int, ...] | None = None
    critic_hidden_dims: tuple[int, ...] | None = None
    map_encoder_channels: tuple[int, ...] = (8, 16)
    map_encoder_kernel_sizes: tuple[int, ...] = (3, 3)
    map_encoder_strides: tuple[int, ...] = (2, 2)
    map_encoder_paddings: tuple[int, ...] = (1, 1)
    vector_obs_dim: int | None = None
    map_size: int = 0
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
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (128,),
        vector_hidden_dims: tuple[int, ...] | None = None,
        map_projection_hidden_dims: tuple[int, ...] | None = None,
        actor_hidden_dims: tuple[int, ...] | None = None,
        critic_hidden_dims: tuple[int, ...] | None = None,
        map_encoder_channels: tuple[int, ...] = (8, 16),
        map_encoder_kernel_sizes: tuple[int, ...] = (3, 3),
        map_encoder_strides: tuple[int, ...] = (2, 2),
        map_encoder_paddings: tuple[int, ...] = (1, 1),
        vector_obs_dim: int | None = None,
        map_size: int = 0,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.vector_obs_dim = int(vector_obs_dim or obs_dim)
        self.map_size = int(map_size)
        self.map_obs_dim = self.map_size * self.map_size
        hidden_dims = normalize_dims(hidden_dims)
        vector_hidden_dims = normalize_dims(vector_hidden_dims or hidden_dims)
        map_projection_hidden_dims = normalize_dims(map_projection_hidden_dims or hidden_dims)
        actor_hidden_dims = normalize_dims(actor_hidden_dims or hidden_dims)
        critic_hidden_dims = normalize_dims(critic_hidden_dims or hidden_dims)
        self.has_map = (
            self.map_size > 0
            and self.vector_obs_dim + self.map_obs_dim <= self.obs_dim
        )
        self.vector_encoder = make_mlp(
            input_dim=self.vector_obs_dim,
            hidden_dims=vector_hidden_dims,
        )
        feature_dim = vector_hidden_dims[-1]
        if self.has_map:
            self.map_encoder = make_conv_encoder(
                channels=map_encoder_channels,
                kernel_sizes=map_encoder_kernel_sizes,
                strides=map_encoder_strides,
                paddings=map_encoder_paddings,
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 1, self.map_size, self.map_size)
                map_feature_dim = int(self.map_encoder(dummy).shape[-1])
            self.map_projection = make_mlp(
                input_dim=map_feature_dim,
                hidden_dims=map_projection_hidden_dims,
            )
            feature_dim += map_projection_hidden_dims[-1]
        else:
            self.map_encoder = None
            self.map_projection = None

        self.actor = make_mlp(
            input_dim=feature_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=action_dim,
        )
        self.critic = make_mlp(
            input_dim=feature_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        vector_obs = obs[:, : self.vector_obs_dim]
        features = [self.vector_encoder(vector_obs)]
        if self.has_map:
            map_start = self.vector_obs_dim
            map_end = map_start + self.map_obs_dim
            map_obs = obs[:, map_start:map_end].reshape(-1, 1, self.map_size, self.map_size)
            features.append(self.map_projection(self.map_encoder(map_obs)))
        return torch.cat(features, dim=-1) if len(features) > 1 else features[0]

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encode(obs)
        mean = self.actor(features)
        value = self.critic(features).squeeze(-1)
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


def make_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = int(input_dim)
    for hidden_dim in normalize_dims(hidden_dims):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.Tanh())
        current_dim = hidden_dim
    if output_dim is not None:
        layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


def make_conv_encoder(
    channels: tuple[int, ...],
    kernel_sizes: tuple[int, ...],
    strides: tuple[int, ...],
    paddings: tuple[int, ...],
) -> nn.Sequential:
    channels = normalize_dims(channels)
    kernel_sizes = expand_or_validate(kernel_sizes, len(channels), "kernel_sizes")
    strides = expand_or_validate(strides, len(channels), "strides")
    paddings = expand_or_validate(paddings, len(channels), "paddings")
    layers: list[nn.Module] = []
    in_channels = 1
    for out_channels, kernel_size, stride, padding in zip(
        channels,
        kernel_sizes,
        strides,
        paddings,
    ):
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)


def normalize_dims(value: int | float | str | list | tuple) -> tuple[int, ...]:
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        dims = tuple(int(dim) for dim in value)
    else:
        dims = (int(value),)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError(f"Network dimensions must be positive, got {value}.")
    return dims


def expand_or_validate(value: int | float | str | list | tuple, length: int, name: str) -> tuple[int, ...]:
    values = normalize_dims(value)
    if len(values) == 1:
        return values * length
    if len(values) != length:
        raise ValueError(f"Expected {name} to have length 1 or {length}, got {values}.")
    return values


def robot_network_kwargs(
    network_cfg: dict,
    hidden_dim_override: int | None = None,
    num_layers_override: int | None = None,
) -> dict:
    num_layers = int(num_layers_override or network_cfg.get("num_layers", 1))
    shared_value = (
        hidden_dim_override
        if hidden_dim_override is not None
        else network_cfg.get("hidden_dims", network_cfg.get("hidden_dim", 128))
    )
    shared_dims = network_hidden_dims(shared_value, num_layers)
    map_encoder_cfg = network_cfg.get("map_encoder", {})
    if not isinstance(map_encoder_cfg, dict):
        map_encoder_cfg = {}
    return {
        "hidden_dims": shared_dims,
        "vector_hidden_dims": optional_hidden_dims(
            network_cfg.get("vector_hidden_dims"),
            default=shared_dims,
        ),
        "map_projection_hidden_dims": optional_hidden_dims(
            network_cfg.get("map_projection_hidden_dims"),
            default=shared_dims,
        ),
        "actor_hidden_dims": optional_hidden_dims(
            network_cfg.get("actor_hidden_dims"),
            default=shared_dims,
        ),
        "critic_hidden_dims": optional_hidden_dims(
            network_cfg.get("critic_hidden_dims"),
            default=shared_dims,
        ),
        "map_encoder_channels": normalize_dims(map_encoder_cfg.get("channels", (8, 16))),
        "map_encoder_kernel_sizes": normalize_dims(map_encoder_cfg.get("kernel_sizes", (3, 3))),
        "map_encoder_strides": normalize_dims(map_encoder_cfg.get("strides", (2, 2))),
        "map_encoder_paddings": normalize_dims(map_encoder_cfg.get("paddings", (1, 1))),
    }


def network_hidden_dims(value, num_layers: int) -> tuple[int, ...]:
    dims = normalize_dims(value)
    if len(dims) == 1:
        return dims * max(1, int(num_layers))
    return dims


def optional_hidden_dims(value, default: tuple[int, ...]) -> tuple[int, ...] | None:
    if value is None:
        return None
    return normalize_dims(value)


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
            hidden_dims=config.hidden_dims,
            vector_hidden_dims=config.vector_hidden_dims,
            map_projection_hidden_dims=config.map_projection_hidden_dims,
            actor_hidden_dims=config.actor_hidden_dims,
            critic_hidden_dims=config.critic_hidden_dims,
            map_encoder_channels=config.map_encoder_channels,
            map_encoder_kernel_sizes=config.map_encoder_kernel_sizes,
            map_encoder_strides=config.map_encoder_strides,
            map_encoder_paddings=config.map_encoder_paddings,
            vector_obs_dim=config.vector_obs_dim,
            map_size=config.map_size,
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
