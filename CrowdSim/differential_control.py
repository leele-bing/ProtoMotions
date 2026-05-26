"""Manual differential-drive control helpers for CrowdSim cars."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Fixed Nova Carter differential-drive parameters.
MAX_LINEAR_SPEED = 2.0
MAX_ANGULAR_SPEED = 3.0
WHEEL_BASE = 0.413
WHEEL_RADIUS = 0.14
MAX_WHEEL_SPEED = (
    MAX_LINEAR_SPEED + 0.5 * WHEEL_BASE * MAX_ANGULAR_SPEED
) / WHEEL_RADIUS
LEFT_WHEEL_JOINT_NAMES = ("joint_wheel_left",)
RIGHT_WHEEL_JOINT_NAMES = ("joint_wheel_right",)


@dataclass(frozen=True)
class DifferentialDriveConfig:
    max_linear_speed: float = MAX_LINEAR_SPEED
    max_angular_speed: float = MAX_ANGULAR_SPEED
    wheel_radius: float = WHEEL_RADIUS
    wheel_base: float = WHEEL_BASE
    max_wheel_speed: float = MAX_WHEEL_SPEED
    left_wheel_joint_names: tuple[str, ...] = LEFT_WHEEL_JOINT_NAMES
    right_wheel_joint_names: tuple[str, ...] = RIGHT_WHEEL_JOINT_NAMES


class ManualDifferentialController:
    """Convert unicycle commands to left/right wheel angular velocities."""

    def __init__(self, config: DifferentialDriveConfig | None = None) -> None:
        self.config = config or DifferentialDriveConfig()

    def forward(self, command: np.ndarray) -> np.ndarray:
        forward_speed = float(
            np.clip(command[0], -self.config.max_linear_speed, self.config.max_linear_speed)
        )
        yaw_rate = float(
            np.clip(command[1], -self.config.max_angular_speed, self.config.max_angular_speed)
        )
        left = (
            forward_speed - 0.5 * self.config.wheel_base * yaw_rate
        ) / self.config.wheel_radius
        right = (
            forward_speed + 0.5 * self.config.wheel_base * yaw_rate
        ) / self.config.wheel_radius
        wheels = np.array([left, right], dtype=np.float32)
        return np.clip(wheels, -self.config.max_wheel_speed, self.config.max_wheel_speed)

    def action_to_wheels(self, action: np.ndarray) -> np.ndarray:
        """Convert a normalized RL action into wheel angular velocities."""
        forward_speed = float(np.clip(action[0], -1.0, 1.0)) * self.config.max_linear_speed
        yaw_rate = float(np.clip(action[1], -1.0, 1.0)) * self.config.max_angular_speed
        return self.forward(np.array([forward_speed, yaw_rate], dtype=np.float32))
