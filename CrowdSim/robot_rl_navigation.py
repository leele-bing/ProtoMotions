"""Robot PPO navigation helpers kept separate from the core navigation manager."""

from __future__ import annotations

import math

import numpy as np
import torch

DEBUG_ROBOT_DRIVE = False
DEBUG_CONSTANT_ROBOT_COMMAND: tuple[float, float] | None = None


class RobotRLNavigationMixin:
    """Mixin for robot-only PPO observations, rewards, and episode resets."""

    def set_robot_rl_actions(self, actions: torch.Tensor | np.ndarray) -> None:
        if not self.config.car_rl_policy or self.config.num_robots == 0:
            return
        if isinstance(actions, torch.Tensor):
            values = actions.detach().cpu().numpy()
        else:
            values = np.asarray(actions, dtype=np.float32)
        if values.shape != (self.config.num_robots, 2):
            raise ValueError(
                f"Expected RL robot actions with shape ({self.config.num_robots}, 2), "
                f"got {values.shape}."
            )
        self._robot_rl_actions[:] = np.clip(values, -1.0, 1.0)

    def _compute_robot_wheel_targets(self) -> np.ndarray:
        wheel_targets = np.zeros((self.config.num_robots, 2), dtype=np.float32)
        if not self.config.car_rl_policy:
            return wheel_targets
        for robot_idx in range(self.config.num_robots):
            wheel_targets[robot_idx] = self._wheel_controller.action_to_wheels(
                self._robot_rl_actions[robot_idx],
            )
        return wheel_targets

    def _write_robot_wheel_targets(self, wheel_targets: np.ndarray) -> None:
        if self._wheel_targets_tensor is None:
            return

        targets_cpu = torch.as_tensor(wheel_targets, dtype=torch.float32)
        self._wheel_targets_tensor.copy_(targets_cpu, non_blocking=True)
        targets = self._wheel_targets_tensor
        if self._wheel_joint_ids is None:
            raise RuntimeError("Robot wheel joint targets were not initialized.")

        self.robot.set_joint_velocity_target(targets, joint_ids=self._wheel_joint_ids)

    def _override_robot_rl_actions_with_constant_command(self) -> None:
        if not self.config.car_rl_policy or self.config.num_robots == 0:
            return

        if DEBUG_CONSTANT_ROBOT_COMMAND is None:
            return

        command = np.asarray(DEBUG_CONSTANT_ROBOT_COMMAND, dtype=np.float32)
        if command.shape != (2,):
            raise ValueError(
                "DEBUG_CONSTANT_ROBOT_COMMAND must have shape (2,) for "
                "(linear_velocity, angular_velocity)."
            )
        action = np.array(
            [
                float(command[0]) / max(float(self.config.rl_max_linear_velocity), 1e-6),
                float(command[1]) / max(float(self.config.rl_max_angular_velocity), 1e-6),
            ],
            dtype=np.float32,
        )
        self._robot_rl_actions[:] = np.clip(action, -1.0, 1.0)

    def _robot_actions_to_velocity_commands(self) -> np.ndarray:
        commands = np.zeros_like(self._robot_rl_actions, dtype=np.float32)
        commands[:, 0] = (
            np.clip(self._robot_rl_actions[:, 0], -1.0, 1.0)
            * float(self.config.rl_max_linear_velocity)
        )
        commands[:, 1] = (
            np.clip(self._robot_rl_actions[:, 1], -1.0, 1.0)
            * float(self.config.rl_max_angular_velocity)
        )
        return commands

    def _robot_drive_debug_enabled(self) -> bool:
        return DEBUG_ROBOT_DRIVE or DEBUG_CONSTANT_ROBOT_COMMAND is not None

    def _apply_robot_drive_after_env_step(self) -> None:
        if not self.config.car_rl_policy or self.config.num_robots == 0:
            return
        if self.config.car_drive_mode != "kinematic":
            return

        self._clear_robot_wheel_targets()
        commands = self._robot_actions_to_velocity_commands()
        yaws = self._robot_yaws()
        dt = float(getattr(self, "_env_dt", 0.0) or 0.0)
        if dt <= 0.0:
            dt = 1.0 / 30.0

        linear = commands[:, 0]
        angular = commands[:, 1]
        yaw_mid = yaws + 0.5 * angular * dt
        next_yaws = yaws + angular * dt

        poses = torch.cat(
            (
                self.robot.data.root_pos_w[:, :3].clone(),
                self.robot.data.root_quat_w[:, :4].clone(),
            ),
            dim=1,
        )
        poses[:, 0] += torch.as_tensor(
            linear * np.cos(yaw_mid) * dt,
            dtype=torch.float32,
            device=self.config.device,
        )
        poses[:, 1] += torch.as_tensor(
            linear * np.sin(yaw_mid) * dt,
            dtype=torch.float32,
            device=self.config.device,
        )
        poses[:, 3:7] = self._yaw_to_quat_tensor(
            torch.as_tensor(next_yaws, dtype=torch.float32, device=self.config.device)
        )

        velocities = torch.zeros(
            (self.config.num_robots, 6),
            dtype=torch.float32,
            device=self.config.device,
        )
        velocities[:, 0] = torch.as_tensor(
            linear * np.cos(next_yaws),
            dtype=torch.float32,
            device=self.config.device,
        )
        velocities[:, 1] = torch.as_tensor(
            linear * np.sin(next_yaws),
            dtype=torch.float32,
            device=self.config.device,
        )
        velocities[:, 5] = torch.as_tensor(
            angular,
            dtype=torch.float32,
            device=self.config.device,
        )

        self.robot.write_root_pose_to_sim(poses)
        self.robot.write_root_velocity_to_sim(velocities)

    def _print_robot_drive_debug(self) -> None:
        if not self.config.car_rl_policy or self.config.num_robots == 0:
            return
        if not self._robot_drive_debug_enabled():
            return
        if self._wheel_targets_tensor is None:
            return

        actual_wheel_velocities = None
        if self._wheel_joint_ids is not None and hasattr(self.robot.data, "joint_vel"):
            actual_wheel_velocities = (
                self.robot.data.joint_vel[:, self._wheel_joint_ids].detach().cpu().numpy()
            )

        print(
            "[CrowdSim][RobotDrive] "
            f"command_vw={self._robot_actions_to_velocity_commands().tolist()} "
            f"wheel_targets={self._wheel_targets_tensor.detach().cpu().numpy().tolist()} "
            f"actual_wheel_velocities="
            f"{None if actual_wheel_velocities is None else actual_wheel_velocities.tolist()}"
        )

    def _clear_robot_wheel_targets(self, env_ids: torch.Tensor | None = None) -> None:
        if self._wheel_targets_tensor is None or self._wheel_joint_ids is None:
            return

        if env_ids is None:
            self._wheel_targets_tensor.zero_()
            targets = self._wheel_targets_tensor
        else:
            self._wheel_targets_tensor[env_ids] = 0.0
            targets = torch.zeros(
                (int(env_ids.numel()), len(self._wheel_joint_ids)),
                dtype=torch.float32,
                device=self.config.device,
            )
        self.robot.set_joint_velocity_target(
            targets,
            joint_ids=self._wheel_joint_ids,
            env_ids=env_ids,
        )

    def get_robot_rl_observations(self) -> torch.Tensor:
        positions, velocities = self._read_agent_state()
        obs = self._build_robot_rl_observations(positions, velocities)
        self._robot_last_obs = obs
        return obs

    def get_robot_rl_feedback(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self._robot_last_obs is None:
            self.get_robot_rl_observations()
        return (
            self._robot_last_obs,
            self._robot_last_rewards,
            self._robot_last_dones,
            self._robot_last_info,
        )

    def _clear_robot_rl_env_step_feedback(self) -> None:
        if not self.config.car_rl_policy or self.config.num_robots == 0:
            return

        self._robot_last_rewards.zero_()
        self._robot_last_dones.zero_()
        zero_float = torch.zeros(self.config.num_robots, dtype=torch.float32, device=self.config.device)
        zero_bool = torch.zeros(self.config.num_robots, dtype=torch.bool, device=self.config.device)
        self._robot_last_info = {
            **self._robot_last_info,
            "is_navigation_step": zero_bool,
            "reached": zero_bool,
            "collision": zero_bool,
            "timeout": zero_bool,
            "progress": zero_float,
            "reward_total": zero_float,
            "reward_time": zero_float,
            "reward_progress": zero_float,
            "reward_goal": zero_float,
            "reward_collision": zero_float,
            "reward_timeout": zero_float,
        }

    @property
    def robot_rl_obs_dim(self) -> int:
        return self.robot_rl_vector_obs_dim + self.robot_rl_map_obs_dim

    @property
    def robot_rl_vector_obs_dim(self) -> int:
        return 8 + 4 * self.config.rl_num_neighbors

    @property
    def robot_rl_map_obs_dim(self) -> int:
        map_size = max(0, int(self.config.rl_map_size))
        return map_size * map_size

    def reset_robot_rl_episodes(self, done: torch.Tensor | np.ndarray) -> None:
        if not self.config.car_rl_policy or self.config.num_robots == 0:
            return
        if isinstance(done, torch.Tensor):
            done_np = done.detach().cpu().numpy().astype(bool)
        else:
            done_np = np.asarray(done, dtype=bool)
        robot_ids = np.nonzero(done_np)[0]
        if len(robot_ids) == 0:
            return

        reset_reasons = {
            int(robot_id): self._robot_reset_reason(int(robot_id))
            for robot_id in robot_ids
        }
        agent_offset = self.config.num_humanoids
        agent_ids = agent_offset + robot_ids
        yaw = self._initial_robot_yaws()[robot_ids]
        env_ids = torch.as_tensor(robot_ids, dtype=torch.long, device=self.config.device)
        poses = self.robot.data.default_root_state[env_ids, :7].clone()
        poses[:, 0:2] = torch.as_tensor(
            self.starts_xy[agent_ids], dtype=torch.float32, device=self.config.device
        )
        poses[:, 3:7] = self._yaw_to_quat_tensor(torch.as_tensor(yaw, device=self.config.device))

        self.robot.write_root_pose_to_sim(poses, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(
            torch.zeros((len(robot_ids), 6), dtype=torch.float32, device=self.config.device),
            env_ids=env_ids,
        )
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids].clone(),
            torch.zeros_like(self.robot.data.default_joint_vel[env_ids]),
            env_ids=env_ids,
        )
        self.robot.reset(env_ids=env_ids)
        self._clear_robot_wheel_targets(env_ids)

        self._robot_rl_actions[robot_ids] = 0.0
        for agent_id in agent_ids:
            self._replan_agent_from_xy(int(agent_id), self.starts_xy[int(agent_id)])
        for robot_id in robot_ids:
            print(
                "[CrowdSim] reset car "
                f"env={int(robot_id)} agent={int(agent_offset + robot_id)} "
                f"reason={reset_reasons[int(robot_id)]}"
            )
        self._robot_episode_steps[robot_ids] = 0
        self._robot_progress_targets[robot_ids] = self.starts_xy[agent_ids]
        self._robot_prev_progress_dist[robot_ids] = 0.0
        agent_id_set = {int(agent_id) for agent_id in agent_ids}
        self.collision_pairs = {
            pair
            for pair in self.collision_pairs
            if pair[0] not in agent_id_set and pair[1] not in agent_id_set
        }
        self._robot_last_rewards[env_ids] = 0.0
        self._robot_last_dones[env_ids] = False
        for value in self._robot_last_info.values():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (self.config.num_robots,):
                if value.dtype == torch.bool:
                    value[env_ids] = False
                else:
                    value[env_ids] = 0.0

    def _robot_reset_reason(self, robot_id: int) -> str:
        reasons = []
        for key in ("reached", "collision", "timeout"):
            value = self._robot_last_info.get(key)
            if isinstance(value, torch.Tensor) and value.numel() > robot_id:
                if bool(value[robot_id].detach().cpu().item()):
                    reasons.append(key)
        return "+".join(reasons) if reasons else "external"

    def _update_robot_rl_feedback(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        new_pairs: set[tuple[int, int]],
    ) -> None:
        if not self.config.car_rl_policy or self.config.num_robots == 0:
            return

        robot_offset = self.config.num_humanoids
        robot_agent_ids = np.arange(robot_offset, robot_offset + self.config.num_robots)
        current_goal_dist = np.linalg.norm(
            positions[robot_agent_ids] - self.goals_xy[robot_agent_ids],
            axis=1,
        )
        current_progress_dist = np.linalg.norm(
            positions[robot_agent_ids] - self._robot_progress_targets,
            axis=1,
        )
        progress = self._robot_prev_progress_dist - current_progress_dist
        self._robot_episode_steps += 1

        collision = np.zeros(self.config.num_robots, dtype=bool)
        for i, j in new_pairs:
            if robot_offset <= i < robot_offset + self.config.num_robots:
                collision[i - robot_offset] = True
            if robot_offset <= j < robot_offset + self.config.num_robots:
                collision[j - robot_offset] = True

        reached = self.reached[robot_agent_ids].copy()
        timeout = self._robot_episode_steps >= self.config.rl_max_episode_steps
        time_reward = np.full(self.config.num_robots, self.config.rl_time_penalty, dtype=np.float32)
        progress_reward = (self.config.rl_progress_reward_scale * progress).astype(np.float32)
        goal_reward = (self.config.rl_goal_reward * reached.astype(np.float32)).astype(np.float32)
        collision_reward = (
            self.config.rl_collision_penalty * collision.astype(np.float32)
        ).astype(np.float32)
        timeout_reward = (self.config.rl_timeout_penalty * timeout.astype(np.float32)).astype(np.float32)
        rewards = (
            time_reward
            + progress_reward
            + goal_reward
            + collision_reward
            + timeout_reward
        ).astype(np.float32)
        dones = reached | collision | timeout

        self._robot_last_obs = self._build_robot_rl_observations(positions, velocities)
        self._robot_last_rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.config.device)
        self._robot_last_dones = torch.as_tensor(dones, dtype=torch.bool, device=self.config.device)
        self._robot_last_info = {
            "reached": torch.as_tensor(reached, dtype=torch.bool, device=self.config.device),
            "collision": torch.as_tensor(collision, dtype=torch.bool, device=self.config.device),
            "timeout": torch.as_tensor(timeout, dtype=torch.bool, device=self.config.device),
            "distance_to_goal": torch.as_tensor(current_goal_dist, dtype=torch.float32, device=self.config.device),
            "distance_to_progress_target": torch.as_tensor(
                current_progress_dist, dtype=torch.float32, device=self.config.device
            ),
            "progress": torch.as_tensor(progress, dtype=torch.float32, device=self.config.device),
            "reward_total": torch.as_tensor(rewards, dtype=torch.float32, device=self.config.device),
            "reward_time": torch.as_tensor(time_reward, dtype=torch.float32, device=self.config.device),
            "reward_progress": torch.as_tensor(progress_reward, dtype=torch.float32, device=self.config.device),
            "reward_goal": torch.as_tensor(goal_reward, dtype=torch.float32, device=self.config.device),
            "reward_collision": torch.as_tensor(collision_reward, dtype=torch.float32, device=self.config.device),
            "reward_timeout": torch.as_tensor(timeout_reward, dtype=torch.float32, device=self.config.device),
            "is_navigation_step": torch.ones(
                self.config.num_robots,
                dtype=torch.bool,
                device=self.config.device,
            ),
        }

    def _build_robot_rl_observations(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> torch.Tensor:
        if self.config.num_robots == 0:
            return torch.zeros((0, self.robot_rl_obs_dim), dtype=torch.float32, device=self.config.device)

        robot_offset = self.config.num_humanoids
        yaws = self._robot_yaws()
        obs_rows: list[np.ndarray] = []
        for robot_idx in range(self.config.num_robots):
            agent_id = robot_offset + robot_idx
            pos = positions[agent_id]
            vel = velocities[agent_id]
            yaw = yaws[robot_idx]

            goal_local = self._world_vec_to_local(self.goals_xy[agent_id] - pos, yaw)
            vel_local = self._world_vec_to_local(vel, yaw)
            goal_norm = max(float(self.config.max_start_goal_distance), 1e-4)
            goal_dist = float(np.linalg.norm(self.goals_xy[agent_id] - pos))
            row = [
                goal_local[0] / goal_norm,
                goal_local[1] / goal_norm,
                vel_local[0] / max(self.config.rl_max_linear_velocity, 1e-4),
                vel_local[1] / max(self.config.rl_max_linear_velocity, 1e-4),
                math.sin(yaw),
                math.cos(yaw),
                goal_dist / goal_norm,
                float(self._robot_episode_steps[robot_idx]) / max(float(self.config.rl_max_episode_steps), 1.0),
            ]

            row.extend(self._robot_rl_neighbor_observation(agent_id, pos, vel, yaw, positions, velocities))

            if self.config.rl_map_size > 0:
                row.extend(self._local_obstacle_patch(pos, yaw).reshape(-1).tolist())

            obs_rows.append(np.asarray(row, dtype=np.float32))

        obs = np.stack(obs_rows, axis=0)
        return torch.as_tensor(obs, dtype=torch.float32, device=self.config.device)

    def _robot_rl_neighbor_observation(
        self,
        agent_id: int,
        pos: np.ndarray,
        vel: np.ndarray,
        yaw: float,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> list[float]:
        relpos = positions - pos
        reldis = np.linalg.norm(relpos, axis=1)
        valid = (np.arange(self.num_agents) != agent_id) & (reldis <= self.config.neighbor_radius)
        neighbor_ids = np.argsort(np.where(valid, reldis, np.inf))[: self.config.rl_num_neighbors]

        values: list[float] = []
        used = 0
        for neighbor_id in neighbor_ids:
            if not np.isfinite(reldis[neighbor_id]):
                continue
            local_rel = self._world_vec_to_local(relpos[neighbor_id], yaw)
            local_vel = self._world_vec_to_local(velocities[neighbor_id] - vel, yaw)
            values.extend(
                [
                    local_rel[0] / max(self.config.neighbor_radius, 1e-4),
                    local_rel[1] / max(self.config.neighbor_radius, 1e-4),
                    local_vel[0] / max(self.config.rl_max_linear_velocity, 1e-4),
                    local_vel[1] / max(self.config.rl_max_linear_velocity, 1e-4),
                ]
            )
            used += 1

        for _ in range(self.config.rl_num_neighbors - used):
            values.extend([0.0, 0.0, 0.0, 0.0])
        return values

    def _local_obstacle_patch(self, xy: np.ndarray, yaw: float) -> np.ndarray:
        size = max(0, int(self.config.rl_map_size))
        if size <= 0:
            return np.zeros((0, 0), dtype=np.float32)
        extent = max(float(self.config.rl_map_extent), self.config.map_resolution)
        half = 0.5 * extent
        cell = extent / float(size)
        coords = (np.arange(size, dtype=np.float32) + 0.5) * cell - half
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        lateral = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        origin_x, origin_y = self.config.map_origin_xy
        patch = np.ones((size, size), dtype=np.float32)

        for row, local_x in enumerate(coords[::-1]):
            for col, local_y in enumerate(coords):
                world = xy + forward * local_x + lateral * local_y
                pixel_x = int(round((float(world[0]) - origin_x) / self.config.map_resolution))
                pixel_y = int(round((self.height - 1) - (float(world[1]) - origin_y) / self.config.map_resolution))
                if 0 <= pixel_y < self.height and 0 <= pixel_x < self.width:
                    patch[row, col] = float(self.obstacle_map[pixel_y, pixel_x] > 0)
        return patch

    def _robot_yaws(self) -> np.ndarray:
        if self.config.num_robots == 0:
            return np.zeros(0, dtype=np.float32)
        quats = self.robot.data.root_quat_w[: self.config.num_robots].detach().cpu().numpy()
        return np.asarray([self._yaw_from_quat_wxyz(quat) for quat in quats], dtype=np.float32)

    @staticmethod
    def _world_vec_to_local(vec: np.ndarray, yaw: float) -> np.ndarray:
        heading = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        lateral = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        return np.array([float(np.dot(vec, heading)), float(np.dot(vec, lateral))], dtype=np.float32)
