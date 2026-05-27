"""Navigation helpers for CrowdSim humanoid/robot scenes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from CrowdSim.differential_control import DifferentialDriveConfig, ManualDifferentialController
from CrowdSim.nav_task import (
    NavigationTask,
    NavigationTaskConfig,
    build_agent_marker_prototypes,
)
from CrowdSim.plan.sfm import Social_Force
from CrowdSim.robot_rl_navigation import RobotRLNavigationMixin


@dataclass
class CrowdNavigationConfig:
    map_path: Path
    map_resolution: float
    free_threshold: int
    num_humanoids: int
    num_robots: int
    device: torch.device
    map_origin_xy: tuple[float, float] = (0.0, 0.0)
    seed: int = 7
    agent_radius: float = 0.35
    safe_distance: float = 0.75
    max_speed: float = 1.5
    waypoint_tolerance: float = 0.45
    goal_tolerance: float = 0.75
    min_start_goal_distance: float = 5.0
    max_start_goal_distance: float = 10.0
    min_spawn_spacing: float = 1.2
    planning_step_size: float = 0.5
    planning_clearance: float = 0.2
    neighbor_radius: float = 4.0
    collision_distance: float = 0.75
    log_interval: int = 120
    update_hz: float = 30.0
    trajectory_recording_enabled: bool = True
    trajectory_output_dir: Path = Path("output/crowdsim_navigation")
    differential_drive: DifferentialDriveConfig = field(default_factory=DifferentialDriveConfig)
    visual_markers_enabled: bool = False
    humanoid_target_enabled: bool = True
    local_target_timestep: float = 1.0
    humanoid_target_min_heading_speed: float = 0.05
    car_rl_policy: bool = True
    car_drive_mode: str = "wheel"
    rl_num_neighbors: int = 4
    rl_max_linear_velocity: float = 2.0
    rl_max_angular_velocity: float = 2.0
    rl_progress_reward_scale: float = 4.0
    rl_goal_reward: float = 10.0
    rl_collision_penalty: float = -10.0
    rl_timeout_penalty: float = -5.0
    rl_time_penalty: float = -0.01
    rl_max_episode_steps: int = 600
    rl_map_size: int = 24
    rl_map_extent: float = 8.0


class CrowdNavigationManager(RobotRLNavigationMixin):
    """Plan and control CrowdSim agents in the shared Office map.

    Humanoids are still controlled by the loaded MaskedMimic policy. This manager
    samples their starts/goals and monitors their progress/collisions. Navigation
    wheel commands are applied only to CrowdRobot/Jetbot agents.
    """

    def __init__(self, config: CrowdNavigationConfig) -> None:
        self.config = config
        self.num_agents = config.num_humanoids + config.num_robots
        self.task = NavigationTask(
            NavigationTaskConfig(
                map_path=config.map_path,
                map_resolution=config.map_resolution,
                free_threshold=config.free_threshold,
                num_agents=self.num_agents,
                map_origin_xy=config.map_origin_xy,
                seed=config.seed,
                min_start_goal_distance=config.min_start_goal_distance,
                max_start_goal_distance=config.max_start_goal_distance,
                min_spawn_spacing=config.min_spawn_spacing,
                planning_step_size=config.planning_step_size,
                planning_clearance=config.planning_clearance,
            )
        )

        self.free_mask = self.task.free_mask
        self.obstacle_map = self.task.obstacle_map
        self.height = self.task.height
        self.width = self.task.width
        self.pixels_per_meter = 1.0 / config.map_resolution
        self.starts_px = self.task.starts_px
        self.goals_px = self.task.goals_px
        self.starts_xy = self.task.starts_xy
        self.goals_xy = self.task.goals_xy
        self.paths_xy = self.task.paths_xy
        self.waypoint_ids = np.ones(self.num_agents, dtype=np.int64)
        self.reached = np.zeros(self.num_agents, dtype=bool)
        self.collision_pairs: set[tuple[int, int]] = set()
        self.env_step_count = 0
        self.step_count = 0
        self._env_dt = 1.0 / 30.0
        self._update_interval_steps = self._compute_update_interval_steps(self._env_dt)
        self._last_positions = self.starts_xy.copy()
        self._has_completed_initial_reset = False
        self._wheel_targets_tensor: torch.Tensor | None = None
        self._wheel_joint_ids: list[int] | None = None
        self._wheel_controller = ManualDifferentialController(config.differential_drive)
        self._sfm_waypoints = self.starts_xy.copy().astype(np.float32)
        self._humanoid_sfm_waypoints = self.starts_xy[: config.num_humanoids].copy().astype(np.float32)
        self._robot_sfm_waypoints = self.starts_xy[
            config.num_humanoids : config.num_humanoids + config.num_robots
        ].copy().astype(np.float32)
        self._robot_progress_targets = self._robot_sfm_waypoints.copy().astype(np.float32)
        self._robot_prev_progress_dist = np.zeros(config.num_robots, dtype=np.float32)
        self._sfm_desired_velocities = np.zeros((self.num_agents, 2), dtype=np.float32)
        self._sfm_interact_forces = np.zeros((self.num_agents, 2), dtype=np.float32)
        self._sfm_repulsive_forces = np.zeros((self.num_agents, 2), dtype=np.float32)
        self._sfm_d_vel = np.zeros((self.num_agents, 2), dtype=np.float32)
        self._humanoid_target_yaws = np.zeros(config.num_humanoids, dtype=np.float32)
        self._humanoid_yaw_source = np.full(
            config.num_humanoids,
            "initial",
            dtype=object,
        )
        self._pending_path_updates: list[dict[str, object]] = []
        self._path_log_dirty = False
        self._pending_humanoid_reset_reasons: dict[int, str] = {}
        self._printed_masked_mimic_target_warning = False
        self._local_target_marker = None
        self.path_log_path = self._write_navigation_path_log()
        self.trajectory_log_path = self._open_trajectory_log()

        self.sfm_controller = self._make_sfm_controller(config.agent_radius)

        self._robot_rl_actions = np.zeros((config.num_robots, 2), dtype=np.float32)
        self._robot_episode_steps = np.zeros(config.num_robots, dtype=np.int64)
        self._robot_last_obs: torch.Tensor | None = None
        self._robot_last_rewards = torch.zeros(config.num_robots, device=config.device)
        self._robot_last_dones = torch.zeros(config.num_robots, dtype=torch.bool, device=config.device)
        self._robot_last_info: dict[str, torch.Tensor] = {}

    @property
    def humanoid_starts_xy(self) -> torch.Tensor:
        values = self.starts_xy[: self.config.num_humanoids]
        return torch.tensor(values, dtype=torch.float32, device=self.config.device)

    @property
    def robot_starts_xy_yaw(self) -> torch.Tensor:
        start = self.config.num_humanoids
        values = np.zeros((self.config.num_robots, 3), dtype=np.float32)
        values[:, :2] = self.starts_xy[start : start + self.config.num_robots]
        values[:, 2] = self._initial_robot_yaws()
        return torch.tensor(values, dtype=torch.float32, device=self.config.device)

    def attach(self, env) -> None:
        self.env = env
        self.robot = getattr(env, "crowdsim_robot", None)
        self._env_dt = self._read_env_dt(env)
        self._update_interval_steps = self._compute_update_interval_steps(self._env_dt)
        self._refresh_controller_dt()
        if self.config.num_robots > 0 and self.robot is None:
            raise RuntimeError("Navigation requested robot control, but crowdsim_robot is missing.")
        if self.config.num_robots > 0:
            available_joint_names = list(
                getattr(self.robot, "joint_names", None)
                or getattr(self.robot.data, "joint_names", [])
                or []
            )
            try:
                left_ids, left_names = self.robot.find_joints(
                    list(self.config.differential_drive.left_wheel_joint_names),
                    preserve_order=True,
                )
                right_ids, right_names = self.robot.find_joints(
                    list(self.config.differential_drive.right_wheel_joint_names),
                    preserve_order=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to resolve fixed robot wheel joints. "
                    f"Expected left={self.config.differential_drive.left_wheel_joint_names}, "
                    f"right={self.config.differential_drive.right_wheel_joint_names}. "
                    f"Available joints: {available_joint_names}"
                ) from exc
            if not left_ids or not right_ids:
                raise RuntimeError(
                    "Failed to resolve fixed robot wheel joints. "
                    f"Expected left={self.config.differential_drive.left_wheel_joint_names}, "
                    f"right={self.config.differential_drive.right_wheel_joint_names}. "
                    f"Available joints: {available_joint_names}"
                )
            if len(left_ids) != 1 or len(right_ids) != 1:
                raise RuntimeError(
                    "Expected exactly one left wheel joint and one right wheel joint for "
                    "IsaacLab Articulation velocity control. "
                    f"Resolved left={list(zip(left_ids, left_names))}, "
                    f"right={list(zip(right_ids, right_names))}."
                )

            self._wheel_joint_ids = [int(left_ids[0]), int(right_ids[0])]
            self._wheel_targets_tensor = torch.zeros(
                self.config.num_robots,
                2,
                dtype=torch.float32,
                device=self.config.device,
            )
            print(
                "[CrowdSim] Differential wheel joints: "
                f"left={list(zip(left_ids, left_names))}, "
                f"right={list(zip(right_ids, right_names))}"
            )

        import types

        original_reset = env.reset
        original_step = env.step

        def reset_with_navigation(env_self, *args, **kwargs):
            result = original_reset(*args, **kwargs)
            env_ids = self._reset_env_ids_from_args(args, kwargs)
            if len(env_ids) > 0:
                initial_full_reset = (
                    not self._has_completed_initial_reset
                    and len(env_ids) == self.config.num_humanoids
                    and np.array_equal(env_ids, np.arange(self.config.num_humanoids))
                )
                if initial_full_reset:
                    self._has_completed_initial_reset = True
                else:
                    self._reset_navigation_agents(env_ids)
                positions, velocities = self._read_agent_state()
                self._compute_sfm_reference_waypoints(positions, velocities)
                self._update_local_target_markers()
                self._last_positions = positions
                self._record_trajectory_frame(positions, velocities)
            return result

        def step_with_navigation(env_self, action):
            should_update_navigation = self._should_update_navigation()
            if should_update_navigation:
                self.pre_step()
            result = original_step(action)
            self.env_step_count += 1
            self._apply_robot_drive_after_env_step()
            if should_update_navigation:
                self.post_step()
            else:
                self._clear_robot_rl_env_step_feedback()
            return result

        env.reset = types.MethodType(reset_with_navigation, env)
        env.step = types.MethodType(step_with_navigation, env)
        env.crowdsim_navigation = self
        self._attach_masked_mimic_navigation_targets(env)
        self.task.create_visualization_markers(
            num_humanoids=self.config.num_humanoids,
            enabled=self.config.visual_markers_enabled
            and not getattr(env.simulator, "headless", True),
        )
        self._create_local_target_markers(
            enabled=self.config.visual_markers_enabled
            and not getattr(env.simulator, "headless", True)
        )
        print(
            f"[CrowdSim] Navigation enabled: {self.config.num_humanoids} humanoid(s), "
            f"{self.config.num_robots} robot(s), car_rl_policy={self.config.car_rl_policy}, "
            f"car_drive_mode={self.config.car_drive_mode}."
        )
        print(
            "[CrowdSim] Navigation update rate: "
            f"{self._navigation_update_hz():.3g} Hz "
            f"(env_dt={self._env_dt:.6f}s, every {self._update_interval_steps} env step(s))."
        )
        if self.config.humanoid_target_enabled:
            print(
                "[CrowdSim] Humanoid navigation targets enabled: "
                "controller=sfm, "
                "format=Pelvis translation + Pelvis rotation."
            )
        print(f"[CrowdSim] Navigation path log: {self.path_log_path}")
        if self._trajectory_log_file is not None:
            print(f"[CrowdSim] Navigation trajectory log: {self.trajectory_log_path}")

    def pre_step(self) -> None:
        positions, velocities = self._read_agent_state()
        self._compute_sfm_reference_waypoints(positions, velocities)
        self._update_robot_progress_targets(positions)
        self._update_local_target_markers()

        if self.config.num_robots == 0:
            return

        self._override_robot_rl_actions_with_constant_command()
        if self.config.car_drive_mode == "kinematic":
            self._clear_robot_wheel_targets()
        elif self.config.car_drive_mode == "wheel":
            wheel_targets = self._compute_robot_wheel_targets()
            self._write_robot_wheel_targets(wheel_targets)
        else:
            raise ValueError(
                f"Unsupported car.drive_mode='{self.config.car_drive_mode}'. "
                "Expected 'wheel' or 'kinematic'."
            )

    def post_step(self) -> None:
        self.step_count += 1
        positions, velocities = self._read_agent_state()
        self._update_waypoints_and_goals(positions)
        new_pairs = self._detect_collisions(positions)
        nav_done_ids = self._navigation_done_agent_ids(new_pairs)
        self._request_humanoid_resets(nav_done_ids)
        self._update_robot_rl_feedback(positions, velocities, new_pairs)
        self._print_robot_drive_debug()
        self._last_positions = positions
        self.env.extras["crowdsim_navigation"] = {
            "reached": int(self.reached.sum()),
            "num_agents": self.num_agents,
            "collision_pairs": len(self.collision_pairs),
            "new_collision_pairs": len(new_pairs),
            "navigation_done_agents": len(nav_done_ids),
            "update_hz": self._navigation_update_hz(),
        }
        if self.config.num_robots > 0:
            robot_sfm_distances = np.linalg.norm(
                self._robot_sfm_waypoints - positions[self.config.num_humanoids :],
                axis=1,
            )
            self.env.extras["crowdsim_navigation"]["robot_sfm_target_distance_mean"] = float(
                robot_sfm_distances.mean()
            )

        if self.config.log_interval > 0 and self.step_count % self.config.log_interval == 0:
            print(
                f"[CrowdSim] nav step={self.step_count}, reached="
                f"{int(self.reached.sum())}/{self.num_agents}, "
                f"collision_pairs={len(self.collision_pairs)}"
            )
        if new_pairs:
            print(f"[CrowdSim] Collision warning: {sorted(new_pairs)}")
        self._record_trajectory_frame(positions, velocities)

    def _initial_robot_yaws(self) -> np.ndarray:
        yaws = np.zeros(self.config.num_robots, dtype=np.float32)
        offset = self.config.num_humanoids
        for idx in range(self.config.num_robots):
            path = self.paths_xy[offset + idx]
            if len(path) > 1:
                delta = path[1] - path[0]
                yaws[idx] = math.atan2(float(delta[1]), float(delta[0]))
        return yaws

    def _write_navigation_path_log(self) -> Path:
        output_dir = Path("output/crowdsim_navigation")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        latest_path = output_dir / "paths_latest.json"
        timestamp_path = output_dir / f"paths_{timestamp}.json"

        records = [self._navigation_path_record(agent_id) for agent_id in range(self.num_agents)]
        self._path_log_dirty = False
        self.path_log_path = latest_path

        payload = {
            "created_at": timestamp,
            "map_path": str(self.config.map_path),
            "map_resolution": self.config.map_resolution,
            "map_origin_xy": list(self.config.map_origin_xy),
            "free_threshold": self.config.free_threshold,
            "planning_step_size": self.config.planning_step_size,
            "planning_clearance": self.config.planning_clearance,
            "navigation_update_hz": self._navigation_update_hz(),
            "local_target_timestep": self.config.local_target_timestep,
            "num_humanoids": self.config.num_humanoids,
            "num_cars": self.config.num_robots,
            "agents": records,
        }
        text = json.dumps(payload, indent=2)
        latest_path.write_text(text, encoding="utf-8")
        timestamp_path.write_text(text, encoding="utf-8")
        return latest_path

    def _open_trajectory_log(self) -> Path | None:
        self._trajectory_log_file = None
        self._trajectory_timestamp_file = None
        if not self.config.trajectory_recording_enabled:
            return None

        output_dir = Path(self.config.trajectory_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        latest_path = output_dir / "trajectory_latest.jsonl"
        timestamp_path = output_dir / f"trajectory_{timestamp}.jsonl"
        self._trajectory_log_file = latest_path.open("w", encoding="utf-8")
        self._trajectory_timestamp_file = timestamp_path.open("w", encoding="utf-8")
        metadata = {
            "type": "metadata",
            "created_at": timestamp,
            "path_log": str(self.path_log_path),
            "timestamp_path": str(timestamp_path),
            "map_path": str(self.config.map_path),
            "map_resolution": self.config.map_resolution,
            "map_origin_xy": list(self.config.map_origin_xy),
            "num_humanoids": self.config.num_humanoids,
            "num_cars": self.config.num_robots,
            "num_agents": self.num_agents,
            "navigation_update_hz": self._navigation_update_hz(),
            "local_target_timestep": self.config.local_target_timestep,
        }
        line = json.dumps(metadata) + "\n"
        self._trajectory_log_file.write(line)
        self._trajectory_timestamp_file.write(line)
        self._trajectory_log_file.flush()
        self._trajectory_timestamp_file.flush()
        return latest_path

    def _navigation_path_record(self, agent_id: int) -> dict[str, object]:
        agent_type = "humanoid" if agent_id < self.config.num_humanoids else "car"
        local_id = agent_id if agent_type == "humanoid" else agent_id - self.config.num_humanoids
        return {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "local_id": local_id,
            "start_xy": self.starts_xy[agent_id].astype(float).tolist(),
            "goal_xy": self.goals_xy[agent_id].astype(float).tolist(),
            "path_xy": self.paths_xy[agent_id].astype(float).tolist(),
        }

    def _consume_pending_path_updates(self) -> list[dict[str, object]]:
        if not self._pending_path_updates:
            return []
        updates = self._pending_path_updates
        self._pending_path_updates = []
        return updates

    def _flush_navigation_path_log_if_dirty(self) -> None:
        if not self._path_log_dirty:
            return
        self._write_navigation_path_log()

    def _record_trajectory_frame(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        if self._trajectory_log_file is None:
            return

        current_waypoints = np.asarray(
            [self._current_waypoint(agent_id) for agent_id in range(self.num_agents)],
            dtype=np.float32,
        )
        frame = {
            "type": "frame",
            "step": int(self.step_count),
            "env_step": int(self.env_step_count),
            "time": float(self.step_count * self._dt()),
            "positions_xy": positions.astype(float).tolist(),
            "velocities_xy": velocities.astype(float).tolist(),
            "current_waypoints_xy": current_waypoints.astype(float).tolist(),
            "goals_xy": self.goals_xy.astype(float).tolist(),
            "waypoint_ids": self.waypoint_ids.astype(int).tolist(),
            "local_targets_xy": self._sfm_waypoints.astype(float).tolist(),
            "sfm_desired_velocities_xy": self._sfm_desired_velocities.astype(float).tolist(),
            "sfm_interact_forces_xy": self._sfm_interact_forces.astype(float).tolist(),
            "sfm_repulsive_forces_xy": self._sfm_repulsive_forces.astype(float).tolist(),
            "sfm_d_vel_xy": self._sfm_d_vel.astype(float).tolist(),
            "humanoid_yaw_source": self._humanoid_yaw_source.astype(str).tolist(),
            "reached": self.reached.astype(bool).tolist(),
            "collision_pairs": [list(pair) for pair in sorted(self.collision_pairs)],
        }
        path_updates = self._consume_pending_path_updates()
        if path_updates:
            frame["path_updates"] = path_updates
        line = json.dumps(frame) + "\n"
        self._trajectory_log_file.write(line)
        if self._trajectory_timestamp_file is not None:
            self._trajectory_timestamp_file.write(line)
            self._trajectory_timestamp_file.flush()
        self._trajectory_log_file.flush()

    def _read_agent_state(self) -> tuple[np.ndarray, np.ndarray]:
        humanoid_state = self.env.simulator.get_root_state()
        humanoid_pos = humanoid_state.root_pos[:, :2].detach().cpu().numpy()
        humanoid_vel = humanoid_state.root_vel[:, :2].detach().cpu().numpy()

        if self.config.num_robots == 0:
            return humanoid_pos, humanoid_vel

        robot_pos = self.robot.data.root_pos_w[:, :2].detach().cpu().numpy()
        robot_vel = self.robot.data.root_lin_vel_w[:, :2].detach().cpu().numpy()
        return (
            np.concatenate([humanoid_pos, robot_pos], axis=0),
            np.concatenate([humanoid_vel, robot_vel], axis=0),
        )

    def _update_robot_progress_targets(self, positions: np.ndarray) -> None:
        if self.config.num_robots == 0:
            return
        robot_positions = positions[
            self.config.num_humanoids : self.config.num_humanoids + self.config.num_robots
        ]
        self._robot_progress_targets[:] = self._robot_sfm_waypoints
        self._robot_prev_progress_dist[:] = np.linalg.norm(
            robot_positions - self._robot_progress_targets,
            axis=1,
        ).astype(np.float32)

    def _compute_sfm_reference_waypoints(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> None:
        self._sfm_waypoints[: self.num_agents] = positions[: self.num_agents]
        target_timestep = max(float(self.config.local_target_timestep), 1e-5)
        self._sfm_desired_velocities.fill(0.0)
        self._sfm_interact_forces.fill(0.0)
        self._sfm_repulsive_forces.fill(0.0)
        self._sfm_d_vel.fill(0.0)
        if self.config.num_humanoids > 0:
            self._humanoid_sfm_waypoints[:] = positions[: self.config.num_humanoids]
        if self.config.num_robots > 0:
            self._robot_sfm_waypoints[:] = positions[
                self.config.num_humanoids : self.config.num_humanoids + self.config.num_robots
            ]

        for agent_id in range(self.num_agents):
            if self.reached[agent_id]:
                if agent_id < self.config.num_humanoids:
                    self._humanoid_yaw_source[agent_id] = "reached"
                continue

            pos = positions[agent_id]
            vel = velocities[agent_id]
            goal = self._current_waypoint(agent_id)
            nbr_state = self._neighbor_state(agent_id, positions, velocities)
            cord_int = self._clip_pixel_yx(self.world_to_pixel(pos))
            desired_vel, force_terms = self.sfm_controller.get_action(
                (pos, cord_int, vel, goal),
                (nbr_state[0], nbr_state[1], nbr_state[2]),
            )
            desired_vel = np.asarray(desired_vel, dtype=np.float32)
            if not np.all(np.isfinite(desired_vel)):
                desired_vel = np.zeros(2, dtype=np.float32)
            self._sfm_desired_velocities[agent_id] = desired_vel
            self._sfm_interact_forces[agent_id] = np.asarray(force_terms[0], dtype=np.float32)
            self._sfm_repulsive_forces[agent_id] = np.asarray(force_terms[1], dtype=np.float32)
            self._sfm_d_vel[agent_id] = np.asarray(force_terms[2], dtype=np.float32)
            sfm_waypoint = (pos + desired_vel * target_timestep).astype(np.float32)
            self._sfm_waypoints[agent_id] = sfm_waypoint

            if agent_id >= self.config.num_humanoids:
                robot_idx = agent_id - self.config.num_humanoids
                self._robot_sfm_waypoints[robot_idx] = sfm_waypoint
                continue

            self._humanoid_sfm_waypoints[agent_id] = sfm_waypoint

            heading_delta = sfm_waypoint - pos
            if np.linalg.norm(heading_delta) >= self.config.humanoid_target_min_heading_speed:
                self._humanoid_target_yaws[agent_id] = math.atan2(
                    float(heading_delta[1]), float(heading_delta[0])
                )
                self._humanoid_yaw_source[agent_id] = "sfm_target"
            else:
                fallback = self._waypoint_desired_velocity(pos, goal)
                if np.linalg.norm(fallback) >= 1e-5:
                    self._humanoid_target_yaws[agent_id] = math.atan2(
                        float(fallback[1]), float(fallback[0])
                    )
                    self._humanoid_yaw_source[agent_id] = "waypoint_fallback"
                else:
                    self._humanoid_yaw_source[agent_id] = "previous"

    def _attach_masked_mimic_navigation_targets(self, env) -> None:
        if not self.config.humanoid_target_enabled or self.config.num_humanoids == 0:
            return
        control_manager = getattr(env, "control_manager", None)
        component = getattr(control_manager, "components", {}).get("masked_mimic")
        if component is None:
            print("[CrowdSim] Humanoid navigation targets skipped: masked_mimic control not found.")
            return

        import types

        original_populate_context = component.populate_context

        def populate_context_with_navigation_targets(component_self, ctx):
            original_populate_context(ctx)
            self._override_masked_mimic_context(component_self, ctx)

        component.populate_context = types.MethodType(populate_context_with_navigation_targets, component)

    def _override_masked_mimic_context(self, component, ctx) -> None:
        base = getattr(ctx, "masked_mimic", None)
        if base is None:
            return

        try:
            from protomotions.envs.context_views import MaskedMimicContext
        except ImportError:
            return

        conditionable_body_ids = getattr(component, "conditionable_body_ids", None)
        if conditionable_body_ids is None:
            self._warn_masked_mimic_target_once("conditionable_body_ids missing.")
            return

        pelvis_body_id = int(getattr(component.env.robot_config, "anchor_body_index", 0))
        pelvis_matches = (conditionable_body_ids == pelvis_body_id).nonzero(as_tuple=False)
        if pelvis_matches.numel() == 0:
            self._warn_masked_mimic_target_once(
                f"pelvis body id {pelvis_body_id} is not conditionable."
            )
            return

        num_envs, num_future_steps = base.ref_pos.shape[:2]
        num_humanoids = min(self.config.num_humanoids, num_envs)
        if num_humanoids <= 0:
            return

        device = base.ref_pos.device
        dtype = base.ref_pos.dtype
        offsets_np = self._humanoid_target_offsets(num_future_steps)
        offsets = torch.as_tensor(offsets_np, dtype=dtype, device=device)

        ref_pos = base.ref_pos.clone()
        ref_rot = base.ref_rot.clone()
        target_times = base.target_times.clone()
        time_offsets = base.time_offsets.clone()
        target_bodies_masks = torch.zeros_like(base.target_bodies_masks)
        target_poses_masks = torch.zeros_like(base.target_poses_masks)

        current_pelvis = ctx.current.rigid_body_pos[:num_humanoids, pelvis_body_id, :]
        xy_targets_np, target_yaws_np = self._humanoid_future_targets_from_path(
            current_pelvis[:, :2].detach().cpu().numpy(),
            offsets_np,
        )
        xy_targets = torch.as_tensor(xy_targets_np, dtype=dtype, device=device)
        yaws = torch.as_tensor(target_yaws_np, dtype=dtype, device=device)
        active = torch.as_tensor(
            ~self.reached[:num_humanoids],
            dtype=torch.bool,
            device=device,
        )

        ref_pos[:num_humanoids, :, pelvis_body_id, :2] = xy_targets
        ref_pos[:num_humanoids, :, pelvis_body_id, 2] = current_pelvis[:, None, 2]

        yaw_quat = self._yaw_to_quat_xyzw_tensor(yaws.reshape(-1)).to(dtype=dtype)
        ref_rot[:num_humanoids, :, pelvis_body_id, :] = yaw_quat.view(
            num_humanoids, num_future_steps, 4
        )

        pelvis_condition_index = int(pelvis_matches[0].item())
        masks = target_bodies_masks.view(
            num_envs,
            num_future_steps,
            int(getattr(component, "num_conditionable_bodies")),
            2,
        )
        masks[:num_humanoids, :, pelvis_condition_index, 0] = active[:, None]
        masks[:num_humanoids, :, pelvis_condition_index, 1] = active[:, None]
        target_poses_masks[:num_humanoids, :] = active[:, None]

        if hasattr(component.env, "motion_manager"):
            motion_times = component.env.motion_manager.motion_times.to(device=device, dtype=dtype)
            target_times[:num_humanoids, :] = motion_times[:num_humanoids, None] + offsets[None, :]
        else:
            target_times[:num_humanoids, :] = offsets[None, :]
        time_offsets[:num_humanoids, :] = offsets[None, :]

        ctx.masked_mimic = MaskedMimicContext(
            mimic=base.mimic,
            ref_pos=ref_pos,
            ref_rot=ref_rot,
            target_times=target_times,
            time_offsets=time_offsets,
            target_poses_masks=target_poses_masks,
            target_bodies_masks=target_bodies_masks,
        )

    def _warn_masked_mimic_target_once(self, reason: str) -> None:
        if self._printed_masked_mimic_target_warning:
            return
        self._printed_masked_mimic_target_warning = True
        print(f"[CrowdSim] Humanoid navigation targets disabled for this run: {reason}")

    def _create_local_target_markers(self, enabled: bool) -> None:
        if not enabled:
            return

        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self._local_target_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/CrowdSim/local_targets",
                markers=build_agent_marker_prototypes(
                    sim_utils,
                    num_humanoids=self.config.num_humanoids,
                    num_robots=self.config.num_robots,
                ),
            )
        )

    def _update_local_target_markers(self) -> None:
        if self._local_target_marker is None or self.num_agents == 0:
            return
        translations = np.zeros((self.num_agents, 3), dtype=np.float32)
        translations[:, :2] = self._sfm_waypoints[: self.num_agents]
        translations[:, 2] = self._agent_root_heights()
        orientations = np.zeros((self.num_agents, 4), dtype=np.float32)
        orientations[:, 0] = 1.0
        scales = np.asarray(
            [
                [0.16, 0.16, 0.16]
                if agent_id < self.config.num_humanoids
                else [0.20, 0.20, 0.08]
                for agent_id in range(self.num_agents)
            ],
            dtype=np.float32,
        )
        marker_indices = np.arange(self.num_agents, dtype=np.int32)
        self._local_target_marker.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
            marker_indices=marker_indices,
        )

    def _agent_root_heights(self) -> np.ndarray:
        heights = np.full(self.num_agents, 0.16, dtype=np.float32)
        if self.config.num_humanoids > 0:
            humanoid_state = self.env.simulator.get_root_state()
            heights[: self.config.num_humanoids] = (
                humanoid_state.root_pos[:, 2].detach().cpu().numpy().astype(np.float32)
            )
        if self.config.num_robots > 0 and self.robot is not None:
            start = self.config.num_humanoids
            heights[start : start + self.config.num_robots] = (
                self.robot.data.root_pos_w[:, 2].detach().cpu().numpy().astype(np.float32)
            )
        return heights

    def _neighbor_state(
        self, agent_id: int, positions: np.ndarray, velocities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        relpos = positions - positions[agent_id]
        reldis = np.linalg.norm(relpos, axis=1)
        mask = (np.arange(self.num_agents) != agent_id) & (reldis < self.config.neighbor_radius)
        nbrs_idx = np.nonzero(mask)[0]
        return (
            nbrs_idx,
            reldis[nbrs_idx],
            relpos[nbrs_idx],
            velocities[agent_id] - velocities[nbrs_idx],
        )

    def _update_waypoints_and_goals(self, positions: np.ndarray) -> None:
        for agent_id, pos in enumerate(positions):
            path = self.paths_xy[agent_id]
            if self.reached[agent_id]:
                continue
            while self.waypoint_ids[agent_id] < len(path) - 1:
                waypoint = path[self.waypoint_ids[agent_id]]
                if np.linalg.norm(pos - waypoint) > self.config.waypoint_tolerance:
                    break
                self.waypoint_ids[agent_id] += 1
            if np.linalg.norm(pos - self.goals_xy[agent_id]) <= self.config.goal_tolerance:
                self.reached[agent_id] = True

    def _navigation_done_agent_ids(self, new_pairs: set[tuple[int, int]]) -> np.ndarray:
        done: set[int] = set(int(agent_id) for agent_id in np.nonzero(self.reached)[0])
        for pair in new_pairs:
            done.update(int(agent_id) for agent_id in pair)
        return np.asarray(sorted(done), dtype=np.int64)

    def _request_humanoid_resets(self, agent_ids: np.ndarray) -> None:
        humanoid_ids = agent_ids[(0 <= agent_ids) & (agent_ids < self.config.num_humanoids)]
        if len(humanoid_ids) == 0:
            return
        for agent_id in humanoid_ids:
            self._pending_humanoid_reset_reasons[int(agent_id)] = self._humanoid_reset_reason(
                int(agent_id)
            )
        ids = torch.as_tensor(humanoid_ids, dtype=torch.long, device=self.config.device)
        self.env.reset_buf[ids] = True

    def _humanoid_reset_reason(self, agent_id: int) -> str:
        reasons = []
        if bool(self.reached[agent_id]):
            reasons.append("reached")
        if any(agent_id in pair for pair in self.collision_pairs):
            reasons.append("collision")
        return "+".join(reasons) if reasons else "external"

    def _detect_collisions(self, positions: np.ndarray) -> set[tuple[int, int]]:
        new_pairs: set[tuple[int, int]] = set()
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                distance = float(np.linalg.norm(positions[i] - positions[j]))
                if distance < self.config.collision_distance:
                    pair = (i, j)
                    if pair not in self.collision_pairs:
                        new_pairs.add(pair)
                    self.collision_pairs.add(pair)
        return new_pairs

    def _reset_env_ids_from_args(self, args: tuple, kwargs: dict) -> np.ndarray:
        if "env_ids" in kwargs:
            env_ids = kwargs["env_ids"]
            if env_ids is None:
                return np.arange(self.config.num_humanoids, dtype=np.int64)
            if isinstance(env_ids, torch.Tensor):
                return np.atleast_1d(
                    env_ids.detach().cpu().numpy().astype(np.int64, copy=False)
                )
            return np.atleast_1d(np.asarray(env_ids, dtype=np.int64))

        if not args or args[0] is None:
            return np.arange(self.config.num_humanoids, dtype=np.int64)

        env_ids = args[0]
        if isinstance(env_ids, torch.Tensor):
            return np.atleast_1d(
                env_ids.detach().cpu().numpy().astype(np.int64, copy=False)
            )
        return np.atleast_1d(np.asarray(env_ids, dtype=np.int64))

    def _reset_navigation_agents(self, env_ids: np.ndarray) -> None:
        if len(env_ids) == 0:
            return

        humanoid_ids = env_ids[
            (0 <= env_ids) & (env_ids < self.config.num_humanoids)
        ]
        if len(humanoid_ids) > 0:
            positions, _ = self._read_agent_state()
            for agent_id in humanoid_ids:
                reason = self._pending_humanoid_reset_reasons.pop(
                    int(agent_id),
                    "external",
                )
                self._replan_agent_from_xy(int(agent_id), positions[int(agent_id)])
                self._humanoid_yaw_source[int(agent_id)] = "reset"
                print(
                    "[CrowdSim] reset humanoid "
                    f"env={int(agent_id)} agent={int(agent_id)} reason={reason}"
                )
            self._flush_navigation_path_log_if_dirty()

    def _replan_agent_from_xy(self, agent_id: int, start_xy: np.ndarray) -> None:
        start_px, goal_px, path_xy = self.task.sample_goal_and_plan_path(start_xy)
        self.starts_px[agent_id] = start_px
        self.goals_px[agent_id] = goal_px
        self.starts_xy[agent_id] = self.task.pixel_to_world(start_px)
        self.goals_xy[agent_id] = self.task.pixel_to_world(goal_px)
        self.paths_xy[agent_id] = path_xy
        self.task.starts_px[agent_id] = start_px
        self.task.goals_px[agent_id] = goal_px
        self.task.starts_xy[agent_id] = self.starts_xy[agent_id]
        self.task.goals_xy[agent_id] = self.goals_xy[agent_id]
        self.task.paths_xy[agent_id] = path_xy
        self.waypoint_ids[agent_id] = 1 if len(path_xy) > 1 else 0
        self.reached[agent_id] = False
        self.collision_pairs = {
            pair for pair in self.collision_pairs if agent_id not in pair
        }
        self.task.refresh_visualization_markers()
        self._pending_path_updates.append(self._navigation_path_record(agent_id))
        self._path_log_dirty = True

    def _current_waypoint(self, agent_id: int) -> np.ndarray:
        path = self.paths_xy[agent_id]
        idx = min(int(self.waypoint_ids[agent_id]), len(path) - 1)
        return path[idx]

    def world_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        return self.task.world_to_pixel(xy)

    def pixel_to_world(self, pixel_yx: np.ndarray) -> np.ndarray:
        return self.task.pixel_to_world(pixel_yx)

    def _planner_cfg(self, radius: float) -> dict:
        return {
            "map": {
                "resolution": self.config.map_resolution,
                "resolution_viz": self.pixels_per_meter,
            },
            "env": {
                "dt": self._dt(),
                "safe_distance": self.config.safe_distance,
                "neighbor_radius": self.config.neighbor_radius,
                "reach_distance": self.config.waypoint_tolerance,
            },
            "agent": {"radius": radius, "max_vel": self.config.max_speed},
        }

    def _make_sfm_controller(self, radius: float) -> Social_Force:
        import cv2

        free_uint8 = self.free_mask.astype(np.uint8)
        distance_px = cv2.distanceTransform(free_uint8, cv2.DIST_L2, 5)
        return Social_Force(distance_px * self.config.map_resolution, self._planner_cfg(radius))

    def _waypoint_desired_velocity(self, pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
        delta = goal - pos
        distance = float(np.linalg.norm(delta))
        if distance < 1e-5:
            return np.zeros(2, dtype=np.float32)
        speed = min(self.config.max_speed, distance / max(self._dt(), 1e-5))
        return (delta / distance * speed).astype(np.float32)

    def _clip_pixel_yx(self, pixel_yx: np.ndarray) -> np.ndarray:
        return np.array(
            [
                int(np.clip(pixel_yx[0], 0, self.height - 1)),
                int(np.clip(pixel_yx[1], 0, self.width - 1)),
            ],
            dtype=np.int64,
        )

    def _humanoid_target_offsets(self, num_future_steps: int) -> np.ndarray:
        timestep = max(float(self.config.local_target_timestep), self._dt())
        return timestep * np.arange(1, num_future_steps + 1, dtype=np.float32)

    def _humanoid_future_targets_from_path(
        self, current_xy: np.ndarray, offsets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        num_humanoids = min(self.config.num_humanoids, current_xy.shape[0])
        num_steps = len(offsets)
        targets = np.zeros((num_humanoids, num_steps, 2), dtype=np.float32)
        yaws = np.zeros((num_humanoids, num_steps), dtype=np.float32)

        for agent_id in range(num_humanoids):
            pos = np.asarray(current_xy[agent_id], dtype=np.float32)
            sfm_target = self._humanoid_sfm_waypoints[agent_id]
            path = self._remaining_path_for_agent(agent_id, sfm_target)
            previous = pos
            last_yaw = float(self._humanoid_target_yaws[agent_id])

            for step_id, offset in enumerate(offsets):
                if step_id == 0:
                    target = sfm_target
                else:
                    remaining_time = max(float(offset - offsets[0]), 0.0)
                    lookahead_distance = float(self.config.max_speed) * remaining_time
                    target = self._sample_path_by_distance(path, lookahead_distance)

                targets[agent_id, step_id] = target
                direction = target - previous
                if np.linalg.norm(direction) < 1e-5:
                    direction = target - pos
                if np.linalg.norm(direction) >= 1e-5:
                    last_yaw = math.atan2(float(direction[1]), float(direction[0]))
                yaws[agent_id, step_id] = last_yaw
                previous = target

        return targets, yaws

    def _remaining_path_for_agent(self, agent_id: int, current_xy: np.ndarray) -> np.ndarray:
        path = self.paths_xy[agent_id]
        waypoint_id = min(max(int(self.waypoint_ids[agent_id]), 0), len(path) - 1)
        remaining = path[waypoint_id:]
        if len(remaining) == 0 or np.linalg.norm(remaining[-1] - path[-1]) > 1e-5:
            remaining = np.concatenate([remaining, path[-1:]], axis=0)
        return np.concatenate([current_xy[None, :], remaining], axis=0).astype(np.float32)

    @staticmethod
    def _sample_path_by_distance(path: np.ndarray, distance: float) -> np.ndarray:
        if len(path) == 0:
            return np.zeros(2, dtype=np.float32)
        if len(path) == 1 or distance <= 0.0:
            return path[0].astype(np.float32)

        remaining = float(distance)
        for idx in range(len(path) - 1):
            start = path[idx]
            end = path[idx + 1]
            segment = end - start
            segment_length = float(np.linalg.norm(segment))
            if segment_length < 1e-6:
                continue
            if remaining <= segment_length:
                alpha = remaining / segment_length
                return (start + alpha * segment).astype(np.float32)
            remaining -= segment_length
        return path[-1].astype(np.float32)

    def _dt(self) -> float:
        return self._update_interval_steps * self._env_dt

    def _should_update_navigation(self) -> bool:
        return self.env_step_count % self._update_interval_steps == 0

    def _navigation_update_hz(self) -> float:
        return 1.0 / max(self._dt(), 1e-6)

    def _compute_update_interval_steps(self, env_dt: float) -> int:
        update_period = 1.0 / max(float(self.config.update_hz), 1e-6)
        return max(1, int(round(update_period / max(env_dt, 1e-6))))

    def _refresh_controller_dt(self) -> None:
        if hasattr(self.sfm_controller, "dt"):
            self.sfm_controller.dt = self._dt()

    @staticmethod
    def _read_env_dt(env) -> float:
        env_dt = float(getattr(env, "dt", 0.0) or 0.0)
        if env_dt > 0.0:
            return env_dt
        simulator = getattr(env, "simulator", None)
        sim_dt = float(getattr(simulator, "dt", 0.0) or 0.0)
        if sim_dt > 0.0:
            return sim_dt
        return 1.0 / 30.0

    @staticmethod
    def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
        w, x, y, z = [float(v) for v in quat]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _yaw_to_quat_tensor(yaw: torch.Tensor) -> torch.Tensor:
        quat = torch.zeros((yaw.shape[0], 4), dtype=yaw.dtype, device=yaw.device)
        half_yaw = 0.5 * yaw
        quat[:, 0] = torch.cos(half_yaw)
        quat[:, 3] = torch.sin(half_yaw)
        return quat

    @staticmethod
    def _yaw_to_quat_xyzw_tensor(yaw: torch.Tensor) -> torch.Tensor:
        quat = torch.zeros((yaw.shape[0], 4), dtype=yaw.dtype, device=yaw.device)
        half_yaw = 0.5 * yaw
        quat[:, 2] = torch.sin(half_yaw)
        quat[:, 3] = torch.cos(half_yaw)
        return quat
