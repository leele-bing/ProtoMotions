"""Navigation helpers for CrowdSim humanoid/robot scenes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from CrowdSim.nav_task import NavigationTask, NavigationTaskConfig
from CrowdSim.plan.orca import ORCA
from CrowdSim.plan.sfm import Social_Force


DEFAULT_WHEEL_RADIUS = 0.0325
DEFAULT_WHEEL_BASE = 0.118
DEFAULT_MAX_WHEEL_SPEED = 12.0
DEFAULT_HEADING_GAIN = 2.5


class CrowdNavigationMarkers:
    """Lightweight IsaacLab visual markers for CrowdSim navigation debugging."""

    def __init__(self, device: torch.device, enabled: bool) -> None:
        self.device = device
        self.enabled = enabled
        self.humanoid_start_marker = None
        self.robot_start_marker = None
        self.humanoid_path_marker = None
        self.robot_path_marker = None
        self.velocity_marker = None

    def create(self, manager: "CrowdNavigationManager") -> None:
        if not self.enabled:
            return

        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        self.humanoid_start_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/CrowdSim/humanoid_starts",
                markers={
                    "marker": sim_utils.SphereCfg(
                        radius=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.1, 0.9, 0.2)
                        ),
                    )
                },
            )
        )
        self.robot_start_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/CrowdSim/robot_starts",
                markers={
                    "marker": sim_utils.SphereCfg(
                        radius=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 0.55, 1.0)
                        ),
                    )
                },
            )
        )
        self.humanoid_path_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/CrowdSim/humanoid_paths",
                markers={
                    "marker": sim_utils.SphereCfg(
                        radius=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.9, 0.25, 0.1)
                        ),
                    )
                },
            )
        )
        self.robot_path_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/CrowdSim/robot_paths",
                markers={
                    "marker": sim_utils.SphereCfg(
                        radius=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 0.9, 1.0)
                        ),
                    )
                },
            )
        )
        self.velocity_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/CrowdSim/velocity_arrows",
                markers={
                    "marker": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                        scale=(1.0, 0.1, 0.1),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.85, 0.0),
                            opacity=0.85,
                        ),
                    )
                },
            )
        )
        self.update_starts(manager.starts_xy, manager.config.num_humanoids)
        self.update_paths(manager.paths_xy, manager.config.num_humanoids)
        self.update_velocity(manager.starts_xy, manager.goals_xy - manager.starts_xy, manager)

    def update_starts(self, starts_xy: np.ndarray, num_humanoids: int) -> None:
        if not self.enabled:
            return

        humanoid_xy = starts_xy[:num_humanoids]
        robot_xy = starts_xy[num_humanoids:]
        self._visualize_spheres(self.humanoid_start_marker, humanoid_xy, z=0.06, scale=0.18)
        self._visualize_spheres(self.robot_start_marker, robot_xy, z=0.08, scale=0.22)

    def update_paths(self, paths_xy: list[np.ndarray], num_humanoids: int) -> None:
        if not self.enabled:
            return

        humanoid_points = self._flatten_paths(paths_xy[:num_humanoids])
        robot_points = self._flatten_paths(paths_xy[num_humanoids:])
        self._visualize_spheres(self.humanoid_path_marker, humanoid_points, z=0.035, scale=0.055)
        self._visualize_spheres(self.robot_path_marker, robot_points, z=0.045, scale=0.065)

    def update_velocity(
        self,
        positions_xy: np.ndarray,
        velocities_xy: np.ndarray,
        manager: "CrowdNavigationManager",
    ) -> None:
        if not self.enabled or self.velocity_marker is None or len(positions_xy) == 0:
            return

        directions = velocities_xy.copy()
        speeds = np.linalg.norm(directions, axis=1)
        for agent_id, speed in enumerate(speeds):
            if speed > 1e-4:
                continue
            fallback = manager._current_waypoint(agent_id) - positions_xy[agent_id]
            if np.linalg.norm(fallback) <= 1e-4:
                fallback = manager.goals_xy[agent_id] - positions_xy[agent_id]
            directions[agent_id] = fallback
            speeds[agent_id] = np.linalg.norm(fallback)

        yaws = np.arctan2(directions[:, 1], directions[:, 0])
        translations = np.zeros((len(positions_xy), 3), dtype=np.float32)
        translations[:, :2] = positions_xy
        translations[:, 2] = 0.55
        orientations = self._yaw_to_quat_wxyz(yaws)
        lengths = np.clip(speeds / max(manager.config.max_speed, 1e-4), 0.25, 1.0)
        scales = np.column_stack(
            [0.85 * lengths, 0.12 * np.ones_like(lengths), 0.12 * np.ones_like(lengths)]
        ).astype(np.float32)

        self.velocity_marker.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
        )

    def _visualize_spheres(self, marker, xy: np.ndarray, z: float, scale: float) -> None:
        if marker is None or len(xy) == 0:
            return
        translations = np.zeros((len(xy), 3), dtype=np.float32)
        translations[:, :2] = xy
        translations[:, 2] = z
        orientations = np.zeros((len(xy), 4), dtype=np.float32)
        orientations[:, 0] = 1.0
        scales = np.full((len(xy), 3), scale, dtype=np.float32)
        marker.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
        )

    @staticmethod
    def _flatten_paths(paths_xy: list[np.ndarray]) -> np.ndarray:
        if not paths_xy:
            return np.zeros((0, 2), dtype=np.float32)
        non_empty = [path for path in paths_xy if len(path) > 0]
        if not non_empty:
            return np.zeros((0, 2), dtype=np.float32)
        return np.concatenate(non_empty, axis=0).astype(np.float32)

    @staticmethod
    def _yaw_to_quat_wxyz(yaws: np.ndarray) -> np.ndarray:
        quats = np.zeros((len(yaws), 4), dtype=np.float32)
        half_yaws = 0.5 * yaws
        quats[:, 0] = np.cos(half_yaws)
        quats[:, 3] = np.sin(half_yaws)
        return quats


@dataclass
class CrowdNavigationConfig:
    map_path: Path
    map_resolution: float
    free_threshold: int
    num_humanoids: int
    num_robots: int
    device: torch.device
    seed: int = 7
    local_controller: str = "orca"
    agent_radius: float = 0.35
    humanoid_radius: float = 0.45
    safe_distance: float = 0.25
    max_speed: float = 0.8
    waypoint_tolerance: float = 0.45
    goal_tolerance: float = 0.75
    min_start_goal_distance: float = 5.0
    min_spawn_spacing: float = 1.2
    neighbor_radius: float = 4.0
    obstacle_query_radius: int = 14
    max_obstacles: int = 16
    collision_distance: float = 0.75
    log_interval: int = 120
    path_thin_spacing: float = 0.35
    wheel_radius: float = DEFAULT_WHEEL_RADIUS
    wheel_base: float = DEFAULT_WHEEL_BASE
    max_wheel_speed: float = DEFAULT_MAX_WHEEL_SPEED
    heading_gain: float = DEFAULT_HEADING_GAIN
    wheel_joint_indices: tuple[int, int] | None = None
    visual_markers_enabled: bool = False
    marker_update_interval: int = 10
    rl_num_neighbors: int = 4
    rl_num_obstacles: int = 8
    rl_obstacle_radius: float = 4.0
    rl_action_yaw_rate: float = 2.5
    rl_progress_reward_scale: float = 4.0
    rl_goal_reward: float = 10.0
    rl_collision_penalty: float = -10.0
    rl_time_penalty: float = -0.01
    rl_max_episode_steps: int = 600


class CrowdNavigationManager:
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
                seed=config.seed,
                min_start_goal_distance=config.min_start_goal_distance,
                min_spawn_spacing=config.min_spawn_spacing,
                path_thin_spacing=config.path_thin_spacing,
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
        self.step_count = 0
        self._last_positions = self.starts_xy.copy()
        self._markers: CrowdNavigationMarkers | None = None
        self._wheel_targets_tensor: torch.Tensor | None = None
        self._joint_velocity_targets: torch.Tensor | None = None
        self.path_log_path = self._write_navigation_path_log()

        planner_cfg = {
            "map": {"resolution_viz": self.pixels_per_meter},
            "env": {"dt": self._dt(), "safe_distance": config.safe_distance},
            "agent": {"radius": config.agent_radius, "max_vel": config.max_speed},
        }
        if config.local_controller == "orca":
            self.local_controller = ORCA(self.obstacle_map, planner_cfg)
        elif config.local_controller == "sfm":
            import cv2

            free_uint8 = self.free_mask.astype(np.uint8)
            distance_px = cv2.distanceTransform(free_uint8, cv2.DIST_L2, 5)
            self.local_controller = Social_Force(distance_px * config.map_resolution, planner_cfg)
        elif config.local_controller == "rl":
            self.local_controller = None
        else:
            raise ValueError(f"Unsupported local controller: {config.local_controller}")

        self._robot_rl_actions = np.zeros((config.num_robots, 2), dtype=np.float32)
        self._robot_prev_goal_dist = self._robot_goal_distances(self.starts_xy)
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
        if self.config.num_robots > 0 and self.robot is None:
            raise RuntimeError("Navigation requested robot control, but crowdsim_robot is missing.")
        if self.config.num_robots > 0:
            num_joints = self._num_robot_joints()
            self._wheel_targets_tensor = torch.zeros(
                self.config.num_robots,
                2,
                dtype=torch.float32,
                device=self.config.device,
            )
            if self.config.wheel_joint_indices is None and num_joints != 2:
                self._joint_velocity_targets = torch.zeros(
                    self.config.num_robots,
                    num_joints,
                    dtype=torch.float32,
                    device=self.config.device,
                )

        import types

        original_step = env.step

        def step_with_navigation(env_self, action):
            self.pre_step()
            result = original_step(action)
            self.post_step()
            return result

        env.step = types.MethodType(step_with_navigation, env)
        env.crowdsim_navigation = self
        self._markers = CrowdNavigationMarkers(
            device=self.config.device,
            enabled=self.config.visual_markers_enabled
            and not getattr(env.simulator, "headless", True),
        )
        self._markers.create(self)
        print(
            f"[CrowdSim] Navigation enabled: {self.config.num_humanoids} humanoid(s), "
            f"{self.config.num_robots} robot(s), controller={self.config.local_controller}."
        )
        print(f"[CrowdSim] Navigation path log: {self.path_log_path}")

    def pre_step(self) -> None:
        if self.config.num_robots == 0:
            return

        positions, velocities = self._read_agent_state()
        wheel_targets = self._compute_robot_wheel_targets(positions, velocities)
        self._write_robot_wheel_targets(wheel_targets)

    def post_step(self) -> None:
        self.step_count += 1
        positions, velocities = self._read_agent_state()
        self._update_waypoints_and_goals(positions)
        new_pairs = self._detect_collisions(positions)
        self._update_robot_rl_feedback(positions, velocities, new_pairs)
        if (
            self._markers is not None
            and self.config.marker_update_interval > 0
            and self.step_count % self.config.marker_update_interval == 0
        ):
            self._markers.update_velocity(positions, velocities, self)
        self._last_positions = positions
        self.env.extras["crowdsim_navigation"] = {
            "reached": int(self.reached.sum()),
            "num_agents": self.num_agents,
            "collision_pairs": len(self.collision_pairs),
            "new_collision_pairs": len(new_pairs),
        }

        if self.config.log_interval > 0 and self.step_count % self.config.log_interval == 0:
            print(
                f"[CrowdSim] nav step={self.step_count}, reached="
                f"{int(self.reached.sum())}/{self.num_agents}, "
                f"collision_pairs={len(self.collision_pairs)}"
            )
        if new_pairs:
            print(f"[CrowdSim] Collision warning: {sorted(new_pairs)}")

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

        records = []
        for agent_id, path in enumerate(self.paths_xy):
            agent_type = "humanoid" if agent_id < self.config.num_humanoids else "car"
            local_id = (
                agent_id
                if agent_type == "humanoid"
                else agent_id - self.config.num_humanoids
            )
            record = {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "local_id": local_id,
                "start_xy": self.starts_xy[agent_id].astype(float).tolist(),
                "goal_xy": self.goals_xy[agent_id].astype(float).tolist(),
                "path_xy": path.astype(float).tolist(),
            }
            records.append(record)
            print(
                f"[CrowdSim] path {agent_type}_{local_id}: "
                f"start={record['start_xy']}, goal={record['goal_xy']}, "
                f"waypoints={len(record['path_xy'])}"
            )

        payload = {
            "created_at": timestamp,
            "map_path": str(self.config.map_path),
            "map_resolution": self.config.map_resolution,
            "num_humanoids": self.config.num_humanoids,
            "num_cars": self.config.num_robots,
            "agents": records,
        }
        text = json.dumps(payload, indent=2)
        latest_path.write_text(text, encoding="utf-8")
        timestamp_path.write_text(text, encoding="utf-8")
        return latest_path

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

    def _compute_robot_wheel_targets(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> np.ndarray:
        wheel_targets = np.zeros((self.config.num_robots, 2), dtype=np.float32)
        robot_offset = self.config.num_humanoids

        for robot_idx in range(self.config.num_robots):
            agent_id = robot_offset + robot_idx
            if self.reached[agent_id]:
                continue

            pos = positions[agent_id]
            vel = velocities[agent_id]
            goal = self._current_waypoint(agent_id)

            if self.config.local_controller == "rl":
                wheel_targets[robot_idx] = self._rl_action_to_wheels(
                    self._robot_rl_actions[robot_idx]
                )
                continue

            nbr_state = self._neighbor_state(agent_id, positions, velocities)
            cord_int = self.world_to_pixel(pos)
            if self.config.local_controller == "orca":
                obstacles = self._nearby_obstacles_world(pos)
                desired_vel, _ = self.local_controller.get_action(
                    (pos, cord_int, vel, goal),
                    nbr_state,
                    obstacles,
                )
            else:
                desired_vel, _ = self.local_controller.get_action(
                    (pos, cord_int, vel, goal),
                    (nbr_state[0], nbr_state[1], nbr_state[2]),
                )

            wheel_targets[robot_idx] = self._desired_velocity_to_wheels(robot_idx, desired_vel)

        return wheel_targets

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

    def _nearby_obstacles_world(self, xy: np.ndarray) -> np.ndarray | None:
        pixel_yx = self.world_to_pixel(xy)
        radius = int(self.config.obstacle_query_radius)
        y0 = max(0, int(pixel_yx[0]) - radius)
        y1 = min(self.height, int(pixel_yx[0]) + radius + 1)
        x0 = max(0, int(pixel_yx[1]) - radius)
        x1 = min(self.width, int(pixel_yx[1]) + radius + 1)

        region = self.obstacle_map[y0:y1, x0:x1]
        obs_y, obs_x = np.nonzero(region)
        if len(obs_y) == 0:
            return None

        pixels = np.column_stack([obs_y + y0, obs_x + x0])
        dists = np.linalg.norm(pixels - pixel_yx[None, :], axis=1)
        keep = np.argsort(dists)[: self.config.max_obstacles]
        return np.asarray([self.pixel_to_world(pixel) for pixel in pixels[keep]], dtype=np.float32)

    def _desired_velocity_to_wheels(self, robot_idx: int, desired_vel: np.ndarray) -> np.ndarray:
        speed = float(np.linalg.norm(desired_vel))
        if speed < 1e-5:
            return np.zeros(2, dtype=np.float32)

        quat = self.robot.data.root_quat_w[robot_idx].detach().cpu().numpy()
        yaw = self._yaw_from_quat_wxyz(quat)
        heading = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        lateral = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)

        forward_speed = float(np.dot(desired_vel, heading))
        lateral_error = float(np.dot(desired_vel, lateral))
        yaw_rate = self.config.heading_gain * math.atan2(
            lateral_error, max(abs(forward_speed), 1e-4)
        )

        r = self.config.wheel_radius
        b = self.config.wheel_base
        left = (forward_speed - 0.5 * b * yaw_rate) / r
        right = (forward_speed + 0.5 * b * yaw_rate) / r
        wheels = np.clip(
            np.array([left, right], dtype=np.float32),
            -self.config.max_wheel_speed,
            self.config.max_wheel_speed,
        )
        return wheels

    def set_robot_rl_actions(self, actions: torch.Tensor | np.ndarray) -> None:
        """Set normalized RL actions for the next robot control step."""
        if self.config.local_controller != "rl" or self.config.num_robots == 0:
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

    def get_robot_rl_observations(self) -> torch.Tensor:
        """Return fixed-size observations for the robot navigation PPO policy."""
        positions, velocities = self._read_agent_state()
        obs = self._build_robot_rl_observations(positions, velocities)
        self._robot_last_obs = obs
        return obs

    def get_robot_rl_feedback(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Return next observations, rewards, dones, and diagnostics after env.step()."""
        if self._robot_last_obs is None:
            self.get_robot_rl_observations()
        return (
            self._robot_last_obs,
            self._robot_last_rewards,
            self._robot_last_dones,
            self._robot_last_info,
        )

    @property
    def robot_rl_obs_dim(self) -> int:
        return 8 + 5 * self.config.rl_num_neighbors + 3 * self.config.rl_num_obstacles

    def reset_robot_rl_episodes(self, done: torch.Tensor | np.ndarray) -> None:
        """Reset finished robot-only navigation episodes without resetting humanoids."""
        if self.config.local_controller != "rl" or self.config.num_robots == 0:
            return
        if isinstance(done, torch.Tensor):
            done_np = done.detach().cpu().numpy().astype(bool)
        else:
            done_np = np.asarray(done, dtype=bool)
        robot_ids = np.nonzero(done_np)[0]
        if len(robot_ids) == 0:
            return

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

        self._robot_rl_actions[robot_ids] = 0.0
        self.waypoint_ids[agent_ids] = 1
        self.reached[agent_ids] = False
        self._robot_episode_steps[robot_ids] = 0
        self._robot_prev_goal_dist[robot_ids] = np.linalg.norm(
            self.starts_xy[agent_ids] - self.goals_xy[agent_ids], axis=1
        )
        agent_id_set = {int(agent_id) for agent_id in agent_ids}
        self.collision_pairs = {
            pair
            for pair in self.collision_pairs
            if pair[0] not in agent_id_set and pair[1] not in agent_id_set
        }

    def _rl_action_to_wheels(self, action: np.ndarray) -> np.ndarray:
        forward_speed = float(np.clip(action[0], -1.0, 1.0)) * self.config.max_speed
        yaw_rate = float(np.clip(action[1], -1.0, 1.0)) * self.config.rl_action_yaw_rate
        left = (forward_speed - 0.5 * self.config.wheel_base * yaw_rate) / self.config.wheel_radius
        right = (forward_speed + 0.5 * self.config.wheel_base * yaw_rate) / self.config.wheel_radius
        return np.clip(
            np.array([left, right], dtype=np.float32),
            -self.config.max_wheel_speed,
            self.config.max_wheel_speed,
        )

    def _update_robot_rl_feedback(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        new_pairs: set[tuple[int, int]],
    ) -> None:
        if self.config.local_controller != "rl" or self.config.num_robots == 0:
            return

        robot_offset = self.config.num_humanoids
        robot_agent_ids = np.arange(robot_offset, robot_offset + self.config.num_robots)
        current_dist = np.linalg.norm(positions[robot_agent_ids] - self.goals_xy[robot_agent_ids], axis=1)
        progress = self._robot_prev_goal_dist - current_dist
        self._robot_prev_goal_dist = current_dist
        self._robot_episode_steps += 1

        collision = np.zeros(self.config.num_robots, dtype=bool)
        for i, j in new_pairs:
            if robot_offset <= i < robot_offset + self.config.num_robots:
                collision[i - robot_offset] = True
            if robot_offset <= j < robot_offset + self.config.num_robots:
                collision[j - robot_offset] = True

        reached = self.reached[robot_agent_ids].copy()
        timeout = self._robot_episode_steps >= self.config.rl_max_episode_steps
        rewards = (
            self.config.rl_time_penalty
            + self.config.rl_progress_reward_scale * progress
            + self.config.rl_goal_reward * reached.astype(np.float32)
            + self.config.rl_collision_penalty * collision.astype(np.float32)
        ).astype(np.float32)
        dones = reached | collision | timeout

        self._robot_last_obs = self._build_robot_rl_observations(positions, velocities)
        self._robot_last_rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.config.device)
        self._robot_last_dones = torch.as_tensor(dones, dtype=torch.bool, device=self.config.device)
        self._robot_last_info = {
            "reached": torch.as_tensor(reached, dtype=torch.bool, device=self.config.device),
            "collision": torch.as_tensor(collision, dtype=torch.bool, device=self.config.device),
            "timeout": torch.as_tensor(timeout, dtype=torch.bool, device=self.config.device),
            "distance_to_goal": torch.as_tensor(current_dist, dtype=torch.float32, device=self.config.device),
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

            waypoint_local = self._world_vec_to_local(self._current_waypoint(agent_id) - pos, yaw)
            goal_local = self._world_vec_to_local(self.goals_xy[agent_id] - pos, yaw)
            vel_local = self._world_vec_to_local(vel, yaw)
            row = [
                waypoint_local[0] / max(self.config.neighbor_radius, 1e-4),
                waypoint_local[1] / max(self.config.neighbor_radius, 1e-4),
                goal_local[0] / max(self.config.min_start_goal_distance, 1e-4),
                goal_local[1] / max(self.config.min_start_goal_distance, 1e-4),
                vel_local[0] / max(self.config.max_speed, 1e-4),
                vel_local[1] / max(self.config.max_speed, 1e-4),
                math.sin(yaw),
                math.cos(yaw),
            ]

            relpos = positions - pos
            reldis = np.linalg.norm(relpos, axis=1)
            mask = np.arange(self.num_agents) != agent_id
            neighbor_ids = np.argsort(np.where(mask, reldis, np.inf))[: self.config.rl_num_neighbors]
            for neighbor_id in neighbor_ids:
                if not np.isfinite(reldis[neighbor_id]) or reldis[neighbor_id] > self.config.neighbor_radius:
                    row.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                    continue
                local_rel = self._world_vec_to_local(relpos[neighbor_id], yaw)
                local_vel = self._world_vec_to_local(velocities[neighbor_id] - vel, yaw)
                row.extend(
                    [
                        local_rel[0] / max(self.config.neighbor_radius, 1e-4),
                        local_rel[1] / max(self.config.neighbor_radius, 1e-4),
                        local_vel[0] / max(self.config.max_speed, 1e-4),
                        local_vel[1] / max(self.config.max_speed, 1e-4),
                        reldis[neighbor_id] / max(self.config.neighbor_radius, 1e-4),
                    ]
                )

            obstacles = self._nearby_obstacles_world(pos)
            obstacle_rows: list[np.ndarray] = []
            if obstacles is not None and len(obstacles) > 0:
                obstacle_rel = obstacles - pos
                order = np.argsort(np.linalg.norm(obstacle_rel, axis=1))
                obstacle_rows = [obstacle_rel[idx] for idx in order[: self.config.rl_num_obstacles]]
            for obstacle_rel in obstacle_rows:
                local_rel = self._world_vec_to_local(obstacle_rel, yaw)
                dist = float(np.linalg.norm(obstacle_rel))
                row.extend(
                    [
                        local_rel[0] / max(self.config.rl_obstacle_radius, 1e-4),
                        local_rel[1] / max(self.config.rl_obstacle_radius, 1e-4),
                        dist / max(self.config.rl_obstacle_radius, 1e-4),
                    ]
                )
            for _ in range(self.config.rl_num_obstacles - len(obstacle_rows)):
                row.extend([0.0, 0.0, 0.0])

            obs_rows.append(np.asarray(row, dtype=np.float32))

        obs = np.stack(obs_rows, axis=0)
        return torch.as_tensor(obs, dtype=torch.float32, device=self.config.device)

    def _robot_goal_distances(self, positions: np.ndarray) -> np.ndarray:
        if self.config.num_robots == 0:
            return np.zeros(0, dtype=np.float32)
        start = self.config.num_humanoids
        end = start + self.config.num_robots
        return np.linalg.norm(positions[start:end] - self.goals_xy[start:end], axis=1).astype(np.float32)

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

    def _write_robot_wheel_targets(self, wheel_targets: np.ndarray) -> None:
        if self._wheel_targets_tensor is None:
            return

        targets_cpu = torch.as_tensor(wheel_targets, dtype=torch.float32)
        self._wheel_targets_tensor.copy_(targets_cpu, non_blocking=True)
        targets = self._wheel_targets_tensor
        if self.config.wheel_joint_indices is not None:
            joint_ids = torch.as_tensor(
                self.config.wheel_joint_indices,
                dtype=torch.long,
                device=self.config.device,
            )
            self.robot.set_joint_velocity_target(targets, joint_ids=joint_ids)
            return

        num_joints = self._num_robot_joints()
        if num_joints != 2:
            if (
                self._joint_velocity_targets is None
                or self._joint_velocity_targets.shape[1] != num_joints
            ):
                self._joint_velocity_targets = torch.zeros(
                    self.config.num_robots,
                    num_joints,
                    dtype=torch.float32,
                    device=self.config.device,
                )
            self._joint_velocity_targets.zero_()
            self._joint_velocity_targets[:, : min(2, num_joints)].copy_(
                targets[:, : min(2, num_joints)]
            )
            targets = self._joint_velocity_targets
        self.robot.set_joint_velocity_target(targets)

    def _num_robot_joints(self) -> int:
        fallback = self._wheel_targets_tensor.shape[1] if self._wheel_targets_tensor is not None else 2
        return int(getattr(self.robot, "num_joints", fallback))

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

    def _current_waypoint(self, agent_id: int) -> np.ndarray:
        path = self.paths_xy[agent_id]
        idx = min(int(self.waypoint_ids[agent_id]), len(path) - 1)
        return path[idx]

    def world_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        return self.task.world_to_pixel(xy)

    def pixel_to_world(self, pixel_yx: np.ndarray) -> np.ndarray:
        return self.task.pixel_to_world(pixel_yx)

    def _dt(self) -> float:
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
