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


JETBOT_WHEEL_RADIUS = 0.0325
JETBOT_WHEEL_BASE = 0.118
JETBOT_MAX_WHEEL_SPEED = 12.0
JETBOT_HEADING_GAIN = 2.5


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
        else:
            raise ValueError(f"Unsupported local controller: {config.local_controller}")

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
            enabled=not getattr(env.simulator, "headless", True),
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
        if self._markers is not None:
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
        yaw_rate = JETBOT_HEADING_GAIN * math.atan2(
            lateral_error, max(abs(forward_speed), 1e-4)
        )

        r = JETBOT_WHEEL_RADIUS
        b = JETBOT_WHEEL_BASE
        left = (forward_speed - 0.5 * b * yaw_rate) / r
        right = (forward_speed + 0.5 * b * yaw_rate) / r
        wheels = np.clip(
            np.array([left, right], dtype=np.float32),
            -JETBOT_MAX_WHEEL_SPEED,
            JETBOT_MAX_WHEEL_SPEED,
        )
        return wheels

    def _write_robot_wheel_targets(self, wheel_targets: np.ndarray) -> None:
        targets = torch.tensor(wheel_targets, dtype=torch.float32, device=self.config.device)
        num_joints = getattr(self.robot, "num_joints", targets.shape[1])
        if num_joints != 2:
            full_targets = torch.zeros(
                self.config.num_robots,
                num_joints,
                dtype=torch.float32,
                device=self.config.device,
            )
            full_targets[:, : min(2, num_joints)] = targets[:, : min(2, num_joints)]
            targets = full_targets
        self.robot.set_joint_velocity_target(targets)

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
