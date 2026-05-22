"""Navigation helpers for CrowdSim humanoid/robot scenes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from CrowdSim.plan.orca import ORCA
from CrowdSim.plan.planning import Path_Planner
from CrowdSim.plan.sfm import Social_Force


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
    jetbot_wheel_radius: float = 0.0325
    jetbot_wheel_base: float = 0.118
    jetbot_max_wheel_speed: float = 12.0
    heading_gain: float = 2.5
    collision_distance: float = 0.75
    log_interval: int = 120


class CrowdNavigationManager:
    """Plan and control CrowdSim agents in the shared Office map.

    Humanoids are still controlled by the loaded MaskedMimic policy. This manager
    samples their starts/goals and monitors their progress/collisions. Navigation
    wheel commands are applied only to CrowdRobot/Jetbot agents.
    """

    def __init__(self, config: CrowdNavigationConfig) -> None:
        self.config = config
        self.num_agents = config.num_humanoids + config.num_robots
        self.rng = random.Random(config.seed)

        self.free_mask, self.obstacle_map = self._load_map(config.map_path)
        self.height, self.width = self.free_mask.shape
        self.pixels_per_meter = 1.0 / config.map_resolution

        self.starts_px, self.goals_px = self._sample_start_goal_pixels()
        self.starts_xy = np.array([self.pixel_to_world(px) for px in self.starts_px], dtype=np.float32)
        self.goals_xy = np.array([self.pixel_to_world(px) for px in self.goals_px], dtype=np.float32)
        self.paths_xy = self._plan_paths()
        self.waypoint_ids = np.ones(self.num_agents, dtype=np.int64)
        self.reached = np.zeros(self.num_agents, dtype=bool)
        self.collision_pairs: set[tuple[int, int]] = set()
        self.step_count = 0
        self._last_positions = self.starts_xy.copy()

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
        print(
            f"[CrowdSim] Navigation enabled: {self.config.num_humanoids} humanoid(s), "
            f"{self.config.num_robots} robot(s), controller={self.config.local_controller}."
        )

    def pre_step(self) -> None:
        if self.config.num_robots == 0:
            return

        positions, velocities = self._read_agent_state()
        wheel_targets = self._compute_robot_wheel_targets(positions, velocities)
        self._write_robot_wheel_targets(wheel_targets)

    def post_step(self) -> None:
        self.step_count += 1
        positions, _ = self._read_agent_state()
        self._last_positions = positions
        self._update_waypoints_and_goals(positions)
        new_pairs = self._detect_collisions(positions)
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

    def _load_map(self, map_path: Path) -> tuple[np.ndarray, np.ndarray]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required for CrowdSim navigation maps.") from exc

        image = Image.open(map_path).convert("L")
        grid = np.asarray(image, dtype=np.uint8)
        free_mask = grid >= int(self.config.free_threshold)
        obstacle_map = (~free_mask).astype(np.uint8)
        return free_mask, obstacle_map

    def _sample_start_goal_pixels(self) -> tuple[np.ndarray, np.ndarray]:
        free_pixels = np.column_stack(np.nonzero(self.free_mask))
        self.rng.shuffle(free_pixels)

        starts: list[np.ndarray] = []
        goals: list[np.ndarray] = []
        min_spacing_px = self.config.min_spawn_spacing / self.config.map_resolution
        min_goal_px = self.config.min_start_goal_distance / self.config.map_resolution

        for candidate in free_pixels:
            if len(starts) == self.num_agents:
                break
            if starts and min(np.linalg.norm(candidate - np.asarray(p)) for p in starts) < min_spacing_px:
                continue
            goal = self._sample_goal_for_start(candidate, min_goal_px)
            if goal is None:
                continue
            starts.append(candidate.copy())
            goals.append(goal)

        if len(starts) < self.num_agents:
            raise RuntimeError(
                f"Only sampled {len(starts)}/{self.num_agents} navigation starts from the map."
            )

        return np.asarray(starts, dtype=np.int64), np.asarray(goals, dtype=np.int64)

    def _sample_goal_for_start(self, start: np.ndarray, min_goal_px: float) -> np.ndarray | None:
        ys, xs = np.nonzero(self.free_mask)
        for _ in range(2000):
            idx = self.rng.randrange(len(ys))
            goal = np.array([ys[idx], xs[idx]], dtype=np.int64)
            if np.linalg.norm(goal - start) >= min_goal_px:
                return goal
        return None

    def _plan_paths(self) -> list[np.ndarray]:
        planner = Path_Planner(self.obstacle_map, smooth=False, viz=False)
        paths: list[np.ndarray] = []
        for agent_id, (start, goal) in enumerate(zip(self.starts_px, self.goals_px)):
            path_px = planner.get_astar_path(start, goal)
            if path_px is None or len(path_px) == 0:
                path_xy = np.stack([self.pixel_to_world(start), self.pixel_to_world(goal)], axis=0)
                print(f"[CrowdSim] Warning: using straight fallback path for agent {agent_id}.")
            else:
                path_xy = np.asarray([self.pixel_to_world(px) for px in path_px], dtype=np.float32)
                path_xy = self._thin_path(path_xy, min_spacing=0.35)
            paths.append(path_xy)
        return paths

    def _thin_path(self, path: np.ndarray, min_spacing: float) -> np.ndarray:
        if len(path) <= 2:
            return path
        thinned = [path[0]]
        for point in path[1:-1]:
            if np.linalg.norm(point - thinned[-1]) >= min_spacing:
                thinned.append(point)
        thinned.append(path[-1])
        return np.asarray(thinned, dtype=np.float32)

    def _initial_robot_yaws(self) -> np.ndarray:
        yaws = np.zeros(self.config.num_robots, dtype=np.float32)
        offset = self.config.num_humanoids
        for idx in range(self.config.num_robots):
            path = self.paths_xy[offset + idx]
            if len(path) > 1:
                delta = path[1] - path[0]
                yaws[idx] = math.atan2(float(delta[1]), float(delta[0]))
        return yaws

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
        yaw_rate = self.config.heading_gain * math.atan2(lateral_error, max(abs(forward_speed), 1e-4))

        r = self.config.jetbot_wheel_radius
        b = self.config.jetbot_wheel_base
        left = (forward_speed - 0.5 * b * yaw_rate) / r
        right = (forward_speed + 0.5 * b * yaw_rate) / r
        wheels = np.clip(
            np.array([left, right], dtype=np.float32),
            -self.config.jetbot_max_wheel_speed,
            self.config.jetbot_max_wheel_speed,
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
        center_x = (self.width - 1) * 0.5
        center_y = (self.height - 1) * 0.5
        pixel_x = int(round(float(xy[0]) / self.config.map_resolution + center_x))
        pixel_y = int(round(center_y - float(xy[1]) / self.config.map_resolution))
        pixel_x = int(np.clip(pixel_x, 0, self.width - 1))
        pixel_y = int(np.clip(pixel_y, 0, self.height - 1))
        return np.array([pixel_y, pixel_x], dtype=np.int64)

    def pixel_to_world(self, pixel_yx: np.ndarray) -> np.ndarray:
        pixel_y = float(pixel_yx[0])
        pixel_x = float(pixel_yx[1])
        center_x = (self.width - 1) * 0.5
        center_y = (self.height - 1) * 0.5
        return np.array(
            [
                (pixel_x - center_x) * self.config.map_resolution,
                (center_y - pixel_y) * self.config.map_resolution,
            ],
            dtype=np.float32,
        )

    def _dt(self) -> float:
        return 1.0 / 30.0

    @staticmethod
    def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
        w, x, y, z = [float(v) for v in quat]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
