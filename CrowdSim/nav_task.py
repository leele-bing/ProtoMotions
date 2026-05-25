"""Map loading, start/goal sampling, and A* task planning for CrowdSim."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from CrowdSim.plan.planning import Path_Planner


@dataclass
class NavigationTaskConfig:
    map_path: Path
    map_resolution: float
    free_threshold: int
    num_agents: int
    map_origin_xy: tuple[float, float] = (0.0, 0.0)
    seed: int = 7
    min_start_goal_distance: float = 5.0
    min_spawn_spacing: float = 1.2
    planning_step_size: float = 0.5
    planning_clearance: float = 0.2


class NavigationTask:
    """Shared navigation task state independent of the local controller."""

    def __init__(self, config: NavigationTaskConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.free_mask, self.obstacle_map = self._load_map(config.map_path)
        self.height, self.width = self.free_mask.shape
        self.planner = Path_Planner(
            self.obstacle_map,
            map_resolution=config.map_resolution,
            step_size_m=config.planning_step_size,
            clearance_m=config.planning_clearance,
            smooth=False,
            viz=False,
        )
        self.planner_free_mask = self.planner.map_dialate == 0
        self.component_labels, self.component_sizes = self._label_planner_free_space()
        self._component_pixel_cache: dict[int, np.ndarray] = {}
        self._markers: NavigationTaskMarkers | None = None
        self.starts_px, self.goals_px = self._sample_start_goal_pixels()
        self.starts_xy = np.asarray(
            [self.pixel_to_world(px) for px in self.starts_px], dtype=np.float32
        )
        self.goals_xy = np.asarray(
            [self.pixel_to_world(px) for px in self.goals_px], dtype=np.float32
        )
        self.paths_xy = self._plan_paths()

    def world_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        origin_x, origin_y = self.config.map_origin_xy
        pixel_x = int(round((float(xy[0]) - origin_x) / self.config.map_resolution))
        pixel_y = int(
            round((self.height - 1) - (float(xy[1]) - origin_y) / self.config.map_resolution)
        )
        pixel_x = int(np.clip(pixel_x, 0, self.width - 1))
        pixel_y = int(np.clip(pixel_y, 0, self.height - 1))
        return np.array([pixel_y, pixel_x], dtype=np.int64)

    def pixel_to_world(self, pixel_yx: np.ndarray) -> np.ndarray:
        pixel_y = float(pixel_yx[0])
        pixel_x = float(pixel_yx[1])
        origin_x, origin_y = self.config.map_origin_xy
        return np.array(
            [
                origin_x + pixel_x * self.config.map_resolution,
                origin_y + (self.height - 1 - pixel_y) * self.config.map_resolution,
            ],
            dtype=np.float32,
        )

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
        free_pixels = [tuple(pixel) for pixel in np.column_stack(np.nonzero(self.planner_free_mask))]
        self.rng.shuffle(free_pixels)

        starts: list[np.ndarray] = []
        goals: list[np.ndarray] = []
        min_spacing_px = self.config.min_spawn_spacing / self.config.map_resolution
        min_goal_px = self.config.min_start_goal_distance / self.config.map_resolution

        for candidate in free_pixels:
            if len(starts) == self.config.num_agents:
                break
            candidate_array = np.asarray(candidate, dtype=np.int64)
            if self.component_labels[candidate] == 0:
                continue
            if starts and min(
                np.linalg.norm(candidate_array - np.asarray(point)) for point in starts
            ) < min_spacing_px:
                continue
            goal = self._sample_goal_for_start(candidate_array, min_goal_px)
            if goal is None:
                continue
            starts.append(candidate_array.copy())
            goals.append(goal)

        if len(starts) < self.config.num_agents:
            raise RuntimeError(
                f"Only sampled {len(starts)}/{self.config.num_agents} navigation starts "
                "from the A* traversable white map area. Check scene_map, free_threshold, "
                "or map connectivity."
            )

        return np.asarray(starts, dtype=np.int64), np.asarray(goals, dtype=np.int64)

    def _sample_goal_for_start(self, start: np.ndarray, min_goal_px: float) -> np.ndarray | None:
        component_id = int(self.component_labels[tuple(start)])
        component_pixels = self._pixels_for_component(component_id)
        if len(component_pixels) == 0:
            return None

        for _ in range(2000):
            idx = self.rng.randrange(len(component_pixels))
            goal = component_pixels[idx]
            if np.linalg.norm(goal - start) >= min_goal_px:
                return goal.copy()
        return None

    def _plan_paths(self) -> list[np.ndarray]:
        paths: list[np.ndarray] = []
        for agent_id, (start, goal) in enumerate(zip(self.starts_px, self.goals_px)):
            path_px = self.planner.get_astar_path(start, goal)
            if path_px is None or len(path_px) == 0:
                raise RuntimeError(
                    f"A* failed for agent {agent_id}: start_px={start.tolist()}, "
                    f"goal_px={goal.tolist()}, start_xy={self.pixel_to_world(start).tolist()}, "
                    f"goal_xy={self.pixel_to_world(goal).tolist()}. "
                    "No straight-line fallback is used."
                )
            path_xy = np.asarray(
                [self.pixel_to_world(px) for px in path_px], dtype=np.float32
            )
            paths.append(path_xy)
        return paths

    def _label_planner_free_space(self) -> tuple[np.ndarray, np.ndarray]:
        num_labels, labels = cv2.connectedComponents(
            self.planner_free_mask.astype(np.uint8), connectivity=8
        )
        sizes = np.bincount(labels.reshape(-1), minlength=num_labels)
        return labels.astype(np.int32, copy=False), sizes

    def _pixels_for_component(self, component_id: int) -> np.ndarray:
        if component_id not in self._component_pixel_cache:
            self._component_pixel_cache[component_id] = np.column_stack(
                np.nonzero(self.component_labels == component_id)
            ).astype(np.int64)
        return self._component_pixel_cache[component_id]

    def create_visualization_markers(self, num_humanoids: int, enabled: bool) -> None:
        self._markers = NavigationTaskMarkers(enabled)
        self._markers.create(self, num_humanoids)


class NavigationTaskMarkers:
    """Static IsaacLab markers for sampled starts, goals, and A* paths."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.humanoid_start_marker = None
        self.car_start_marker = None
        self.humanoid_goal_marker = None
        self.car_goal_marker = None
        self.humanoid_path_marker = None
        self.car_path_marker = None

    def create(self, task: NavigationTask, num_humanoids: int) -> None:
        if not self.enabled:
            return

        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self.humanoid_start_marker = self._sphere_marker(
            "/Visuals/CrowdSim/humanoid_starts",
            sim_utils,
            color=(0.1, 0.9, 0.2),
        )
        self.car_start_marker = self._sphere_marker(
            "/Visuals/CrowdSim/car_starts",
            sim_utils,
            color=(0.0, 0.55, 1.0),
        )
        self.humanoid_goal_marker = self._sphere_marker(
            "/Visuals/CrowdSim/humanoid_goals",
            sim_utils,
            color=(1.0, 0.25, 0.1),
        )
        self.car_goal_marker = self._sphere_marker(
            "/Visuals/CrowdSim/car_goals",
            sim_utils,
            color=(1.0, 0.8, 0.0),
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
        self.car_path_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/CrowdSim/car_paths",
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

        humanoid_slice = slice(0, num_humanoids)
        robot_slice = slice(num_humanoids, None)
        self._visualize_spheres(
            self.humanoid_start_marker, task.starts_xy[humanoid_slice], z=0.06, scale=0.18
        )
        self._visualize_spheres(
            self.car_start_marker, task.starts_xy[robot_slice], z=0.08, scale=0.22
        )
        self._visualize_spheres(
            self.humanoid_goal_marker, task.goals_xy[humanoid_slice], z=0.06, scale=0.2
        )
        self._visualize_spheres(
            self.car_goal_marker, task.goals_xy[robot_slice], z=0.08, scale=0.24
        )
        self._visualize_spheres(
            self.humanoid_path_marker,
            self._flatten_paths(task.paths_xy[humanoid_slice]),
            z=0.035,
            scale=0.055,
        )
        self._visualize_spheres(
            self.car_path_marker,
            self._flatten_paths(task.paths_xy[robot_slice]),
            z=0.045,
            scale=0.065,
        )

    @staticmethod
    def _sphere_marker(prim_path: str, sim_utils, color: tuple[float, float, float]):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        return VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=prim_path,
                markers={
                    "marker": sim_utils.SphereCfg(
                        radius=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    )
                },
            )
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
    def _flatten_paths(paths_xy) -> np.ndarray:
        paths = list(paths_xy)
        if not paths:
            return np.zeros((0, 2), dtype=np.float32)
        non_empty = [path for path in paths if len(path) > 0]
        if not non_empty:
            return np.zeros((0, 2), dtype=np.float32)
        return np.concatenate(non_empty, axis=0).astype(np.float32)
