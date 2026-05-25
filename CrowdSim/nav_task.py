"""Map loading, start/goal sampling, and A* task planning for CrowdSim."""

from __future__ import annotations

import colorsys
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
            if not self._is_white_traversable(candidate_array):
                continue
            if self.component_labels[candidate] == 0:
                continue
            if starts and min(
                np.linalg.norm(candidate_array - np.asarray(point)) for point in starts
            ) < min_spacing_px:
                continue
            goal = self._sample_goal_for_start(candidate_array, min_goal_px)
            if goal is None:
                continue
            if not self._is_white_traversable(goal):
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

    def _is_white_traversable(self, pixel_yx: np.ndarray) -> bool:
        pixel = tuple(int(value) for value in pixel_yx)
        return bool(self.free_mask[pixel] and self.planner_free_mask[pixel])

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


def agent_marker_color(agent_id: int) -> tuple[float, float, float]:
    hue = (0.08 + 0.61803398875 * float(agent_id)) % 1.0
    return tuple(float(value) for value in colorsys.hsv_to_rgb(hue, 0.78, 0.95))


def build_agent_marker_prototypes(sim_utils, num_humanoids: int, num_robots: int) -> dict:
    prototypes = {}
    total = int(num_humanoids) + int(num_robots)
    for agent_id in range(total):
        color = agent_marker_color(agent_id)
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
        if agent_id < num_humanoids:
            prototypes[f"humanoid_{agent_id}"] = sim_utils.SphereCfg(
                radius=1.0,
                visual_material=material,
            )
        else:
            robot_id = agent_id - num_humanoids
            prototypes[f"car_{robot_id}"] = sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                visual_material=material,
            )
    return prototypes


class NavigationTaskMarkers:
    """Static IsaacLab markers for A* paths and final goals."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.agent_marker = None

    def create(self, task: NavigationTask, num_humanoids: int) -> None:
        if not self.enabled:
            return

        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self.agent_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/CrowdSim/nav_paths_and_goals",
                markers=build_agent_marker_prototypes(
                    sim_utils,
                    num_humanoids=num_humanoids,
                    num_robots=max(0, len(task.paths_xy) - num_humanoids),
                ),
            )
        )

        translations, orientations, scales, marker_indices = self._static_marker_arrays(
            paths_xy=task.paths_xy,
            goals_xy=task.goals_xy,
            num_humanoids=num_humanoids,
        )
        self.agent_marker.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
            marker_indices=marker_indices,
        )

    def _static_marker_arrays(
        self,
        paths_xy: list[np.ndarray],
        goals_xy: np.ndarray,
        num_humanoids: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        translations: list[list[float]] = []
        scales: list[list[float]] = []
        marker_indices: list[int] = []

        for agent_id, path in enumerate(paths_xy):
            path_points = path[1:-1] if len(path) > 2 else np.zeros((0, 2), dtype=np.float32)
            for point in path_points:
                translations.append([float(point[0]), float(point[1]), 0.04])
                scales.append(self._path_scale(agent_id, num_humanoids))
                marker_indices.append(agent_id)

            goal = goals_xy[agent_id]
            translations.append([float(goal[0]), float(goal[1]), 0.10])
            scales.append(self._goal_scale(agent_id, num_humanoids))
            marker_indices.append(agent_id)

        if not translations:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        translations_array = np.asarray(translations, dtype=np.float32)
        orientations = np.zeros((len(translations_array), 4), dtype=np.float32)
        orientations[:, 0] = 1.0
        return (
            translations_array,
            orientations,
            np.asarray(scales, dtype=np.float32),
            np.asarray(marker_indices, dtype=np.int32),
        )

    @staticmethod
    def _path_scale(agent_id: int, num_humanoids: int) -> list[float]:
        if agent_id < num_humanoids:
            return [0.055, 0.055, 0.055]
        return [0.075, 0.075, 0.035]

    @staticmethod
    def _goal_scale(agent_id: int, num_humanoids: int) -> list[float]:
        if agent_id < num_humanoids:
            return [0.22, 0.22, 0.22]
        return [0.26, 0.26, 0.10]
