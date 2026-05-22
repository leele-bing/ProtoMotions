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
    seed: int = 7
    min_start_goal_distance: float = 5.0
    min_spawn_spacing: float = 1.2
    path_thin_spacing: float = 0.35


class NavigationTask:
    """Shared navigation task state independent of the local controller."""

    def __init__(self, config: NavigationTaskConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.free_mask, self.obstacle_map = self._load_map(config.map_path)
        self.height, self.width = self.free_mask.shape
        self.planner = Path_Planner(self.obstacle_map, smooth=False, viz=False)
        self.planner_free_mask = self.planner.map_dialate == 0
        self.component_labels, self.component_sizes = self._label_planner_free_space()
        self._component_pixel_cache: dict[int, np.ndarray] = {}
        self.starts_px, self.goals_px = self._sample_start_goal_pixels()
        self.starts_xy = np.asarray(
            [self.pixel_to_world(px) for px in self.starts_px], dtype=np.float32
        )
        self.goals_xy = np.asarray(
            [self.pixel_to_world(px) for px in self.goals_px], dtype=np.float32
        )
        self.paths_xy = self._plan_paths()

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
            path_xy = self._thin_path(path_xy)
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

    def _thin_path(self, path: np.ndarray) -> np.ndarray:
        min_spacing = self.config.path_thin_spacing
        if len(path) <= 2:
            return path

        thinned = [path[0]]
        for point in path[1:-1]:
            if np.linalg.norm(point - thinned[-1]) >= min_spacing:
                thinned.append(point)
        thinned.append(path[-1])
        return np.asarray(thinned, dtype=np.float32)
