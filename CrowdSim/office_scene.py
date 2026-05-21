"""Shared helpers for CrowdSim global USD scenes."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


DEFAULT_OFFICE_MAP_RESOLUTION = 100.0 / 1999.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root() / path).resolve()


def parse_spawn_xy(value: str, num_envs: int, device: torch.device) -> torch.Tensor:
    pairs: list[tuple[float, float]] = []
    for item in value.split(";"):
        if not item.strip():
            continue
        x_str, y_str = item.split(",", maxsplit=1)
        pairs.append((float(x_str), float(y_str)))

    if not pairs:
        raise ValueError("spawn_xy must contain at least one x,y pair")

    while len(pairs) < num_envs:
        pairs.append(pairs[len(pairs) % len(pairs)])

    return torch.tensor(pairs[:num_envs], dtype=torch.float32, device=device)


def sample_spawn_xy_from_map(
    map_path: Path,
    num_envs: int,
    device: torch.device,
    map_resolution: float = DEFAULT_OFFICE_MAP_RESOLUTION,
    humanoid_radius: float = 0.45,
    min_spacing: float = 0.9,
    free_threshold: int = 200,
    seed: int = 0,
) -> torch.Tensor:
    """Sample initial XY positions from an occupancy image.

    The image center is world (0, 0). Image +x maps to world +x, and image -y
    maps to world +y. The default resolution matches a 3999x3999 map exported
    with stage X/Y bounds [-100, 100]. By default, white pixels are treated as
    free space and dark/gray pixels as obstacles or unknown area.
    """
    if map_resolution <= 0:
        raise ValueError(f"map_resolution must be positive, got {map_resolution}")

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for CrowdSim PNG map sampling.") from exc

    image = Image.open(map_path).convert("L")
    grid = np.asarray(image, dtype=np.uint8)
    free = grid >= int(free_threshold)
    height, width = free.shape
    if not free.any():
        print(f"[CrowdSim] Warning: no free pixels found in {map_path}; using fallback grid.")
        return _fallback_spawn_grid(num_envs, device, min_spacing)

    radius_px = max(1, int(round(humanoid_radius / map_resolution)))
    spacing_px = max(1, int(round(min_spacing / map_resolution)))
    free_integral = _integral_image(free.astype(np.uint8))

    ys, xs = np.nonzero(free)
    stride = max(1, radius_px // 2)
    candidate_pixels = [
        (int(x), int(y))
        for x, y in zip(xs[::stride], ys[::stride])
        if _is_clear_square(free_integral, int(x), int(y), radius_px, width, height)
    ]

    rng = random.Random(seed)
    rng.shuffle(candidate_pixels)

    selected_pixels: list[tuple[int, int]] = []
    min_dist_sq = spacing_px * spacing_px
    for pixel in candidate_pixels:
        if all(
            (pixel[0] - other[0]) ** 2 + (pixel[1] - other[1]) ** 2 >= min_dist_sq
            for other in selected_pixels
        ):
            selected_pixels.append(pixel)
            if len(selected_pixels) == num_envs:
                break

    if len(selected_pixels) < num_envs:
        print(
            f"[CrowdSim] Warning: only found {len(selected_pixels)}/{num_envs} "
            f"free spawn points in {map_path}; filling remaining points with fallback grid."
        )
        fallback = _fallback_spawn_grid(num_envs - len(selected_pixels), device, min_spacing)
        selected_xy = [_pixel_to_world(px, py, width, height, map_resolution) for px, py in selected_pixels]
        selected_xy.extend((float(x), float(y)) for x, y in fallback.cpu().tolist())
        return torch.tensor(selected_xy[:num_envs], dtype=torch.float32, device=device)

    selected_xy = [
        _pixel_to_world(px, py, width, height, map_resolution)
        for px, py in selected_pixels[:num_envs]
    ]
    print(
        f"[CrowdSim] PNG map spawn: {len(candidate_pixels)} clear candidate pixel(s), "
        f"resolution={map_resolution} m/px, free_threshold={free_threshold}."
    )
    return torch.tensor(selected_xy, dtype=torch.float32, device=device)


def _pixel_to_world(
    pixel_x: int, pixel_y: int, width: int, height: int, resolution: float
) -> tuple[float, float]:
    center_x = (width - 1) * 0.5
    center_y = (height - 1) * 0.5
    world_x = (pixel_x - center_x) * resolution
    world_y = (center_y - pixel_y) * resolution
    return world_x, world_y


def _integral_image(mask: np.ndarray) -> np.ndarray:
    values = mask.astype(np.int64, copy=False)
    return np.pad(values.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def _is_clear_square(
    integral: np.ndarray,
    x: int,
    y: int,
    radius: int,
    width: int,
    height: int,
) -> bool:
    x0 = max(0, x - radius)
    x1 = min(width - 1, x + radius)
    y0 = max(0, y - radius)
    y1 = min(height - 1, y + radius)
    area = (x1 - x0 + 1) * (y1 - y0 + 1)
    free_count = (
        integral[y1 + 1, x1 + 1]
        - integral[y0, x1 + 1]
        - integral[y1 + 1, x0]
        + integral[y0, x0]
    )
    return int(free_count) == area


def _fallback_spawn_grid(
    num_envs: int, device: torch.device, spacing: float = 1.0
) -> torch.Tensor:
    side = int(num_envs**0.5)
    if side * side < num_envs:
        side += 1
    origin = -0.5 * spacing * (side - 1)
    points = []
    for i in range(num_envs):
        row = i // side
        col = i % side
        points.append((origin + col * spacing, origin + row * spacing))
    return torch.tensor(points, dtype=torch.float32, device=device)


def add_global_usd_reference(
    usd_path: Path,
    prim_path: str = "/World/Office",
    z_offset: float = 0.0,
) -> None:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(str(usd_path))
    UsdGeom.XformCommonAPI(prim).SetTranslate((0.0, 0.0, z_offset))


def patch_isaaclab_scene_with_global_usd(
    usd_path: Path,
    z_offset: float = 0.0,
    prim_path: str = "/World/Office",
) -> None:
    """Add a global USD asset to IsaacLab SceneCfg before simulator construction."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    import protomotions.simulator.isaaclab.simulator as simulator_module
    from protomotions.simulator.isaaclab.utils.scene import SceneCfg as BaseSceneCfg

    class GlobalUsdSceneCfg(BaseSceneCfg):
        def __init__(self, *scene_args, **scene_kwargs):
            super().__init__(*scene_args, **scene_kwargs)
            self.global_usd_asset = AssetBaseCfg(
                prim_path=prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(usd_path),
                    activate_contact_sensors=False,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, z_offset)),
                collision_group=-1,
            )

    simulator_module.SceneCfg = GlobalUsdSceneCfg


def apply_fixed_spawn_offsets(env, spawn_xy: torch.Tensor) -> None:
    """Pin ProtoMotions respawn offsets to fixed XY positions."""
    import types

    offsets = torch.zeros(env.num_envs, 3, dtype=torch.float32, device=env.device)
    offsets[:, :2] = spawn_xy

    def update_respawn_root_offset_by_env_ids(self, env_ids, ref_state=None, sample_flat=False):
        self.respawn_root_offset[env_ids] = offsets[env_ids]

    env.update_respawn_root_offset_by_env_ids = types.MethodType(
        update_respawn_root_offset_by_env_ids, env
    )
    env.respawn_root_offset[:] = offsets
