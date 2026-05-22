"""Config-driven CrowdSim entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch("isaaclab")

import torch  # noqa: E402

from CrowdSim.navigation import CrowdNavigationConfig, CrowdNavigationManager  # noqa: E402
from CrowdSim.sensor_stream import (  # noqa: E402
    RobotCameraStreamConfig,
    configure_robot_camera_recorder,
)
from CrowdSim.sim_agent import (  # noqa: E402
    build_runtime,
    configure_viewer_camera,
    create_fabric,
    enable_human_mesh,
    make_crowd_robot_config,
    resolve_robot_usd,
    suppress_known_isaaclab_warning_spam,
)
from CrowdSim.sim_world import (  # noqa: E402
    add_global_usd_reference,
    apply_fixed_crowd_robot_spawns,
    apply_fixed_spawn_offsets,
    hide_prims_matching_keywords,
    parse_spawn_xy,
    parse_spawn_xy_yaw,
    patch_isaaclab_scene_with_crowdsim_assets,
    resolve_repo_path,
    sample_spawn_xy_from_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CrowdSim from a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="CrowdSim/config/cfg.yaml")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--full-eval", action="store_true")
    parser.add_argument("--scene-physics", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load the small CrowdSim YAML subset without requiring PyYAML."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if ":" not in stripped:
            raise ValueError(f"Invalid config line {line_number}: {raw_line}")

        key, raw_value = stripped.split(":", maxsplit=1)
        key = key.strip()
        raw_value = raw_value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid indentation at line {line_number}: {raw_line}")

        parent = stack[-1][1]
        if raw_value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(raw_value)

    return root


def parse_scalar(value: str):
    text = value.strip()
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(item.strip()) for item in inner.split(",")]
    if "," in text:
        return [parse_scalar(item) for item in text.split(",")]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def cfg_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    config = load_config(cfg_path(args.config))
    scene_cfg = config.get("scene", {})
    humanoid_cfg = config.get("humanoid", {})
    car_cfg = config.get("car", {})
    sensor_cfg = config.get("sensors", {})

    checkpoint = resolve_repo_path(humanoid_cfg["checkpoint"])
    motion_file = resolve_repo_path(humanoid_cfg["motion_file"])
    scene_usd = resolve_repo_path(scene_cfg["scene_usd"])
    scene_map = resolve_repo_path(scene_cfg["scene_map"])
    validate_paths(
        {
            "Checkpoint": checkpoint,
            "Motion file": motion_file,
            "Scene USD": scene_usd,
            "Scene map": scene_map,
        }
    )

    if humanoid_cfg.get("human_mesh", False):
        enable_human_mesh(hide_humanoid=bool(humanoid_cfg.get("hide_humanoid", False)))

    fabric = create_fabric()
    launcher_args = {
        "headless": args.headless,
        "device": str(fabric.device),
    }
    if sensor_cfg.get("camera", {}).get("enabled", False):
        launcher_args["enable_cameras"] = True
    app_launcher = AppLauncher(launcher_args)
    suppress_known_isaaclab_warning_spam()

    robot_usd = resolve_robot_usd(car_cfg.get("usd"))
    crowd_robot_config = make_crowd_robot_config(car_cfg, sensor_cfg, robot_usd)
    scene_loaded_in_scene_cfg = args.scene_physics or crowd_robot_config is not None
    scene_prim_path = str(scene_cfg.get("prim_path", "/World/Scene"))
    scene_z_offset = float(scene_cfg.get("z_offset", 0.0))
    if scene_loaded_in_scene_cfg:
        patch_isaaclab_scene_with_crowdsim_assets(
            scene_usd_path=scene_usd,
            scene_z_offset=scene_z_offset,
            scene_prim_path=scene_prim_path,
            crowd_robot=crowd_robot_config,
        )

    runtime = build_runtime(
        checkpoint=checkpoint,
        motion_file=motion_file,
        num_envs=args.num_envs,
        headless=args.headless,
        simulation_app=app_launcher.app,
        fabric=fabric,
    )
    env = runtime.env
    configure_viewer_camera(env, config.get("viewer", {}), args.headless)

    if not scene_loaded_in_scene_cfg:
        add_global_usd_reference(scene_usd, prim_path=scene_prim_path, z_offset=scene_z_offset)
    apply_scene_visual_config(scene_cfg.get("scene_visual", {}), scene_prim_path)

    nav_manager = build_navigation_manager(
        scene_map=scene_map,
        num_humanoids=env.num_envs,
        num_robots=env.num_envs if crowd_robot_config is not None else 0,
        device=fabric.device,
        cfg=config,
    )

    if nav_manager is not None:
        spawn_xy = nav_manager.humanoid_starts_xy
        print(f"[CrowdSim] Navigation humanoid starts: {spawn_xy.cpu().tolist()}")
    else:
        spawn_xy = sample_or_parse_humanoid_spawns(scene_map, env.num_envs, fabric.device, config)
    apply_fixed_spawn_offsets(env, spawn_xy)

    if crowd_robot_config is not None:
        if nav_manager is not None:
            robot_spawn_xy_yaw = nav_manager.robot_starts_xy_yaw
            print(f"[CrowdSim] Navigation robot starts: {robot_spawn_xy_yaw.cpu().tolist()}")
        else:
            robot_spawn_xy_yaw = sample_or_parse_car_spawns(
                scene_map, env.num_envs, fabric.device, config
            )
        apply_fixed_crowd_robot_spawns(env, robot_spawn_xy_yaw)
        configure_robot_camera_recorder(
            env,
            RobotCameraStreamConfig(
                output_dir=resolve_repo_path(sensor_cfg.get("camera", {}).get("record_dir", "output/crowdsim_camera")),
                fps=float(sensor_cfg.get("camera", {}).get("record_fps", 10.0)),
                env_ids=str(sensor_cfg.get("camera", {}).get("record_envs", "0")),
                auto_record=bool(sensor_cfg.get("camera", {}).get("auto_record", False)),
            ),
        )
        print(f"[CrowdSim] Navigation robots ready: {env.num_envs}.")

    if nav_manager is not None:
        nav_manager.attach(env)

    print("[CrowdSim] Scene ready; starting MaskedMimic policy.")
    if args.full_eval:
        runtime.agent.evaluator.eval_count = 0
        print(runtime.agent.evaluator.evaluate())
    else:
        runtime.agent.evaluator.simple_test_policy(collect_metrics=True)


def validate_paths(paths: dict[str, Path]) -> None:
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


def sample_or_parse_humanoid_spawns(
    scene_map: Path,
    num_envs: int,
    device: torch.device,
    cfg: dict[str, Any],
) -> torch.Tensor:
    humanoid_cfg = cfg.get("humanoid", {})
    if humanoid_cfg.get("spawn_xy"):
        spawn_xy = parse_spawn_xy(humanoid_cfg["spawn_xy"], num_envs, device)
        print(f"[CrowdSim] Using manual humanoid spawn XY: {spawn_xy.cpu().tolist()}")
        return spawn_xy

    scene_cfg = cfg.get("scene", {})
    spawn_xy = sample_spawn_xy_from_map(
        map_path=scene_map,
        num_envs=num_envs,
        device=device,
        map_resolution=float(scene_cfg.get("resolution", 0.05002501250625312)),
        humanoid_radius=float(humanoid_cfg.get("radius", 0.45)),
        min_spacing=float(humanoid_cfg.get("spacing", 0.9)),
        free_threshold=int(scene_cfg.get("free_threshold", 200)),
        seed=int(humanoid_cfg.get("spawn_seed", 0)),
    )
    print(f"[CrowdSim] Sampled humanoid spawn XY: {spawn_xy.cpu().tolist()}")
    return spawn_xy


def sample_or_parse_car_spawns(
    scene_map: Path,
    num_envs: int,
    device: torch.device,
    cfg: dict[str, Any],
) -> torch.Tensor:
    car_cfg = cfg.get("car", {})
    if car_cfg.get("spawn_xy"):
        spawn_xy_yaw = parse_spawn_xy_yaw(car_cfg["spawn_xy"], num_envs, device)
        print(f"[CrowdSim] Using manual car spawn XY/yaw: {spawn_xy_yaw.cpu().tolist()}")
        return spawn_xy_yaw

    scene_cfg = cfg.get("scene", {})
    spawn_xy = sample_spawn_xy_from_map(
        map_path=scene_map,
        num_envs=num_envs,
        device=device,
        map_resolution=float(scene_cfg.get("resolution", 0.05002501250625312)),
        humanoid_radius=float(car_cfg.get("radius", 0.35)),
        min_spacing=float(car_cfg.get("spacing", 1.2)),
        free_threshold=int(scene_cfg.get("free_threshold", 200)),
        seed=int(car_cfg.get("spawn_seed", 1)),
    )
    spawn_xy_yaw = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
    spawn_xy_yaw[:, :2] = spawn_xy
    print(f"[CrowdSim] Sampled car spawn XY/yaw: {spawn_xy_yaw.cpu().tolist()}")
    return spawn_xy_yaw


def build_navigation_manager(
    scene_map: Path,
    num_humanoids: int,
    num_robots: int,
    device: torch.device,
    cfg: dict[str, Any],
) -> CrowdNavigationManager | None:
    nav_cfg = cfg.get("navigation", {})
    if not bool(nav_cfg.get("enabled", False)):
        return None

    scene_cfg = cfg.get("scene", {})
    humanoid_cfg = cfg.get("humanoid", {})
    path_cfg = nav_cfg.get("path", {})
    config = CrowdNavigationConfig(
        map_path=scene_map,
        map_resolution=float(scene_cfg.get("resolution", 0.05002501250625312)),
        free_threshold=int(scene_cfg.get("free_threshold", 200)),
        num_humanoids=num_humanoids,
        num_robots=num_robots,
        device=device,
        seed=int(path_cfg.get("seed", 7)),
        local_controller=str(nav_cfg.get("local_controller", "orca")),
        agent_radius=float(nav_cfg.get("agent_radius", 0.35)),
        humanoid_radius=float(humanoid_cfg.get("radius", 0.45)),
        safe_distance=float(nav_cfg.get("safe_distance", 0.25)),
        max_speed=float(nav_cfg.get("max_speed", 0.8)),
        waypoint_tolerance=float(nav_cfg.get("waypoint_tolerance", 0.45)),
        goal_tolerance=float(nav_cfg.get("goal_tolerance", 0.75)),
        min_start_goal_distance=float(path_cfg.get("min_start_goal_distance", 5.0)),
        min_spawn_spacing=float(path_cfg.get("min_spawn_spacing", 1.2)),
        neighbor_radius=float(nav_cfg.get("neighbor_radius", 4.0)),
        obstacle_query_radius=int(nav_cfg.get("obstacle_query_radius", 14)),
        max_obstacles=int(nav_cfg.get("max_obstacles", 16)),
        collision_distance=float(nav_cfg.get("collision_distance", 0.75)),
        log_interval=int(nav_cfg.get("log_interval", 120)),
        path_thin_spacing=float(path_cfg.get("path_thin_spacing", 0.35)),
    )
    return CrowdNavigationManager(config)


def apply_scene_visual_config(visual_cfg: dict[str, Any], scene_prim_path: str) -> None:
    if not bool(visual_cfg.get("deactivate", False)):
        return

    hidden = hide_prims_matching_keywords(
        root_prim_path=scene_prim_path,
        keywords=tuple(
            visual_cfg.get("deactivate_keywords", ["ceiling", "cube", "building", "door"])
        ),
        deactivate=True,
    )
    print(f"[CrowdSim] Deactivated {len(hidden)} scene prim(s).")


if __name__ == "__main__":
    main()
