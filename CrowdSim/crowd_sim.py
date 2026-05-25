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

from CrowdSim.utils.humanoid_state_recorder import (  # noqa: E402
    HumanoidStateRecorderConfig,
    configure_humanoid_state_recorder,
)
from CrowdSim.utils.map_metadata import OccupancyMapMetadata, load_occupancy_map_metadata  # noqa: E402
from CrowdSim.navigation import CrowdNavigationConfig, CrowdNavigationManager  # noqa: E402
from CrowdSim.utils.sensor_stream import (  # noqa: E402
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
    parser.add_argument(
        "--config",
        default=None,
        help="Environment config path. Kept as a compatibility alias for --env-config.",
    )
    parser.add_argument("--env-config", default="CrowdSim/config/env.yaml")
    parser.add_argument("--ppo-config", default="CrowdSim/config/ppo.yaml")
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
    text = strip_inline_comment(value).strip()
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


def strip_inline_comment(value: str) -> str:
    quote: str | None = None
    for idx, char in enumerate(value):
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        elif char == "#" and quote is None:
            return value[:idx]
    return value


def cfg_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    config = load_config(cfg_path(args.config or args.env_config))
    ppo_config_path = cfg_path(args.ppo_config)
    if ppo_config_path.exists():
        merge_ppo_config(config, load_config(ppo_config_path))
    scene_cfg = config.get("scene", {})
    humanoid_cfg = config.get("humanoid", {})
    car_cfg = config.get("car", {})
    sensor_cfg = config.get("sensors", {})

    checkpoint = resolve_repo_path(humanoid_cfg["checkpoint"])
    motion_file = resolve_repo_path(humanoid_cfg["motion_file"])
    scene_usd = resolve_repo_path(scene_cfg["scene_usd"])
    scene_map_ref = resolve_repo_path(scene_cfg["scene_map"])
    map_metadata = load_occupancy_map_metadata(scene_map_ref)
    validate_paths(
        {
            "Checkpoint": checkpoint,
            "Motion file": motion_file,
            "Scene USD": scene_usd,
            "Scene map image": map_metadata.image_path,
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
    configure_humanoid_state_recording(env, config)

    if not scene_loaded_in_scene_cfg:
        add_global_usd_reference(scene_usd, prim_path=scene_prim_path, z_offset=scene_z_offset)

    nav_manager = build_navigation_manager(
        map_metadata=map_metadata,
        num_humanoids=env.num_envs,
        num_robots=env.num_envs if crowd_robot_config is not None else 0,
        device=fabric.device,
        cfg=config,
    )

    if nav_manager is not None:
        spawn_xy = nav_manager.humanoid_starts_xy
        print(f"[CrowdSim] Navigation humanoid starts: {spawn_xy.cpu().tolist()}")
    else:
        spawn_xy = sample_or_parse_humanoid_spawns(
            map_metadata, env.num_envs, fabric.device, config
        )
    apply_fixed_spawn_offsets(env, spawn_xy)

    if crowd_robot_config is not None:
        if nav_manager is not None:
            robot_spawn_xy_yaw = nav_manager.robot_starts_xy_yaw
            print(f"[CrowdSim] Navigation robot starts: {robot_spawn_xy_yaw.cpu().tolist()}")
        else:
            robot_spawn_xy_yaw = sample_or_parse_car_spawns(
                map_metadata, env.num_envs, fabric.device, config
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
    elif nav_manager is not None and nav_manager.config.robot_control_mode.lower() == "rl":
        run_masked_mimic_with_robot_ppo(runtime, nav_manager, config)
    else:
        runtime.agent.evaluator.simple_test_policy(collect_metrics=True)


def validate_paths(paths: dict[str, Path]) -> None:
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


def configure_humanoid_state_recording(env, cfg: dict[str, Any]):
    record_cfg = cfg.get("humanoid", {}).get("state_recording", {})
    if not isinstance(record_cfg, dict) or not bool(record_cfg.get("enabled", False)):
        return None

    return configure_humanoid_state_recorder(
        env,
        HumanoidStateRecorderConfig(
            output_dir=resolve_repo_path(
                record_cfg.get("record_dir", "output/crowdsim_humanoid_state")
            ),
            fps=float(record_cfg.get("record_fps", 30.0)),
            env_ids=str(record_cfg.get("record_envs", "0")),
            auto_record=bool(record_cfg.get("auto_record", False)),
            key=str(record_cfg.get("key", "H")),
        ),
    )


def sample_or_parse_humanoid_spawns(
    map_metadata: OccupancyMapMetadata,
    num_envs: int,
    device: torch.device,
    cfg: dict[str, Any],
) -> torch.Tensor:
    humanoid_cfg = cfg.get("humanoid", {})
    path_cfg = cfg.get("navigation", {}).get("path", {})
    if humanoid_cfg.get("spawn_xy"):
        spawn_xy = parse_spawn_xy(humanoid_cfg["spawn_xy"], num_envs, device)
        print(f"[CrowdSim] Using manual humanoid spawn XY: {spawn_xy.cpu().tolist()}")
        return spawn_xy

    spawn_xy = sample_spawn_xy_from_map(
        map_path=map_metadata.image_path,
        num_envs=num_envs,
        device=device,
        map_resolution=map_metadata.resolution,
        map_origin_xy=map_metadata.origin_xy,
        humanoid_radius=float(path_cfg.get("planning_clearance", 0.2)),
        min_spacing=float(path_cfg.get("min_spawn_spacing", 1.2)),
        free_threshold=map_metadata.free_threshold,
        seed=int(path_cfg.get("seed", 7)),
    )
    print(f"[CrowdSim] Sampled humanoid spawn XY: {spawn_xy.cpu().tolist()}")
    return spawn_xy


def sample_or_parse_car_spawns(
    map_metadata: OccupancyMapMetadata,
    num_envs: int,
    device: torch.device,
    cfg: dict[str, Any],
) -> torch.Tensor:
    car_cfg = cfg.get("car", {})
    path_cfg = cfg.get("navigation", {}).get("path", {})
    if car_cfg.get("spawn_xy"):
        spawn_xy_yaw = parse_spawn_xy_yaw(car_cfg["spawn_xy"], num_envs, device)
        print(f"[CrowdSim] Using manual car spawn XY/yaw: {spawn_xy_yaw.cpu().tolist()}")
        return spawn_xy_yaw

    spawn_xy = sample_spawn_xy_from_map(
        map_path=map_metadata.image_path,
        num_envs=num_envs,
        device=device,
        map_resolution=map_metadata.resolution,
        map_origin_xy=map_metadata.origin_xy,
        humanoid_radius=float(path_cfg.get("planning_clearance", 0.2)),
        min_spacing=float(path_cfg.get("min_spawn_spacing", 1.2)),
        free_threshold=map_metadata.free_threshold,
        seed=int(path_cfg.get("seed", 7)) + 1,
    )
    spawn_xy_yaw = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
    spawn_xy_yaw[:, :2] = spawn_xy
    print(f"[CrowdSim] Sampled car spawn XY/yaw: {spawn_xy_yaw.cpu().tolist()}")
    return spawn_xy_yaw


def build_navigation_manager(
    map_metadata: OccupancyMapMetadata,
    num_humanoids: int,
    num_robots: int,
    device: torch.device,
    cfg: dict[str, Any],
) -> CrowdNavigationManager | None:
    nav_cfg = cfg.get("navigation", {})
    if not bool(nav_cfg.get("enabled", False)):
        return None

    path_cfg = nav_cfg.get("path", {})
    local_cfg = nav_cfg.get("local", {})
    recording_cfg = nav_cfg.get("recording", {})
    humanoid_cfg = cfg.get("humanoid", {})
    marker_cfg = get_marker_config(cfg)
    rl_cfg = get_robot_rl_config(cfg)
    config = CrowdNavigationConfig(
        map_path=map_metadata.image_path,
        map_resolution=map_metadata.resolution,
        free_threshold=map_metadata.free_threshold,
        map_origin_xy=map_metadata.origin_xy,
        num_humanoids=num_humanoids,
        num_robots=num_robots,
        device=device,
        seed=int(path_cfg.get("seed", 7)),
        robot_control_mode=str(local_cfg.get("method", "sfm")),
        agent_radius=float(local_cfg.get("agent_radius", 0.35)),
        safe_distance=float(local_cfg.get("safe_distance", 0.75)),
        max_speed=float(local_cfg.get("max_speed", 1.5)),
        waypoint_tolerance=float(nav_cfg.get("waypoint_tolerance", 0.45)),
        goal_tolerance=float(nav_cfg.get("goal_tolerance", 0.75)),
        min_start_goal_distance=float(path_cfg.get("min_start_goal_distance", 5.0)),
        min_spawn_spacing=float(path_cfg.get("min_spawn_spacing", 1.2)),
        planning_step_size=float(path_cfg.get("planning_step_size", 0.5)),
        planning_clearance=float(path_cfg.get("planning_clearance", 0.2)),
        neighbor_radius=float(local_cfg.get("neighbor_radius", 4.0)),
        collision_distance=float(nav_cfg.get("collision_distance", 0.75)),
        log_interval=int(nav_cfg.get("log_interval", 120)),
        update_hz=float(nav_cfg.get("update_hz", 30.0)),
        trajectory_recording_enabled=bool(recording_cfg.get("enabled", True)),
        trajectory_output_dir=resolve_repo_path(
            recording_cfg.get("output_dir", "output/crowdsim_navigation")
        ),
        visual_markers_enabled=bool(marker_cfg.get("enabled", False)),
        humanoid_target_enabled=bool(humanoid_cfg.get("target_enabled", True)),
        local_target_timestep=float(local_cfg.get("target_timestep", 1.0)),
        humanoid_target_min_heading_speed=float(
            humanoid_cfg.get("min_heading_speed", 0.05)
        ),
        rl_num_neighbors=int(rl_cfg.get("num_neighbors", 4)),
        rl_num_obstacles=int(rl_cfg.get("num_obstacles", 8)),
        rl_obstacle_radius=float(rl_cfg.get("obstacle_radius", 4.0)),
        rl_progress_reward_scale=float(rl_cfg.get("progress_reward_scale", 4.0)),
        rl_goal_reward=float(rl_cfg.get("goal_reward", 10.0)),
        rl_collision_penalty=float(rl_cfg.get("collision_penalty", -10.0)),
        rl_time_penalty=float(rl_cfg.get("time_penalty", -0.01)),
        rl_max_episode_steps=int(rl_cfg.get("max_episode_steps", 600)),
    )
    return CrowdNavigationManager(config)


def run_masked_mimic_with_robot_ppo(
    runtime,
    nav_manager: CrowdNavigationManager,
    cfg: dict[str, Any],
) -> None:
    rl_cfg = get_robot_rl_config(cfg)
    checkpoint = rl_cfg.get("policy_checkpoint")
    if checkpoint is None:
        raise RuntimeError(
            "navigation.local.method is 'rl', but rl.policy_checkpoint is not set in the PPO config. "
            "Train a policy with CrowdSim/train_robot_ppo.py first, then point this field at the checkpoint."
        )

    from CrowdSim.robot_ppo import RobotPPOConfig, RobotPPOTrainer

    ppo = RobotPPOTrainer(
        RobotPPOConfig(
            obs_dim=nav_manager.robot_rl_obs_dim,
            hidden_dim=int(rl_cfg.get("hidden_dim", 128)),
        ),
        nav_manager.config.device,
    )
    loaded_step = ppo.load(resolve_repo_path(str(checkpoint)))
    deterministic = bool(rl_cfg.get("deterministic", True))
    print(f"[CrowdSim] Loaded robot PPO policy from {checkpoint} at robot_step={loaded_step}.")

    agent = runtime.agent
    env = runtime.env
    agent.eval()
    done_indices = None
    step = 0
    print("Evaluating MaskedMimic + robot PPO policy... (Ctrl+C to stop)")
    try:
        while True:
            obs, _ = env.reset(done_indices)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)
            with torch.no_grad():
                model_outs = agent.model(obs_td)
                humanoid_action = model_outs.get("mean_action", model_outs["action"])
                robot_obs = nav_manager.get_robot_rl_observations()
                if deterministic:
                    mean, _, _ = ppo.model(robot_obs)
                    robot_action = torch.tanh(mean)
                else:
                    robot_action, _, _, _ = ppo.act(robot_obs)

            nav_manager.set_robot_rl_actions(robot_action)
            _, _, dones, _, _ = env.step(humanoid_action)
            _, _, robot_done, _ = nav_manager.get_robot_rl_feedback()
            nav_manager.reset_robot_rl_episodes(robot_done)
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            step += 1
    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps.")


def merge_ppo_config(env_cfg: dict[str, Any], ppo_cfg: dict[str, Any]) -> None:
    rl_cfg = ppo_cfg.get("rl", ppo_cfg)
    if not isinstance(rl_cfg, dict):
        return
    env_cfg.setdefault("navigation", {})["rl"] = rl_cfg


def get_robot_rl_config(cfg: dict[str, Any]) -> dict[str, Any]:
    nav_cfg = cfg.get("navigation", {})
    rl_cfg = nav_cfg.get("rl", cfg.get("rl", {}))
    return rl_cfg if isinstance(rl_cfg, dict) else {}


def get_marker_config(cfg: dict[str, Any]) -> dict[str, Any]:
    nav_cfg = cfg.get("navigation", {})
    marker_cfg = cfg.get("markers", nav_cfg.get("markers", {}))
    return marker_cfg if isinstance(marker_cfg, dict) else {}


if __name__ == "__main__":
    main()
