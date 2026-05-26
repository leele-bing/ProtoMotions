"""Train a simple PPO policy for CrowdSim Jetbot navigation."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch("isaaclab")

import torch  # noqa: E402

torch.set_float32_matmul_precision("high")

from CrowdSim.utils.humanoid_state_recorder import (  # noqa: E402
    HumanoidStateRecorderConfig,
    configure_humanoid_state_recorder,
)
from CrowdSim.utils.map_metadata import OccupancyMapMetadata, load_occupancy_map_metadata  # noqa: E402
from CrowdSim.navigation import CrowdNavigationConfig, CrowdNavigationManager  # noqa: E402
from CrowdSim.differential_control import DifferentialDriveConfig  # noqa: E402
from CrowdSim.robot_ppo import RobotPPOConfig, RobotPPOTrainer, RobotRolloutBuffer  # noqa: E402
from CrowdSim.utils.sensor_stream import RobotCameraStreamConfig, configure_robot_camera_recorder  # noqa: E402
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
    patch_isaaclab_scene_with_crowdsim_assets,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO local navigation policy for CrowdSim robots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Environment config path. Kept as a compatibility alias for --env-config.",
    )
    parser.add_argument("--env-config", default="CrowdSim/config/env.yaml")
    parser.add_argument("--ppo-config", default="CrowdSim/config/ppo.yaml")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--scene-physics", action="store_true")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config_path = cfg_path(args.config or args.env_config)
    config = load_config(env_config_path)
    ppo_config_path = cfg_path(args.ppo_config)
    ppo_config = load_config(ppo_config_path) if ppo_config_path.exists() else {}
    merge_ppo_config(config, ppo_config)
    config.setdefault("navigation", {})["enabled"] = True
    config["navigation"].setdefault("local", {})["method"] = "rl"
    config["navigation"]["local"]["method"] = "rl"
    training_cfg = get_config_section(ppo_config, "training")
    network_cfg = get_config_section(ppo_config, "network")

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
    headless = args.headless
    launcher_args = {"headless": headless, "device": str(fabric.device)}
    if sensor_cfg.get("camera", {}).get("enabled", False):
        launcher_args["enable_cameras"] = True
    app_launcher = AppLauncher(launcher_args)
    suppress_known_isaaclab_warning_spam()

    robot_usd = resolve_robot_usd(car_cfg.get("usd"))
    crowd_robot_config = make_crowd_robot_config(car_cfg, sensor_cfg, robot_usd)
    if crowd_robot_config is None:
        raise RuntimeError("Robot PPO training requires car.usd to be configured.")

    scene_prim_path = str(scene_cfg.get("prim_path", "/World/Scene"))
    scene_z_offset = float(scene_cfg.get("z_offset", 0.0))
    scene_loaded_in_scene_cfg = args.scene_physics or crowd_robot_config is not None
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
        headless=headless,
        simulation_app=app_launcher.app,
        fabric=fabric,
    )
    env = runtime.env
    configure_viewer_camera(env, config.get("viewer", {}), headless)
    configure_humanoid_state_recording(env, config)

    if not scene_loaded_in_scene_cfg:
        add_global_usd_reference(scene_usd, prim_path=scene_prim_path, z_offset=scene_z_offset)

    nav_manager = build_navigation_manager(
        map_metadata=map_metadata,
        num_humanoids=env.num_envs,
        num_robots=env.num_envs,
        device=fabric.device,
        cfg=config,
    )
    if nav_manager is None:
        raise RuntimeError("Navigation manager was not created.")

    apply_fixed_spawn_offsets(env, nav_manager.humanoid_starts_xy)
    apply_fixed_crowd_robot_spawns(env, nav_manager.robot_starts_xy_yaw)
    if sensor_cfg.get("camera", {}).get("enabled", False):
        configure_robot_camera_recorder(
            env,
            RobotCameraStreamConfig(
                output_dir=resolve_repo_path(
                    sensor_cfg.get("camera", {}).get("record_dir", "output/crowdsim_camera")
                ),
                fps=float(sensor_cfg.get("camera", {}).get("record_fps", 10.0)),
                env_ids=str(sensor_cfg.get("camera", {}).get("record_envs", "0")),
                auto_record=bool(sensor_cfg.get("camera", {}).get("auto_record", False)),
            ),
        )
    nav_manager.attach(env)

    ppo_cfg = RobotPPOConfig(
        obs_dim=nav_manager.robot_rl_obs_dim,
        vector_obs_dim=nav_manager.robot_rl_vector_obs_dim,
        map_size=nav_manager.config.rl_map_size,
        hidden_dim=int(config_value(args.hidden_dim, network_cfg, "hidden_dim", 128)),
        lr=float(config_value(args.lr, training_cfg, "lr", 3.0e-4)),
        ppo_epochs=int(config_value(args.ppo_epochs, training_cfg, "ppo_epochs", 4)),
        minibatch_size=int(config_value(args.minibatch_size, training_cfg, "minibatch_size", 256)),
    )
    ppo = RobotPPOTrainer(ppo_cfg, fabric.device)
    start_step = 0
    if args.resume is not None:
        start_step = ppo.load(Path(args.resume).expanduser().resolve())
        print(f"[CrowdSim][PPO] Resumed from {args.resume} at robot_step={start_step}.")

    buffer = RobotRolloutBuffer(
        rollout_steps=int(config_value(args.rollout_steps, training_cfg, "rollout_steps", 256)),
        num_envs=nav_manager.config.num_robots,
        obs_dim=nav_manager.robot_rl_obs_dim,
        action_dim=2,
        device=fabric.device,
    )
    base_output_dir = resolve_repo_path(
        str(config_value(args.output_dir, training_cfg, "output_dir", "output/crowdsim_robot_ppo"))
    )
    output_dir = make_training_output_dir(base_output_dir)
    copy_training_configs(output_dir, env_config_path, ppo_config_path)
    print(f"[CrowdSim][PPO] Output directory: {output_dir}")

    train_loop(
        runtime=runtime,
        nav_manager=nav_manager,
        ppo=ppo,
        buffer=buffer,
        total_steps=int(config_value(args.total_steps, training_cfg, "total_steps", 200_000)),
        start_step=start_step,
        save_interval=int(config_value(args.save_interval, training_cfg, "save_interval", 20_000)),
        output_dir=output_dir,
        use_tensorboard=not args.no_tensorboard,
    )


def make_training_output_dir(base_output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / timestamp
    suffix = 1
    while output_dir.exists():
        output_dir = base_output_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    latest_link = base_output_dir / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(output_dir.name, target_is_directory=True)
    except OSError:
        pass
    return output_dir


def copy_training_configs(output_dir: Path, env_config_path: Path, ppo_config_path: Path) -> None:
    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(env_config_path, config_dir / "env.yaml")
    if ppo_config_path.exists():
        shutil.copy2(ppo_config_path, config_dir / "ppo.yaml")


def train_loop(
    runtime,
    nav_manager: CrowdNavigationManager,
    ppo: RobotPPOTrainer,
    buffer: RobotRolloutBuffer,
    total_steps: int,
    start_step: int,
    save_interval: int,
    output_dir: Path,
    use_tensorboard: bool,
) -> None:
    env = runtime.env
    agent = runtime.agent
    robot_steps = start_step
    done_indices = None
    episode_returns = torch.zeros(nav_manager.config.num_robots, device=ppo.device)
    episode_lengths = torch.zeros(nav_manager.config.num_robots, device=ppo.device)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    next_save = ((robot_steps // save_interval) + 1) * save_interval if save_interval > 0 else None
    logger = RobotTrainingLogger(output_dir, use_tensorboard)

    print(
        f"[CrowdSim][PPO] Training robot policy: robots={nav_manager.config.num_robots}, "
        f"obs_dim={nav_manager.robot_rl_obs_dim}, total_robot_steps={total_steps}."
    )
    while robot_steps < total_steps:
        buffer.reset()
        for _ in range(buffer.rollout_steps):
            obs, _ = env.reset(done_indices)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)

            with torch.no_grad():
                model_outs = agent.model(obs_td)
                humanoid_action = model_outs.get("mean_action", model_outs["action"])
                robot_obs = nav_manager.get_robot_rl_observations()
                robot_action, raw_action, log_prob, value = ppo.act(robot_obs)

            nav_manager.set_robot_rl_actions(robot_action)
            _, _, humanoid_dones, _, _ = env.step(humanoid_action)
            _, robot_reward, robot_done, info = nav_manager.get_robot_rl_feedback()

            buffer.add(robot_obs, raw_action, log_prob, robot_reward, robot_done, value)
            episode_returns += robot_reward
            episode_lengths += 1

            if robot_done.any():
                done_ids = robot_done.nonzero(as_tuple=False).squeeze(-1)
                completed_returns.extend(episode_returns[done_ids].detach().cpu().tolist())
                completed_lengths.extend(episode_lengths[done_ids].detach().cpu().tolist())
                episode_returns[done_ids] = 0.0
                episode_lengths[done_ids] = 0.0
                nav_manager.reset_robot_rl_episodes(robot_done)

            done_indices = humanoid_dones.nonzero(as_tuple=False).squeeze(-1)
            robot_steps += nav_manager.config.num_robots

        with torch.no_grad():
            last_value = ppo.value(nav_manager.get_robot_rl_observations())
        buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=ppo.config.gamma,
            gae_lambda=ppo.config.gae_lambda,
        )
        stats = ppo.update(buffer)

        recent_returns = completed_returns[-100:]
        recent_lengths = completed_lengths[-100:]
        mean_return = sum(recent_returns) / max(len(recent_returns), 1)
        mean_length = sum(recent_lengths) / max(len(recent_lengths), 1)
        reached = info.get("reached", torch.zeros_like(robot_done)).float().mean().item()
        collision = info.get("collision", torch.zeros_like(robot_done)).float().mean().item()
        timeout = info.get("timeout", torch.zeros_like(robot_done)).float().mean().item()
        goal_distance = info.get("distance_to_goal", torch.zeros_like(robot_reward)).float().mean().item()
        progress = info.get("progress", torch.zeros_like(robot_reward)).float().mean().item()
        reward_total = info.get("reward_total", robot_reward).float().mean().item()
        reward_progress = info.get("reward_progress", torch.zeros_like(robot_reward)).float().mean().item()
        reward_goal = info.get("reward_goal", torch.zeros_like(robot_reward)).float().mean().item()
        reward_collision = info.get("reward_collision", torch.zeros_like(robot_reward)).float().mean().item()
        reward_timeout = info.get("reward_timeout", torch.zeros_like(robot_reward)).float().mean().item()
        reward_terminal = reward_goal + reward_collision + reward_timeout
        print(
            f"[CrowdSim][PPO] robot_steps={robot_steps} "
            f"return={mean_return:.3f} len={mean_length:.1f} "
            f"reached={reached:.2f} collision={collision:.2f} timeout={timeout:.2f} "
            f"goal_distance={goal_distance:.3f} progress={progress:.3f} "
            f"reward={reward_total:.3f} progress_reward={reward_progress:.3f} "
            f"policy_loss={stats['policy_loss']:.4f} "
            f"value_loss={stats['value_loss']:.4f} entropy={stats['entropy']:.4f}"
        )
        logger.log(
            step=robot_steps,
            mean_return=mean_return,
            mean_length=mean_length,
            reached=reached,
            collision=collision,
            timeout=timeout,
            goal_distance=goal_distance,
            progress=progress,
            reward_total=reward_total,
            reward_progress=reward_progress,
            reward_terminal=reward_terminal,
            policy_loss=stats["policy_loss"],
            value_loss=stats["value_loss"],
            entropy=stats["entropy"],
        )

        if next_save is not None and robot_steps >= next_save:
            ppo.save(output_dir / "robot_ppo_latest.pt", robot_steps)
            ppo.save(output_dir / f"robot_ppo_{robot_steps}.pt", robot_steps)
            next_save += save_interval

    ppo.save(output_dir / "robot_ppo_latest.pt", robot_steps)
    logger.close()
    print(f"[CrowdSim][PPO] Saved final checkpoint to {output_dir / 'robot_ppo_latest.pt'}")


class RobotTrainingLogger:
    def __init__(self, output_dir: Path, use_tensorboard: bool) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(self.output_dir / "tb"))
            except ImportError as exc:
                print(f"[CrowdSim][PPO] TensorBoard disabled: {exc}")

    def log(self, step: int, **metrics: float) -> None:
        if self.tb is not None:
            self.tb.add_scalars(
                "loss_entropy",
                {
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy": metrics["entropy"],
                },
                step,
            )
            self.tb.add_scalars(
                "reward",
                {
                    "total": metrics["reward_total"],
                    "progress": metrics["reward_progress"],
                    "terminal": metrics["reward_terminal"],
                },
                step,
            )
            self.tb.add_scalars(
                "outcome",
                {
                    "reached": metrics["reached"],
                    "collision": metrics["collision"],
                    "timeout": metrics["timeout"],
                },
                step,
            )
            self.tb.add_scalars(
                "mean",
                {
                    "return": metrics["mean_return"],
                    "length": metrics["mean_length"],
                    "goal_distance": metrics["goal_distance"],
                    "progress": metrics["progress"],
                },
                step,
            )

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()


def load_config(path: Path) -> dict[str, Any]:
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
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value.strip() == "":
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = parse_scalar(raw_value.strip())
    return root


def parse_scalar(value: str):
    value = strip_inline_comment(value).strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(item.strip()) for item in inner.split(",")]
    if "," in value:
        return [parse_scalar(item.strip()) for item in value.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


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
    return CrowdNavigationManager(
        CrowdNavigationConfig(
            map_path=map_metadata.image_path,
            map_resolution=map_metadata.resolution,
            free_threshold=map_metadata.free_threshold,
            map_origin_xy=map_metadata.origin_xy,
            num_humanoids=num_humanoids,
            num_robots=num_robots,
            device=device,
            seed=int(path_cfg.get("seed", 7)),
            robot_control_mode=str(local_cfg.get("method", "rl")),
            agent_radius=float(local_cfg.get("agent_radius", 0.35)),
            safe_distance=float(local_cfg.get("safe_distance", 0.75)),
            max_speed=float(local_cfg.get("max_speed", 1.5)),
            waypoint_tolerance=float(nav_cfg.get("waypoint_tolerance", 0.45)),
            goal_tolerance=float(nav_cfg.get("goal_tolerance", 0.75)),
            min_start_goal_distance=float(path_cfg.get("min_start_goal_distance", 5.0)),
            max_start_goal_distance=float(path_cfg.get("max_start_goal_distance", 10.0)),
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
            differential_drive=DifferentialDriveConfig(),
            rl_num_neighbors=int(rl_cfg.get("num_neighbors", 4)),
            rl_progress_reward_scale=float(rl_cfg.get("progress_reward_scale", 4.0)),
            rl_goal_reward=float(rl_cfg.get("goal_reward", 10.0)),
            rl_collision_penalty=float(rl_cfg.get("collision_penalty", -10.0)),
            rl_timeout_penalty=float(rl_cfg.get("timeout_penalty", -5.0)),
            rl_time_penalty=float(rl_cfg.get("time_penalty", -0.01)),
            rl_max_episode_steps=int(rl_cfg.get("max_episode_steps", 600)),
            rl_map_size=int(rl_cfg.get("map_size", 24)),
            rl_map_extent=float(rl_cfg.get("map_extent", 8.0)),
        )
    )


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


def get_config_section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    section = cfg.get(key, {})
    return section if isinstance(section, dict) else {}


def config_value(cli_value, cfg: dict[str, Any], key: str, default):
    if cli_value is not None:
        return cli_value
    return cfg.get(key, default)


if __name__ == "__main__":
    main()
