"""Train a simple PPO policy for CrowdSim Jetbot navigation."""

from __future__ import annotations

import argparse
import csv
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
from CrowdSim.robot_ppo import RobotPPOConfig, RobotPPOTrainer, RobotRolloutBuffer  # noqa: E402
from CrowdSim.sensor_stream import RobotCameraStreamConfig, configure_robot_camera_recorder  # noqa: E402
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
    patch_isaaclab_scene_with_crowdsim_assets,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO local navigation policy for CrowdSim robots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="CrowdSim/config/cfg.yaml")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--render", action="store_true", help="Run with viewer instead of headless.")
    parser.add_argument("--scene-physics", action="store_true")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--save-interval", type=int, default=20_000)
    parser.add_argument("--output-dir", default="output/crowdsim_robot_ppo")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(cfg_path(args.config))
    config.setdefault("navigation", {})["enabled"] = True
    config["navigation"]["local_controller"] = "rl"

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
    headless = args.headless and not args.render
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

    if not scene_loaded_in_scene_cfg:
        add_global_usd_reference(scene_usd, prim_path=scene_prim_path, z_offset=scene_z_offset)
    apply_scene_visual_config(scene_cfg.get("scene_visual", {}), scene_prim_path)

    nav_manager = build_navigation_manager(
        scene_map=scene_map,
        num_humanoids=env.num_envs,
        num_robots=env.num_envs,
        device=fabric.device,
        cfg=config,
    )
    if nav_manager is None:
        raise RuntimeError("Navigation manager was not created.")

    apply_fixed_spawn_offsets(env, nav_manager.humanoid_starts_xy)
    apply_fixed_crowd_robot_spawns(env, nav_manager.robot_starts_xy_yaw)
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
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
    )
    ppo = RobotPPOTrainer(ppo_cfg, fabric.device)
    start_step = 0
    if args.resume is not None:
        start_step = ppo.load(Path(args.resume).expanduser().resolve())
        print(f"[CrowdSim][PPO] Resumed from {args.resume} at robot_step={start_step}.")

    buffer = RobotRolloutBuffer(
        rollout_steps=args.rollout_steps,
        num_envs=nav_manager.config.num_robots,
        obs_dim=nav_manager.robot_rl_obs_dim,
        action_dim=2,
        device=fabric.device,
    )
    train_loop(
        runtime=runtime,
        nav_manager=nav_manager,
        ppo=ppo,
        buffer=buffer,
        total_steps=args.total_steps,
        start_step=start_step,
        save_interval=args.save_interval,
        output_dir=Path(args.output_dir),
        use_tensorboard=not args.no_tensorboard,
    )


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
        print(
            f"[CrowdSim][PPO] robot_steps={robot_steps} "
            f"return={mean_return:.3f} len={mean_length:.1f} "
            f"reached={reached:.2f} collision={collision:.2f} "
            f"policy_loss={stats['policy_loss']:.4f} "
            f"value_loss={stats['value_loss']:.4f} entropy={stats['entropy']:.4f}"
        )
        logger.log(
            step=robot_steps,
            mean_return=mean_return,
            mean_length=mean_length,
            reached=reached,
            collision=collision,
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
        self.csv_path = self.output_dir / "train_metrics.csv"
        self.csv_file = self.csv_path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "step",
                "mean_return",
                "mean_length",
                "reached",
                "collision",
                "policy_loss",
                "value_loss",
                "entropy",
            ],
        )
        if self.csv_path.stat().st_size == 0:
            self.writer.writeheader()

        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(self.output_dir / "tb"))
            except ImportError as exc:
                print(f"[CrowdSim][PPO] TensorBoard disabled: {exc}")

    def log(self, step: int, **metrics: float) -> None:
        row = {"step": step, **metrics}
        self.writer.writerow(row)
        self.csv_file.flush()
        if self.tb is not None:
            for key, value in metrics.items():
                self.tb.add_scalar(key, value, step)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
        self.csv_file.close()


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


def cfg_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def validate_paths(paths: dict[str, Path]) -> None:
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


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
    car_cfg = cfg.get("car", {})
    path_cfg = nav_cfg.get("path", {})
    marker_cfg = nav_cfg.get("markers", {})
    rl_cfg = nav_cfg.get("rl", {})
    return CrowdNavigationManager(
        CrowdNavigationConfig(
            map_path=scene_map,
            map_resolution=float(scene_cfg.get("resolution", 0.05002501250625312)),
            free_threshold=int(scene_cfg.get("free_threshold", 200)),
            num_humanoids=num_humanoids,
            num_robots=num_robots,
            device=device,
            seed=int(path_cfg.get("seed", 7)),
            local_controller=str(nav_cfg.get("local_controller", "rl")),
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
            wheel_radius=float(car_cfg.get("wheel_radius", 0.0325)),
            wheel_base=float(car_cfg.get("wheel_base", 0.118)),
            max_wheel_speed=float(car_cfg.get("max_wheel_speed", 12.0)),
            heading_gain=float(car_cfg.get("heading_gain", 2.5)),
            wheel_joint_indices=parse_optional_int_pair(car_cfg.get("wheel_joint_indices")),
            visual_markers_enabled=bool(marker_cfg.get("enabled", False)),
            marker_update_interval=max(1, int(marker_cfg.get("update_interval", 10))),
            rl_num_neighbors=int(rl_cfg.get("num_neighbors", 4)),
            rl_num_obstacles=int(rl_cfg.get("num_obstacles", 8)),
            rl_obstacle_radius=float(rl_cfg.get("obstacle_radius", 4.0)),
            rl_action_yaw_rate=float(rl_cfg.get("action_yaw_rate", 2.5)),
            rl_progress_reward_scale=float(rl_cfg.get("progress_reward_scale", 4.0)),
            rl_goal_reward=float(rl_cfg.get("goal_reward", 10.0)),
            rl_collision_penalty=float(rl_cfg.get("collision_penalty", -10.0)),
            rl_time_penalty=float(rl_cfg.get("time_penalty", -0.01)),
            rl_max_episode_steps=int(rl_cfg.get("max_episode_steps", 600)),
        )
    )


def parse_optional_int_pair(value) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", ""}:
        return None
    items = list(value)
    if len(items) != 2:
        raise ValueError("wheel_joint_indices must be null or a two-item list.")
    return (int(items[0]), int(items[1]))


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
