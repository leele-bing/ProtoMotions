"""Kinematic playback of four SMPL humanoids in one global Office USD scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OFFICE_MAP_RESOLUTION = 100.0 / 1999.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay SMPL motions in one shared Office USD scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--motion-file", default="/home/pcl/amp/Assets/motion/amass_smpl_test.pt")
    parser.add_argument("--office-usd", default="/home/pcl/amp/Assets/Office/office.usd")
    parser.add_argument("--office-map", default="/home/pcl/amp/Assets/Office/office.png")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--spawn-xy",
        default=None,
        help="Manual semicolon-separated XY spawn offsets. If omitted, CrowdSim samples free points from office.png.",
    )
    parser.add_argument("--map-resolution", type=float, default=DEFAULT_OFFICE_MAP_RESOLUTION)
    parser.add_argument("--map-free-threshold", type=int, default=200)
    parser.add_argument("--spawn-radius", type=float, default=0.45)
    parser.add_argument("--spawn-spacing", type=float, default=0.9)
    parser.add_argument("--spawn-seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    return parser.parse_args()


args = parse_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch("isaaclab")

import torch  # noqa: E402

from CrowdSim.office_scene import (  # noqa: E402
    add_global_usd_reference,
    parse_spawn_xy,
    resolve_repo_path,
    sample_spawn_xy_from_map,
)


def main() -> None:
    motion_file = resolve_repo_path(args.motion_file)
    office_usd = Path(args.office_usd).expanduser().resolve()
    office_map = Path(args.office_map).expanduser().resolve()
    if not motion_file.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")
    if not office_usd.exists():
        raise FileNotFoundError(f"Office USD not found: {office_usd}")
    if not office_map.exists():
        raise FileNotFoundError(f"Office map not found: {office_map}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    app_launcher = AppLauncher({"headless": args.headless, "device": str(device)})

    from protomotions.components.motion_lib import MotionLib, MotionLibConfig
    from protomotions.components.scene_lib import SceneLib
    from protomotions.robot_configs.factory import robot_config
    from protomotions.simulator.base_simulator.simulator_state import ResetState
    from protomotions.simulator.factory import simulator_config
    from protomotions.utils.hydra_replacement import get_class

    robot_cfg = robot_config("smpl")
    robot_cfg.contact_bodies = None
    sim_cfg = simulator_config(
        "isaaclab",
        robot_cfg,
        headless=args.headless,
        num_envs=args.num_envs,
        experiment_name="crowdsim_office_kinematic",
    )

    simulator_cls = get_class(sim_cfg._target_)
    simulator = simulator_cls(
        config=sim_cfg,
        robot_config=robot_cfg,
        terrain=None,
        scene_lib=SceneLib.empty(num_envs=args.num_envs, device=device),
        device=device,
        simulation_app=app_launcher.app,
    )
    motion_lib = MotionLib(MotionLibConfig(motion_file=str(motion_file)), device=device)

    simulator._initialize_with_markers({})
    add_global_usd_reference(office_usd)

    env_ids = torch.arange(args.num_envs, dtype=torch.long, device=device)
    motion_ids = env_ids % motion_lib.num_motions()
    motion_times = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    spawn_offset = torch.zeros(args.num_envs, 3, dtype=torch.float32, device=device)
    if args.spawn_xy:
        spawn_xy = parse_spawn_xy(args.spawn_xy, args.num_envs, device)
        print(f"[CrowdSim] Using manual spawn XY: {spawn_xy.cpu().tolist()}")
    else:
        spawn_xy = sample_spawn_xy_from_map(
            map_path=office_map,
            num_envs=args.num_envs,
            device=device,
            map_resolution=args.map_resolution,
            humanoid_radius=args.spawn_radius,
            min_spacing=args.spawn_spacing,
            free_threshold=args.map_free_threshold,
            seed=args.spawn_seed,
        )
        print(f"[CrowdSim] Sampled spawn XY: {spawn_xy.cpu().tolist()}")
    spawn_offset[:, :2] = spawn_xy
    actions = torch.zeros(args.num_envs, robot_cfg.number_of_actions, device=device)

    step_count = 0
    try:
        while simulator.is_simulation_running():
            motion_times = torch.remainder(
                motion_times + simulator.dt, motion_lib.get_motion_length(motion_ids)
            )
            ref_state = motion_lib.get_motion_state(motion_ids, motion_times)
            reset_state = ResetState.from_robot_state(ref_state)
            reset_state.root_pos = reset_state.root_pos + spawn_offset
            simulator.reset_envs(reset_state, env_ids=env_ids)
            simulator.step(actions)

            step_count += 1
            if args.max_steps > 0 and step_count >= args.max_steps:
                break
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
