# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
"""Play four SMPL humanoids inside one global Office USD scene.

This is a visualization script: it uses kinematic replay to place SMPL bodies
directly on reference AMASS motions, bypassing policy inference and physics
tracking. The Office USD is loaded once at /World/Office, outside the cloned env
namespace, so all four SMPL envs share the same global visual scene.

Example:
    python examples/office_four_smpl_kinematic.py \
        --motion-file ../Amass/motion/amass_smpl_train.pt
"""

import argparse
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play four SMPL humanoids in one global Office USD scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--simulator", default="isaaclab", choices=["isaaclab"])
    parser.add_argument("--robot-name", default="smpl", choices=["smpl"])
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--motion-file",
        default="../Amass/motion/amass_smpl_train.pt",
        help="MotionLib .pt file to play back.",
    )
    parser.add_argument(
        "--office-usd",
        default="/home/pcl/amp/assets/Office/office.usd",
        help="Local Office root USD file.",
    )
    parser.add_argument(
        "--office-z",
        type=float,
        default=0.0,
        help="Vertical offset applied to the Office asset.",
    )
    parser.add_argument(
        "--spawn-xy",
        default="-1.5,-1.5;1.5,-1.5;-1.5,1.5;1.5,1.5",
        help="Semicolon-separated XY spawn offsets, one per humanoid.",
    )
    parser.add_argument("--experiment-name", default="office_four_smpl")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after this many steps. 0 runs until the Isaac Sim window closes.",
    )
    return parser


parser = create_parser()
args = parser.parse_args()

# IsaacLab/IsaacSim must be imported before torch.
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import torch  # noqa: E402


def parse_spawn_xy(value: str, num_envs: int, device: torch.device) -> torch.Tensor:
    pairs = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        x_str, y_str = item.split(",", maxsplit=1)
        pairs.append((float(x_str), float(y_str)))

    if not pairs:
        raise ValueError("--spawn-xy must contain at least one x,y pair")

    while len(pairs) < num_envs:
        pairs.append(pairs[len(pairs) % len(pairs)])

    return torch.tensor(pairs[:num_envs], dtype=torch.float32, device=device)


def add_global_office_reference(office_usd: Path, office_z: float) -> None:
    """Add Office as a global visual reference after the physics scene exists."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    office_prim = stage.DefinePrim("/World/Office", "Xform")
    office_prim.GetReferences().AddReference(str(office_usd))
    UsdGeom.XformCommonAPI(office_prim).SetTranslate((0.0, 0.0, office_z))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    office_usd = Path(args.office_usd).expanduser().resolve()
    motion_file = Path(args.motion_file).expanduser()
    if not motion_file.is_absolute():
        motion_file = (repo_root / motion_file).resolve()

    if not office_usd.exists():
        raise FileNotFoundError(f"Office USD not found: {office_usd}")
    if not motion_file.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    device = torch.device("cpu" if args.cpu_only else "cuda:0")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    app_launcher = AppLauncher({"headless": args.headless, "device": str(device)})
    simulation_app = app_launcher.app

    from protomotions.robot_configs.factory import robot_config
    from protomotions.simulator.factory import simulator_config
    from protomotions.components.motion_lib import MotionLibConfig
    from protomotions.components.motion_lib import MotionLib
    from protomotions.components.scene_lib import SceneLib
    from protomotions.simulator.base_simulator.simulator_state import ResetState
    from protomotions.utils.hydra_replacement import get_class

    robot_cfg = robot_config(args.robot_name)
    robot_cfg.contact_bodies = None

    sim_cfg = simulator_config(
        args.simulator,
        robot_cfg,
        headless=args.headless,
        num_envs=args.num_envs,
        experiment_name=args.experiment_name,
    )

    scene_lib = SceneLib.empty(num_envs=args.num_envs, device=device)
    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=str(motion_file)), device=device
    )

    SimulatorClass = get_class(sim_cfg._target_)
    simulator = SimulatorClass(
        config=sim_cfg,
        robot_config=robot_cfg,
        terrain=None,
        scene_lib=scene_lib,
        device=device,
        simulation_app=simulation_app,
    )
    simulator._initialize_with_markers({})
    add_global_office_reference(office_usd, args.office_z)

    spawn_xy = parse_spawn_xy(args.spawn_xy, args.num_envs, device)
    spawn_offset = torch.zeros(args.num_envs, 3, dtype=torch.float32, device=device)
    spawn_offset[:, :2] = spawn_xy

    actions = torch.zeros(args.num_envs, robot_cfg.number_of_actions, device=device)
    env_ids = torch.arange(args.num_envs, dtype=torch.long, device=device)
    motion_ids = (
        torch.arange(args.num_envs, dtype=torch.long, device=device)
        % motion_lib.num_motions()
    )
    motion_times = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    print("\n=== Office SMPL Kinematic Playback ===")
    print(f"Office USD: {office_usd}")
    print(f"Motion file: {motion_file}")
    print(f"Humanoids: {args.num_envs}")
    print(f"Spawn XY: {spawn_xy.cpu().tolist()}")
    print("Close the Isaac Sim window, or press Ctrl-C in the terminal, to stop.\n")

    step_count = 0
    try:
        while simulator.is_simulation_running():
            motion_lengths = motion_lib.get_motion_length(motion_ids)
            motion_times = torch.remainder(
                motion_times + simulator.dt, motion_lengths
            )
            ref_state = motion_lib.get_motion_state(motion_ids, motion_times)
            ref_reset_state = ResetState.from_robot_state(ref_state)
            ref_reset_state.root_pos = ref_reset_state.root_pos + spawn_offset

            simulator.reset_envs(ref_reset_state, env_ids=env_ids)
            simulator.step(actions)

            step_count += 1
            if args.max_steps > 0 and step_count >= args.max_steps:
                break
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
