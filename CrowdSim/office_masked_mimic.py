"""Run MaskedMimic humanoids in a shared Office scene with PNG-map spawns."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OFFICE_MAP_RESOLUTION = 100.0 / 1999.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MaskedMimic humanoids in a global Office USD scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="results/smpl_amass/last.ckpt")
    parser.add_argument("--motion-file", default="/home/pcl/amp/Assets/motion/amass_smpl_test.pt")
    parser.add_argument("--office-usd", default="/home/pcl/amp/Assets/Office/office.usd")
    parser.add_argument("--office-map", default="/home/pcl/amp/Assets/Office/office.png")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--spawn-xy",
        default=None,
        help="Manual x,y positions separated by semicolons. If omitted, sample from office-map.",
    )
    parser.add_argument("--map-resolution", type=float, default=DEFAULT_OFFICE_MAP_RESOLUTION)
    parser.add_argument("--map-free-threshold", type=int, default=200)
    parser.add_argument("--spawn-radius", type=float, default=0.45)
    parser.add_argument("--spawn-spacing", type=float, default=0.9)
    parser.add_argument("--spawn-seed", type=int, default=0)
    parser.add_argument(
        "--office-physics",
        action="store_true",
        help="Load the Office USD through IsaacLab SceneCfg so authored collisions can participate in physics.",
    )
    parser.add_argument("--human-mesh", action="store_true")
    parser.add_argument("--hide-humanoid", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--full-eval", action="store_true")
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


args = parse_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch("isaaclab")

import os  # noqa: E402
import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

from CrowdSim.office_scene import (  # noqa: E402
    add_global_usd_reference,
    apply_fixed_spawn_offsets,
    parse_spawn_xy,
    patch_isaaclab_scene_with_global_usd,
    resolve_repo_path,
    sample_spawn_xy_from_map,
)
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402


def enable_human_mesh(hide_humanoid: bool) -> None:
    os.environ["CROWDSIM_ENABLE_HUMAN_MESH"] = "1"
    os.environ["CROWDSIM_HIDE_HUMANOID"] = "1" if hide_humanoid else "0"
    os.environ.setdefault("CROWDSIM_SMPL_MODEL_DIR", str(resolve_repo_path("data/smpl")))


def validate_paths() -> tuple[Path, Path, Path, Path]:
    checkpoint = resolve_repo_path(args.checkpoint)
    motion_file = resolve_repo_path(args.motion_file)
    office_usd = Path(args.office_usd).expanduser().resolve()
    office_map = Path(args.office_map).expanduser().resolve()

    for label, path in (
        ("Checkpoint", checkpoint),
        ("Motion file", motion_file),
        ("Office USD", office_usd),
        ("Office map", office_map),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    return checkpoint, motion_file, office_usd, office_map


def configure_from_checkpoint(checkpoint: Path, motion_file: Path):
    resolved_configs = torch.load(
        checkpoint.parent / "resolved_configs_inference.pt",
        map_location="cpu",
        weights_only=False,
    )
    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]

    from protomotions.simulator.factory import update_simulator_config_for_test
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    current_simulator = simulator_config._target_.split(".")[-3]
    if current_simulator != "isaaclab":
        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator="isaaclab",
            robot_config=robot_config,
        )

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)
    simulator_config.num_envs = args.num_envs
    simulator_config.headless = args.headless
    motion_lib_config.motion_file = str(motion_file)

    from protomotions.utils.config_utils import (
        apply_config_overrides,
        parse_cli_overrides,
    )

    overrides = parse_cli_overrides(args.overrides) if args.overrides else None
    if overrides:
        apply_config_overrides(
            overrides,
            env_config,
            simulator_config,
            robot_config,
            agent_config,
            terrain_config,
            motion_lib_config,
            scene_lib_config,
        )

    return (
        robot_config,
        simulator_config,
        terrain_config,
        scene_lib_config,
        motion_lib_config,
        env_config,
        agent_config,
    )


def sample_or_parse_spawns(office_map: Path, num_envs: int, device: torch.device) -> torch.Tensor:
    if args.spawn_xy:
        spawn_xy = parse_spawn_xy(args.spawn_xy, num_envs, device)
        print(f"[CrowdSim] Using manual spawn XY: {spawn_xy.cpu().tolist()}")
        return spawn_xy

    spawn_xy = sample_spawn_xy_from_map(
        map_path=office_map,
        num_envs=num_envs,
        device=device,
        map_resolution=args.map_resolution,
        humanoid_radius=args.spawn_radius,
        min_spacing=args.spawn_spacing,
        free_threshold=args.map_free_threshold,
        seed=args.spawn_seed,
    )
    print(f"[CrowdSim] Sampled spawn XY: {spawn_xy.cpu().tolist()}")
    return spawn_xy


def main() -> None:
    checkpoint, motion_file, office_usd, office_map = validate_paths()

    if args.human_mesh:
        enable_human_mesh(args.hide_humanoid)
    if args.office_physics:
        patch_isaaclab_scene_with_global_usd(office_usd)

    (
        robot_config,
        simulator_config,
        terrain_config,
        scene_lib_config,
        motion_lib_config,
        env_config,
        agent_config,
    ) = configure_from_checkpoint(checkpoint, motion_file)

    fabric = Fabric(
        **asdict(
            FabricConfig(accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[])
        )
    )
    fabric.launch()
    app_launcher = AppLauncher({"headless": args.headless, "device": str(fabric.device)})

    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    from protomotions.utils.component_builder import build_all_components

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=getattr(env_config, "save_dir", None),
        simulation_app=app_launcher.app,
    )

    env_cls = get_class(env_config._target_)
    env = env_cls(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    if not args.office_physics:
        add_global_usd_reference(office_usd)

    spawn_xy = sample_or_parse_spawns(office_map, env.num_envs, fabric.device)
    apply_fixed_spawn_offsets(env, spawn_xy)
    print("[CrowdSim] Office scene ready; starting MaskedMimic policy.")

    agent_cls = get_class(agent_config._target_)
    agent = agent_cls(config=agent_config, env=env, fabric=fabric, root_dir=checkpoint.parent)
    agent.setup()
    agent.load(str(checkpoint), load_env=False)

    if args.full_eval:
        agent.evaluator.eval_count = 0
        print(agent.evaluator.evaluate())
    else:
        agent.evaluator.simple_test_policy(collect_metrics=True)


if __name__ == "__main__":
    main()
