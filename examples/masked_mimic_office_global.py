# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
"""Run MaskedMimic inference with one global Office USD scene.

The Office asset is placed at /World/Office, outside IsaacLab's cloned
/World/envs/env_* namespace, so all SMPL environments share the same global
scene. By default the Office is added as a visual USD reference after simulator
initialization. Pass --office-physics to include it during IsaacLab scene
construction so authored collisions may participate in PhysX.

Example:
    python examples/masked_mimic_office_global.py \
        --checkpoint data/pretrained_models/masked_mimic/smpl/last.ckpt \
        --motion-file ../Amass/motion/amass_smpl_train.pt \
        --simulator isaaclab \
        --num-envs 4
"""

import argparse
from dataclasses import asdict
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MaskedMimic inference with a global Office USD scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default="data/pretrained_models/masked_mimic/smpl/last.ckpt",
        help="MaskedMimic checkpoint.",
    )
    parser.add_argument("--simulator", default="isaaclab", choices=["isaaclab"])
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--motion-file",
        default="../Amass/motion/amass_smpl_train.pt",
        help="MotionLib .pt/.motion file used by MaskedMimic targets.",
    )
    parser.add_argument(
        "--office-usd",
        default="/home/pcl/amp/assets/Office/office.usd",
        help="Local Office root USD file.",
    )
    parser.add_argument("--office-z", type=float, default=0.0)
    parser.add_argument(
        "--office-physics",
        action="store_true",
        help="Load Office through IsaacLab SceneCfg so authored collisions can be active. This can be slow for full scenes.",
    )
    parser.add_argument(
        "--spawn-xy",
        default="-1.5,-1.5;1.5,-1.5;-1.5,1.5;1.5,1.5",
        help="Semicolon-separated XY spawn offsets for the humanoid envs.",
    )
    parser.add_argument("--full-eval", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides, e.g. env.max_episode_length=5000",
    )
    return parser


parser = create_parser()
args, unknown_args = parser.parse_known_args()

# IsaacLab/IsaacSim must be imported before torch.
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402


def resolve_from_repo(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parents[1] / path).resolve()


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


def patch_isaaclab_scene_with_physics_office(office_usd: Path, office_z: float) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    import protomotions.simulator.isaaclab.simulator as simulator_module
    from protomotions.simulator.isaaclab.utils.scene import SceneCfg as BaseSceneCfg

    class OfficeSceneCfg(BaseSceneCfg):
        def __init__(self, *scene_args, **scene_kwargs):
            super().__init__(*scene_args, **scene_kwargs)
            self.office = AssetBaseCfg(
                prim_path="/World/Office",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(office_usd),
                    activate_contact_sensors=False,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, office_z)),
                collision_group=-1,
            )

    simulator_module.SceneCfg = OfficeSceneCfg


def add_visual_office_reference(office_usd: Path, office_z: float) -> None:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    office_prim = stage.DefinePrim("/World/Office", "Xform")
    office_prim.GetReferences().AddReference(str(office_usd))
    UsdGeom.XformCommonAPI(office_prim).SetTranslate((0.0, 0.0, office_z))


def apply_spawn_offsets(env, spawn_xy: torch.Tensor) -> None:
    import types

    fixed_offsets = torch.zeros(env.num_envs, 3, dtype=torch.float32, device=env.device)
    fixed_offsets[:, :2] = spawn_xy

    def fixed_update_respawn_root_offset_by_env_ids(
        self, env_ids, ref_state=None, sample_flat=False
    ):
        self.respawn_root_offset[env_ids] = fixed_offsets[env_ids]

    env.update_respawn_root_offset_by_env_ids = types.MethodType(
        fixed_update_respawn_root_offset_by_env_ids, env
    )
    env.respawn_root_offset[:] = fixed_offsets


def main() -> None:
    args = parser.parse_args()

    checkpoint = resolve_from_repo(args.checkpoint)
    motion_file = resolve_from_repo(args.motion_file)
    office_usd = Path(args.office_usd).expanduser().resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not motion_file.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")
    if not office_usd.exists():
        raise FileNotFoundError(f"Office USD not found: {office_usd}")

    resolved_configs_path = checkpoint.parent / "resolved_configs_inference.pt"
    if not resolved_configs_path.exists():
        raise FileNotFoundError(
            f"Could not find resolved configs: {resolved_configs_path}"
        )

    resolved_configs = torch.load(
        resolved_configs_path, map_location="cpu", weights_only=False
    )
    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]

    current_simulator = simulator_config._target_.split(".")[-3]
    if args.simulator != current_simulator:
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )

    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    simulator_config.num_envs = args.num_envs
    simulator_config.headless = args.headless
    motion_lib_config.motion_file = str(motion_file)

    from protomotions.utils.config_utils import (
        apply_config_overrides,
        parse_cli_overrides,
    )

    cli_overrides = parse_cli_overrides(args.overrides) if args.overrides else None
    if cli_overrides:
        apply_config_overrides(
            cli_overrides,
            env_config,
            simulator_config,
            robot_config,
            agent_config,
            terrain_config,
            motion_lib_config,
            scene_lib_config,
        )

    fabric_config = FabricConfig(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        loggers=[],
        callbacks=[],
    )
    fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    app_launcher = AppLauncher({"headless": args.headless, "device": str(fabric.device)})
    simulator_extra_params = {"simulation_app": app_launcher.app}

    if args.office_physics:
        patch_isaaclab_scene_with_physics_office(office_usd, args.office_z)

    from protomotions.simulator.base_simulator.utils import (
        convert_friction_for_simulator,
    )

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    from protomotions.utils.component_builder import build_all_components

    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=getattr(env_config, "save_dir", None),
        **simulator_extra_params,
    )

    EnvClass = get_class(env_config._target_)
    env = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    if not args.office_physics:
        add_visual_office_reference(office_usd, args.office_z)

    spawn_xy = parse_spawn_xy(args.spawn_xy, env.num_envs, fabric.device)
    apply_spawn_offsets(env, spawn_xy)

    AgentClass = get_class(agent_config._target_)
    agent = AgentClass(
        config=agent_config,
        env=env,
        fabric=fabric,
        root_dir=checkpoint.parent,
    )

    print("\n=== MaskedMimic Office Global Scene ===")
    print(f"Checkpoint: {checkpoint}")
    print(f"Motion file: {motion_file}")
    print(f"Office USD: {office_usd}")
    print(f"Office mode: {'physics' if args.office_physics else 'visual'}")
    print(f"Spawn XY: {spawn_xy.cpu().tolist()}\n")

    agent.setup()
    agent.load(str(checkpoint), load_env=False)

    try:
        if args.full_eval:
            agent.evaluator.eval_count = 0
            evaluation_log, evaluated_score = agent.evaluator.evaluate()
            print("\nEVALUATION RESULTS")
            for key, value in sorted(evaluation_log.items()):
                print(f"  {key}: {value:.6f}")
            if evaluated_score is not None:
                print(f"  Overall Score: {evaluated_score:.6f}")
        else:
            agent.evaluator.simple_test_policy(collect_metrics=True)
    finally:
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()


if __name__ == "__main__":
    main()
