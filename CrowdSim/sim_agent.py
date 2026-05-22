"""ProtoMotions agent and environment construction helpers for CrowdSim."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from CrowdSim.sim_world import CrowdRobotSceneConfig, resolve_repo_path


@dataclass
class ProtoMotionsRuntime:
    fabric: Any
    env: Any
    agent: Any
    robot_config: Any
    simulator_config: Any
    terrain_config: Any
    scene_lib_config: Any
    motion_lib_config: Any
    env_config: Any
    agent_config: Any


def enable_human_mesh(model_dir: str | None = None, hide_humanoid: bool = False) -> None:
    os.environ["CROWDSIM_ENABLE_HUMAN_MESH"] = "1"
    os.environ["CROWDSIM_HIDE_HUMANOID"] = "1" if hide_humanoid else "0"
    os.environ.setdefault(
        "CROWDSIM_SMPL_MODEL_DIR",
        str(resolve_repo_path(model_dir or "data/smpl")),
    )


def resolve_robot_usd(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() == "jetbot":
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        return f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"
    if "://" in value:
        return value

    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else resolve_repo_path(value)
    if not resolved.exists():
        raise FileNotFoundError(f"Robot USD not found: {resolved}")
    return str(resolved)


def make_crowd_robot_config(robot_cfg: dict, sensor_cfg: dict, robot_usd: str | None):
    if robot_usd is None:
        return None

    camera_cfg = sensor_cfg.get("camera", {})
    lidar_cfg = sensor_cfg.get("lidar", {})
    return CrowdRobotSceneConfig(
        usd_path=robot_usd,
        prim_name=robot_cfg.get("prim_name", "CrowdRobot"),
        articulation_root_prim_path=robot_cfg.get("articulation_root_prim_path"),
        mount_prim_path=robot_cfg.get("mount_prim_path", ""),
        init_z=float(robot_cfg.get("z", 0.0)),
        enable_camera=bool(camera_cfg.get("enabled", False)),
        camera_height=int(camera_cfg.get("height", 480)),
        camera_width=int(camera_cfg.get("width", 640)),
        enable_lidar=bool(lidar_cfg.get("enabled", False)),
        lidar_horizontal_res=float(lidar_cfg.get("horizontal_res", 1.0)),
        lidar_mesh_prim_paths=tuple(lidar_cfg.get("mesh_prims", ["/World/Scene"])),
        debug_vis=bool(lidar_cfg.get("debug_vis", False)),
    )


def configure_from_checkpoint(
    checkpoint: Path,
    motion_file: Path,
    num_envs: int,
    headless: bool,
):
    import torch

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
    simulator_config.num_envs = int(num_envs)
    simulator_config.headless = bool(headless)
    if getattr(simulator_config, "projectile", None) is None:
        from protomotions.simulator.base_simulator.config import ProjectileConfig

        simulator_config.projectile = ProjectileConfig(num_projectiles=0)
    else:
        simulator_config.projectile.num_projectiles = 0
    print("[CrowdSim] ProtoMotions projectiles disabled (num_projectiles=0).")
    motion_lib_config.motion_file = str(motion_file)

    return (
        robot_config,
        simulator_config,
        terrain_config,
        scene_lib_config,
        motion_lib_config,
        env_config,
        agent_config,
    )


def create_fabric():
    from lightning.fabric import Fabric
    from protomotions.utils.fabric_config import FabricConfig

    fabric = Fabric(
        **asdict(
            FabricConfig(accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[])
        )
    )
    fabric.launch()
    return fabric


def build_runtime(
    checkpoint: Path,
    motion_file: Path,
    num_envs: int,
    headless: bool,
    simulation_app,
    fabric,
) -> ProtoMotionsRuntime:
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    from protomotions.utils.component_builder import build_all_components
    from protomotions.utils.hydra_replacement import get_class

    (
        robot_config,
        simulator_config,
        terrain_config,
        scene_lib_config,
        motion_lib_config,
        env_config,
        agent_config,
    ) = configure_from_checkpoint(checkpoint, motion_file, num_envs, headless)

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
        simulation_app=simulation_app,
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

    agent_cls = get_class(agent_config._target_)
    agent = agent_cls(config=agent_config, env=env, fabric=fabric, root_dir=checkpoint.parent)
    agent.setup()
    agent.load(str(checkpoint), load_env=False)

    return ProtoMotionsRuntime(
        fabric=fabric,
        env=env,
        agent=agent,
        robot_config=robot_config,
        simulator_config=simulator_config,
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        env_config=env_config,
        agent_config=agent_config,
    )


def configure_viewer_camera(env, viewer_cfg: dict, headless: bool) -> None:
    if headless or viewer_cfg.get("camera_mode", "free") != "free":
        return

    import types
    import numpy as np

    eye = np.asarray(viewer_cfg.get("camera_eye", [8.0, -8.0, 6.0]), dtype=np.float64)
    target = np.asarray(viewer_cfg.get("camera_target", [0.0, 0.0, 1.0]), dtype=np.float64)

    def init_free_camera(self) -> None:
        self._cam_prev_char_pos = target.copy()
        self._perspective_view.set_camera_view(eye, target)

    def keep_user_camera(self) -> None:
        return

    env.simulator._init_camera = types.MethodType(init_free_camera, env.simulator)
    env.simulator._update_camera = types.MethodType(keep_user_camera, env.simulator)


def suppress_known_isaaclab_warning_spam() -> None:
    import omni.log

    omni_log = omni.log.get_log()
    for channel in ("omni.physx.plugin", "isaaclab.sim.utils"):
        omni_log.set_channel_enabled(
            channel,
            False,
            omni.log.SettingBehavior.OVERRIDE,
        )
