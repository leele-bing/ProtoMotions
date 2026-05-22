"""Run MaskedMimic humanoids in a shared Office scene with PNG-map spawns."""

from __future__ import annotations

import argparse
from datetime import datetime
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OFFICE_MAP_RESOLUTION = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MaskedMimic humanoids in a global Office USD scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="data/pretrained_models/masked_mimic/smpl/last.ckpt")
    parser.add_argument("--motion-file", default="../Assets/motion/amass_smpl_test.pt")
    parser.add_argument("--office-usd", default="../Assets/Office/office.usd")
    parser.add_argument("--office-map", default="../Assets/Office/office.png")
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
        "--enable-navigation",
        action="store_true",
        help="Sample starts/goals, plan paths, drive robots, and monitor agent collisions.",
    )
    parser.add_argument("--nav-local-controller", choices=("orca", "sfm"), default="orca")
    parser.add_argument("--nav-seed", type=int, default=7)
    parser.add_argument("--nav-agent-radius", type=float, default=0.35)
    parser.add_argument("--nav-safe-distance", type=float, default=0.25)
    parser.add_argument("--nav-max-speed", type=float, default=0.8)
    parser.add_argument("--nav-waypoint-tolerance", type=float, default=0.45)
    parser.add_argument("--nav-goal-tolerance", type=float, default=0.75)
    parser.add_argument("--nav-min-start-goal-distance", type=float, default=5.0)
    parser.add_argument("--nav-min-spawn-spacing", type=float, default=1.2)
    parser.add_argument("--nav-neighbor-radius", type=float, default=4.0)
    parser.add_argument("--nav-collision-distance", type=float, default=0.75)
    parser.add_argument("--nav-log-interval", type=int, default=120)
    parser.add_argument("--jetbot-wheel-radius", type=float, default=0.0325)
    parser.add_argument("--jetbot-wheel-base", type=float, default=0.118)
    parser.add_argument("--jetbot-max-wheel-speed", type=float, default=12.0)
    parser.add_argument("--jetbot-heading-gain", type=float, default=2.5)
    parser.add_argument(
        "--office-physics",
        action="store_true",
        help="Load the Office USD through IsaacLab SceneCfg so authored collisions can participate in physics.",
    )
    parser.add_argument(
        "--robot-usd",
        default=None,
        help="Optional navigation robot USD. Use a local path, an Omniverse URI, or alias 'jetbot'.",
    )
    parser.add_argument("--robot-prim-name", default="CrowdRobot")
    parser.add_argument("--robot-articulation-root-prim-path", default=None)
    parser.add_argument(
        "--robot-mount-prim-path",
        default="",
        help="Prim under the robot where sensors attach, e.g. chassis or base_link.",
    )
    parser.add_argument(
        "--robot-spawn-xy",
        default=None,
        help="Manual robot x,y[,yaw-rad] poses separated by semicolons. If omitted, sample from office-map.",
    )
    parser.add_argument("--robot-z", type=float, default=0.0)
    parser.add_argument("--robot-radius", type=float, default=0.35)
    parser.add_argument("--robot-spacing", type=float, default=1.2)
    parser.add_argument("--robot-spawn-seed", type=int, default=1)
    parser.add_argument("--enable-robot-camera", action="store_true")
    parser.add_argument("--robot-camera-height", type=int, default=480)
    parser.add_argument("--robot-camera-width", type=int, default=640)
    parser.add_argument("--robot-camera-record-fps", type=float, default=10.0)
    parser.add_argument("--robot-camera-record-dir", default="output/crowdsim_camera")
    parser.add_argument(
        "--robot-camera-record-envs",
        default="0",
        help="Camera env ids to record, e.g. '0', '0,2', or 'all'.",
    )
    parser.add_argument(
        "--auto-record-robot-camera",
        action="store_true",
        help="Start robot camera recording immediately; otherwise press Y in the viewer.",
    )
    parser.add_argument("--enable-robot-lidar", action="store_true")
    parser.add_argument("--robot-lidar-debug-vis", action="store_true")
    parser.add_argument("--robot-lidar-horizontal-res", type=float, default=1.0)
    parser.add_argument(
        "--robot-lidar-mesh-prim",
        nargs="*",
        default=["/World/Office"],
        help="Mesh prim paths hit by the robot lidar raycaster.",
    )
    parser.add_argument(
        "--viewer-camera-mode",
        choices=("free", "humanoid"),
        default="free",
        help="Viewport camera mode. 'free' leaves the camera under user control; 'humanoid' follows ProtoMotions default.",
    )
    parser.add_argument(
        "--viewer-camera-eye",
        nargs=3,
        type=float,
        default=(8.0, -8.0, 6.0),
        metavar=("X", "Y", "Z"),
        help="Initial viewport camera eye for --viewer-camera-mode free.",
    )
    parser.add_argument(
        "--viewer-camera-target",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="Initial viewport camera target for --viewer-camera-mode free.",
    )
    parser.add_argument(
        "--hide-office-ceiling",
        action="store_true",
        help="Hide Office prims whose paths match --hide-office-keywords.",
    )
    parser.add_argument(
        "--hide-office-keywords",
        nargs="*",
        default=["ceiling", "roof", "top"],
        help="Case-insensitive prim path keywords hidden when --hide-office-ceiling is set.",
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

from CrowdSim.sim_world import (  # noqa: E402
    CrowdRobotSceneConfig,
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
from CrowdSim.navigation import CrowdNavigationConfig, CrowdNavigationManager  # noqa: E402
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


def sample_or_parse_robot_spawns(
    office_map: Path, num_envs: int, device: torch.device
) -> torch.Tensor:
    if args.robot_spawn_xy:
        spawn_xy_yaw = parse_spawn_xy_yaw(args.robot_spawn_xy, num_envs, device)
        print(f"[CrowdSim] Using manual robot spawn XY/yaw: {spawn_xy_yaw.cpu().tolist()}")
        return spawn_xy_yaw

    spawn_xy = sample_spawn_xy_from_map(
        map_path=office_map,
        num_envs=num_envs,
        device=device,
        map_resolution=args.map_resolution,
        humanoid_radius=args.robot_radius,
        min_spacing=args.robot_spacing,
        free_threshold=args.map_free_threshold,
        seed=args.robot_spawn_seed,
    )
    spawn_xy_yaw = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
    spawn_xy_yaw[:, :2] = spawn_xy
    print(f"[CrowdSim] Sampled robot spawn XY/yaw: {spawn_xy_yaw.cpu().tolist()}")
    return spawn_xy_yaw


def make_crowd_robot_config(robot_usd: str | None) -> CrowdRobotSceneConfig | None:
    if robot_usd is None:
        return None

    return CrowdRobotSceneConfig(
        usd_path=robot_usd,
        prim_name=args.robot_prim_name,
        articulation_root_prim_path=args.robot_articulation_root_prim_path,
        mount_prim_path=args.robot_mount_prim_path,
        init_z=args.robot_z,
        enable_camera=args.enable_robot_camera,
        camera_height=args.robot_camera_height,
        camera_width=args.robot_camera_width,
        enable_lidar=args.enable_robot_lidar,
        lidar_horizontal_res=args.robot_lidar_horizontal_res,
        lidar_mesh_prim_paths=tuple(args.robot_lidar_mesh_prim),
        debug_vis=args.robot_lidar_debug_vis and not args.headless,
    )


def suppress_known_isaaclab_warning_spam() -> None:
    """Disable noisy Isaac/PhysX channels after SimulationApp is available."""
    import omni.log

    omni_log = omni.log.get_log()
    for channel in ("omni.physx.plugin", "isaaclab.sim.utils"):
        omni_log.set_channel_enabled(
            channel,
            False,
            omni.log.SettingBehavior.OVERRIDE,
        )


def configure_viewer_camera(env) -> None:
    if args.headless or args.viewer_camera_mode != "free":
        return

    import types
    import numpy as np

    eye = np.asarray(args.viewer_camera_eye, dtype=np.float64)
    target = np.asarray(args.viewer_camera_target, dtype=np.float64)

    def init_free_camera(self) -> None:
        self._cam_prev_char_pos = target.copy()
        self._perspective_view.set_camera_view(eye, target)

    def keep_user_camera(self) -> None:
        return

    env.simulator._init_camera = types.MethodType(init_free_camera, env.simulator)
    env.simulator._update_camera = types.MethodType(keep_user_camera, env.simulator)


class RobotCameraRecorder:
    def __init__(
        self,
        env,
        output_dir: Path,
        fps: float,
        env_ids: list[int],
    ) -> None:
        if fps <= 0:
            raise ValueError(f"robot-camera-record-fps must be positive, got {fps}")

        self.env = env
        self.output_dir = output_dir
        self.fps = fps
        self.env_ids = env_ids
        self.enabled = False
        self.frame_idx = 0
        self.sim_time = 0.0
        self.next_capture_time = 0.0
        self.session_dir: Path | None = None

    def toggle(self) -> None:
        if self.enabled:
            self.enabled = False
            print(f"[CrowdSim] Robot camera recording stopped: {self.session_dir}")
            return

        self.session_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = True
        self.frame_idx = 0
        self.sim_time = 0.0
        self.next_capture_time = 0.0
        self._write_metadata()
        print(
            f"[CrowdSim] Robot camera recording started at {self.fps:g} fps: "
            f"{self.session_dir}"
        )

    def step(self) -> None:
        self.sim_time += float(getattr(self.env, "dt", 0.0))
        if not self.enabled:
            return

        if self.sim_time + 1e-9 < self.next_capture_time:
            return

        self._save_frame()
        self.frame_idx += 1
        self.next_capture_time += 1.0 / self.fps

    def _write_metadata(self) -> None:
        camera = self.env.crowdsim_robot_camera
        metadata_path = self.session_dir / "metadata.pt"
        torch.save(
            {
                "fps": self.fps,
                "env_ids": self.env_ids,
                "image_shape": camera.data.image_shape,
                "intrinsic_matrices": None
                if camera.data.intrinsic_matrices is None
                else camera.data.intrinsic_matrices.detach().cpu(),
            },
            metadata_path,
        )

    def _save_frame(self) -> None:
        from PIL import Image

        camera = self.env.crowdsim_robot_camera
        output = camera.data.output
        if "rgb" not in output or "distance_to_image_plane" not in output:
            print("[CrowdSim] Camera output is not ready yet; skipping frame.")
            return

        rgb = output["rgb"].detach().cpu()
        depth = output["distance_to_image_plane"].detach().cpu()
        frame_dir = self.session_dir / f"frame_{self.frame_idx:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        for env_id in self.env_ids:
            rgb_np = rgb[env_id].numpy()
            if rgb_np.dtype != "uint8":
                rgb_np = rgb_np.clip(0, 255).astype("uint8")
            Image.fromarray(rgb_np[..., :3]).save(frame_dir / f"env_{env_id:04d}_rgb.png")
            torch.save(depth[env_id], frame_dir / f"env_{env_id:04d}_depth.pt")


def parse_record_env_ids(num_envs: int) -> list[int]:
    value = args.robot_camera_record_envs.strip().lower()
    if value == "all":
        return list(range(num_envs))

    env_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not env_ids:
        raise ValueError("--robot-camera-record-envs must contain at least one env id")
    for env_id in env_ids:
        if env_id < 0 or env_id >= num_envs:
            raise ValueError(f"Camera record env id {env_id} outside [0, {num_envs})")
    return env_ids


def configure_robot_camera_recorder(env) -> RobotCameraRecorder | None:
    if not hasattr(env, "crowdsim_robot_camera"):
        return None

    recorder = RobotCameraRecorder(
        env=env,
        output_dir=Path(args.robot_camera_record_dir).expanduser().resolve(),
        fps=args.robot_camera_record_fps,
        env_ids=parse_record_env_ids(env.num_envs),
    )
    env.crowdsim_robot_camera_recorder = recorder

    import types

    original_step = env.step

    def step_with_camera_recording(self, action):
        result = original_step(action)
        recorder.step()
        return result

    env.step = types.MethodType(step_with_camera_recording, env)

    keyboard = getattr(env.simulator, "keyboard_interface", None)
    if keyboard is not None:
        try:
            keyboard.add_callback("Y", recorder.toggle)
            print("[CrowdSim] Press Y to start/stop robot camera recording.")
        except Exception as exc:
            print(f"[CrowdSim] Warning: failed to register Y recording key: {exc}")

    if args.auto_record_robot_camera:
        recorder.toggle()

    return recorder


def hide_office_ceiling_if_requested() -> None:
    if not args.hide_office_ceiling:
        return

    hidden = hide_prims_matching_keywords(
        root_prim_path="/World/Office",
        keywords=tuple(args.hide_office_keywords),
    )
    print(
        f"[CrowdSim] Hidden {len(hidden)} Office prim(s) matching "
        f"{list(args.hide_office_keywords)}."
    )


def build_navigation_manager(
    office_map: Path,
    num_humanoids: int,
    num_robots: int,
    device: torch.device,
) -> CrowdNavigationManager | None:
    if not args.enable_navigation:
        return None

    config = CrowdNavigationConfig(
        map_path=office_map,
        map_resolution=args.map_resolution,
        free_threshold=args.map_free_threshold,
        num_humanoids=num_humanoids,
        num_robots=num_robots,
        device=device,
        seed=args.nav_seed,
        local_controller=args.nav_local_controller,
        agent_radius=args.nav_agent_radius,
        humanoid_radius=args.spawn_radius,
        safe_distance=args.nav_safe_distance,
        max_speed=args.nav_max_speed,
        waypoint_tolerance=args.nav_waypoint_tolerance,
        goal_tolerance=args.nav_goal_tolerance,
        min_start_goal_distance=args.nav_min_start_goal_distance,
        min_spawn_spacing=args.nav_min_spawn_spacing,
        neighbor_radius=args.nav_neighbor_radius,
        jetbot_wheel_radius=args.jetbot_wheel_radius,
        jetbot_wheel_base=args.jetbot_wheel_base,
        jetbot_max_wheel_speed=args.jetbot_max_wheel_speed,
        heading_gain=args.jetbot_heading_gain,
        collision_distance=args.nav_collision_distance,
        log_interval=args.nav_log_interval,
    )
    return CrowdNavigationManager(config)


def main() -> None:
    checkpoint, motion_file, office_usd, office_map = validate_paths()

    if args.human_mesh:
        enable_human_mesh(args.hide_humanoid)

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
    launcher_args = {"headless": args.headless, "device": str(fabric.device)}
    if args.enable_robot_camera:
        launcher_args["enable_cameras"] = True
    app_launcher = AppLauncher(launcher_args)
    suppress_known_isaaclab_warning_spam()

    robot_usd = resolve_robot_usd(args.robot_usd)
    crowd_robot_config = make_crowd_robot_config(robot_usd)
    office_loaded_in_scene_cfg = args.office_physics or crowd_robot_config is not None
    if office_loaded_in_scene_cfg:
        patch_isaaclab_scene_with_crowdsim_assets(
            office_usd_path=office_usd,
            crowd_robot=crowd_robot_config,
        )

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
    configure_viewer_camera(env)

    if not office_loaded_in_scene_cfg:
        add_global_usd_reference(office_usd)
    hide_office_ceiling_if_requested()

    nav_manager = build_navigation_manager(
        office_map=office_map,
        num_humanoids=env.num_envs,
        num_robots=env.num_envs if crowd_robot_config is not None else 0,
        device=fabric.device,
    )

    if nav_manager is not None:
        spawn_xy = nav_manager.humanoid_starts_xy
        print(f"[CrowdSim] Navigation humanoid starts: {spawn_xy.cpu().tolist()}")
    else:
        spawn_xy = sample_or_parse_spawns(office_map, env.num_envs, fabric.device)
    apply_fixed_spawn_offsets(env, spawn_xy)
    if crowd_robot_config is not None:
        if nav_manager is not None:
            robot_spawn_xy_yaw = nav_manager.robot_starts_xy_yaw
            print(
                f"[CrowdSim] Navigation robot starts: "
                f"{robot_spawn_xy_yaw.cpu().tolist()}"
            )
        else:
            robot_spawn_xy_yaw = sample_or_parse_robot_spawns(
                office_map, env.num_envs, fabric.device
            )
        apply_fixed_crowd_robot_spawns(env, robot_spawn_xy_yaw)
        enabled_sensors = [
            name
            for name, enabled in (
                ("camera", args.enable_robot_camera),
                ("lidar", args.enable_robot_lidar),
            )
            if enabled
        ]
        sensor_text = ", ".join(enabled_sensors) if enabled_sensors else "no sensors"
        print(f"[CrowdSim] Navigation robots ready: {env.num_envs} ({sensor_text}).")
        configure_robot_camera_recorder(env)
    if nav_manager is not None:
        nav_manager.attach(env)
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
