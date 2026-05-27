"""Create a filtered USD scene by deactivating/removing keyword-matched prims."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch("isaaclab")


@dataclass(frozen=True)
class NewGroundConfig:
    prim: str = "CrowdSimGround"
    z: float = 0.0
    size: tuple[float, float] = (400.0, 400.0)
    color: tuple[float, float, float] | None = (0.2, 0.2, 0.2)
    static_friction: float = 1.5
    dynamic_friction: float = 1.5
    restitution: float = 0.0
    friction_combine_mode: str = "max"
    restitution_combine_mode: str = "average"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a USD scene using keyword-matched prim paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-usd", default="/home/pcl/amp/Assets/Warehouse/warehouse_removed.usd")
    parser.add_argument("--output-usd", default="/home/pcl/amp/Assets/Warehouse/warehouse_floor.usd")
    parser.add_argument(
        "--root-prim",
        default=None,
        help="Optional prim path to search under. Defaults to auto-detecting the scene root.",
    )
    parser.add_argument("--keywords", nargs="+", required=True)
    parser.add_argument(
        "--mode",
        choices=("remove", "deactivate"),
        default="deactivate",
        help="remove deletes matched prims; deactivate authors active=false opinions.",
    )
    parser.add_argument(
        "--new-ground",
        action="store_true",
        help="Add a generated flat ground plane mesh to the output USD.",
    )
    parser.add_argument(
        "--ground-prim",
        default="CrowdSimGround",
        help="Ground prim path, or a prim name to create under the detected scene root.",
    )
    parser.add_argument(
        "--ground-z",
        type=float,
        default=0.0,
        help="Ground plane z position.",
    )
    parser.add_argument(
        "--ground-size",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=(400.0, 400.0),
        help="Ground plane size. The default covers x/y -200..200 around the origin.",
    )
    parser.add_argument("--ground-static-friction", type=float, default=1.5)
    parser.add_argument("--ground-dynamic-friction", type=float, default=1.5)
    parser.add_argument("--ground-restitution", type=float, default=0.0)
    parser.add_argument(
        "--ground-friction-combine-mode",
        choices=("average", "min", "multiply", "max"),
        default="max",
    )
    parser.add_argument(
        "--ground-restitution-combine-mode",
        choices=("average", "min", "multiply", "max"),
        default="average",
    )
    parser.add_argument(
        "--ground-color",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=(0.2, 0.2, 0.2),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_usd = resolve_path(args.input_usd)
    output_usd = resolve_path(args.output_usd) if args.output_usd else default_output_path(input_usd)
    root_prim = str(args.root_prim) if args.root_prim else None
    keywords = tuple(args.keywords)
    ground_config = make_ground_config(args) if args.new_ground else None

    app_launcher = AppLauncher({"headless": True})
    try:
        matched = filter_usd_scene(
            input_usd=input_usd,
            output_usd=output_usd,
            root_prim_path=root_prim,
            keywords=keywords,
            mode=args.mode,
            dry_run=args.dry_run,
            new_ground=ground_config,
        )
        action = "Would filter" if args.dry_run else "Filtered"
        print(f"[CrowdSim] {action} {len(matched)} prim(s) from {input_usd}", flush=True)
        for path in matched[:80]:
            print(f"  {path}", flush=True)
        if len(matched) > 80:
            print(f"  ... {len(matched) - 80} more", flush=True)
        if args.new_ground:
            ground_action = "Would add" if args.dry_run else "Added"
            print(
                f"[CrowdSim] {ground_action} ground: "
                f"size={tuple(args.ground_size)}, z={args.ground_z}",
                flush=True,
            )
        if not args.dry_run:
            print(f"[CrowdSim] Saved filtered USD: {output_usd}", flush=True)
    finally:
        app_launcher.app.close()


def filter_usd_scene(
    input_usd: Path,
    output_usd: Path,
    root_prim_path: str | None,
    keywords: tuple[str, ...],
    mode: str,
    dry_run: bool,
    new_ground: NewGroundConfig | None = None,
) -> list[str]:
    import omni.usd
    from isaaclab.sim.utils import stage as stage_utils

    if not input_usd.exists():
        raise FileNotFoundError(f"Input USD not found: {input_usd}")

    opened = stage_utils.open_stage(str(input_usd))
    if not opened:
        raise RuntimeError(f"Failed to open USD stage: {input_usd}")
    stage_utils.update_stage()

    stage = omni.usd.get_context().get_stage()
    if root_prim_path is None:
        root_prim_path = find_first_scene_root(stage)
        root_prim = stage.GetPrimAtPath(root_prim_path)
        print(f"[CrowdSim] Auto-detected root prim: {root_prim_path}")
    else:
        root_prim = stage.GetPrimAtPath(root_prim_path)
        if not root_prim.IsValid():
            root_prim_path = find_first_scene_root(stage)
            root_prim = stage.GetPrimAtPath(root_prim_path)
            print(f"[CrowdSim] Root prim not found; using {root_prim_path}")

    matched = find_keyword_matched_prims(stage, root_prim, keywords)
    if dry_run:
        return matched

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    if output_usd.exists():
        output_usd.unlink()

    for prim_path in reversed(matched):
        if mode == "remove":
            stage.RemovePrim(prim_path)
        else:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                prim.SetActive(False)

    if new_ground is not None:
        add_default_ground(stage, root_prim_path, new_ground)

    stage_utils.update_stage()
    saved = stage_utils.save_stage(str(output_usd), save_and_reload_in_place=False)
    if not saved:
        raise RuntimeError(f"Failed to save filtered USD: {output_usd}")
    return matched


def make_ground_config(args: argparse.Namespace) -> NewGroundConfig:
    return NewGroundConfig(
        prim=str(args.ground_prim),
        z=float(args.ground_z),
        size=(float(args.ground_size[0]), float(args.ground_size[1])),
        color=tuple(float(value) for value in args.ground_color),
        static_friction=float(args.ground_static_friction),
        dynamic_friction=float(args.ground_dynamic_friction),
        restitution=float(args.ground_restitution),
        friction_combine_mode=str(args.ground_friction_combine_mode),
        restitution_combine_mode=str(args.ground_restitution_combine_mode),
    )


def add_default_ground(stage, root_prim_path: str, ground: NewGroundConfig) -> str:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material
    from isaaclab.sim.utils import stage as stage_utils
    from pxr import Gf, UsdGeom, UsdPhysics

    ground_path = resolve_ground_prim_path(root_prim_path, ground.prim)
    if stage.GetPrimAtPath(ground_path).IsValid():
        stage.RemovePrim(ground_path)
        stage_utils.update_stage()

    ground_xform = UsdGeom.Xform.Define(stage, ground_path)
    UsdGeom.XformCommonAPI(ground_xform).SetTranslate((0.0, 0.0, ground.z))

    half_x = 0.5 * ground.size[0]
    half_y = 0.5 * ground.size[1]
    mesh_path = f"{ground_path}/Plane"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-half_x, -half_y, 0.0),
            Gf.Vec3f(half_x, -half_y, 0.0),
            Gf.Vec3f(half_x, half_y, 0.0),
            Gf.Vec3f(-half_x, half_y, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateExtentAttr(
        [Gf.Vec3f(-half_x, -half_y, 0.0), Gf.Vec3f(half_x, half_y, 0.0)]
    )
    mesh.CreateSubdivisionSchemeAttr("none")
    if ground.color is not None:
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*ground.color)])

    mesh_prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    UsdPhysics.MeshCollisionAPI.Apply(mesh_prim).GetApproximationAttr().Set("none")

    material_path = f"{ground_path}/physicsMaterial"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=ground.static_friction,
        dynamic_friction=ground.dynamic_friction,
        restitution=ground.restitution,
        friction_combine_mode=ground.friction_combine_mode,
        restitution_combine_mode=ground.restitution_combine_mode,
    )
    material_cfg.func(material_path, material_cfg)
    bind_physics_material(mesh_path, material_path, stage=stage)
    return ground_path


def resolve_ground_prim_path(root_prim_path: str, ground_prim: str) -> str:
    if ground_prim.startswith("/"):
        return ground_prim
    return f"{root_prim_path.rstrip('/')}/{ground_prim}"


def find_keyword_matched_prims(stage, root_prim, keywords: tuple[str, ...]) -> list[str]:
    from pxr import Usd

    normalized_keywords = tuple(keyword.lower() for keyword in keywords if keyword)
    matched: list[str] = []
    for prim in Usd.PrimRange(root_prim):
        prim_path = str(prim.GetPath())
        if prim == root_prim:
            continue
        path_lower = prim_path.lower()
        if any(keyword in path_lower for keyword in normalized_keywords):
            matched.append(prim_path)
    return prune_descendants(matched)


def prune_descendants(paths: list[str]) -> list[str]:
    """Keep only highest matched prims so removing parents does not double count children."""
    kept: list[str] = []
    for path in sorted(paths, key=lambda value: (value.count("/"), value)):
        prefix = f"{path}/"
        if any(path == parent or path.startswith(f"{parent}/") for parent in kept):
            continue
        kept.append(path)
    return kept


def find_first_scene_root(stage) -> str:
    pseudo_root = stage.GetPseudoRoot()
    children = [str(child.GetPath()) for child in pseudo_root.GetChildren()]
    for candidate in ("/Root", "/World", "/Environment", "/Scene"):
        if stage.GetPrimAtPath(candidate).IsValid():
            return candidate
    if children:
        return children[0]
    raise RuntimeError("The USD stage has no root prims to filter.")


def default_output_path(input_usd: Path) -> Path:
    suffix = input_usd.suffix or ".usd"
    return input_usd.with_name(f"{input_usd.stem}_filtered{suffix}")


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


if __name__ == "__main__":
    main()
