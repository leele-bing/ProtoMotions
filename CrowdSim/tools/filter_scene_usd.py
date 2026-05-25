"""Create a filtered USD scene by deactivating/removing keyword-matched prims."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch("isaaclab")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a USD scene using keyword-matched prim paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-usd", default="/home/pcl/amp/Assets/Warehouse/warehouse.usd")
    parser.add_argument("--output-usd", default="/home/pcl/amp/Assets/Warehouse/warehouse_removed.usd")
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_usd = resolve_path(args.input_usd)
    output_usd = resolve_path(args.output_usd) if args.output_usd else default_output_path(input_usd)
    root_prim = str(args.root_prim) if args.root_prim else None
    keywords = tuple(args.keywords)

    app_launcher = AppLauncher({"headless": True})
    try:
        matched = filter_usd_scene(
            input_usd=input_usd,
            output_usd=output_usd,
            root_prim_path=root_prim,
            keywords=keywords,
            mode=args.mode,
            dry_run=args.dry_run,
        )
        action = "Would filter" if args.dry_run else "Filtered"
        print(f"[CrowdSim] {action} {len(matched)} prim(s) from {input_usd}", flush=True)
        for path in matched[:80]:
            print(f"  {path}", flush=True)
        if len(matched) > 80:
            print(f"  ... {len(matched) - 80} more", flush=True)
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
) -> list[str]:
    import omni.usd
    from isaaclab.sim.utils import stage as stage_utils
    from pxr import Usd

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

    stage_utils.update_stage()
    saved = stage_utils.save_stage(str(output_usd), save_and_reload_in_place=False)
    if not saved:
        raise RuntimeError(f"Failed to save filtered USD: {output_usd}")
    return matched


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
