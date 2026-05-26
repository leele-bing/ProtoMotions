"""Evaluate CrowdSim navigation trajectories.

This is an offline evaluator for logs produced by CrowdSim's navigation
recorder.  It does not import Isaac, ProtoMotions, or CrowdSim runtime code.
It reads trajectory_*.jsonl and computes path-following precision, goal
success rate, collision rate, and related navigation statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "CrowdSim").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate ProtoMotions project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)


@dataclass
class NavigationRun:
    label: str
    trajectory_path: Path
    path_log_path: Path | None


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate CrowdSim navigation trajectory logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trajectory",
        default="output/crowdsim_navigation/trajectory_latest.jsonl",
        help="Trajectory JSONL to evaluate when --runs is not provided.",
    )
    parser.add_argument(
        "--path-log",
        default=None,
        help="Path JSON. If omitted, uses trajectory metadata path_log when available.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional multi-run specs as label=trajectory.jsonl or label=trajectory.jsonl:path_log.json.",
    )
    parser.add_argument(
        "--agent-type",
        choices=["humanoid", "car", "all"],
        default="humanoid",
        help="Which agents to aggregate.",
    )
    parser.add_argument(
        "--goal-tolerance",
        type=float,
        default=0.75,
        help="Goal distance threshold used to infer success if reached flags are missing.",
    )
    parser.add_argument(
        "--max-frame-gap",
        type=float,
        default=5.0,
        help="Frames with dt larger than median_dt * this value are treated as reset/log gaps for path-length integration.",
    )
    parser.add_argument(
        "--out-dir",
        default="output/navigation_eval",
        help="Directory for evaluation outputs.",
    )
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-format", choices=["png", "pdf", "svg"], default="png")
    parser.add_argument(
        "--plot-agents",
        type=int,
        default=8,
        help="Maximum selected agents to draw per run in trajectory plots.",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    require_numpy()
    out_dir = resolve_output_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = parse_runs(args)
    results = [
        evaluate_run(
            run,
            agent_type=args.agent_type,
            goal_tolerance=args.goal_tolerance,
            max_frame_gap=args.max_frame_gap,
        )
        for run in runs
    ]

    payload = {
        "agent_type": args.agent_type,
        "goal_tolerance": args.goal_tolerance,
        "runs": results,
    }
    write_json(out_dir / "navigation_eval_summary.json", payload)
    write_run_metrics_csv(out_dir / "navigation_eval_metrics.csv", results)
    write_per_agent_csv(out_dir / "navigation_eval_per_agent.csv", results)
    if args.plot:
        write_plots(out_dir, results, args.plot_format, args.plot_agents)

    print(f"[NavigationEval] Summary: {out_dir / 'navigation_eval_summary.json'}")
    print(f"[NavigationEval] Metrics: {out_dir / 'navigation_eval_metrics.csv'}")
    print(f"[NavigationEval] Per-agent: {out_dir / 'navigation_eval_per_agent.csv'}")
    print_short_table(results)


def require_numpy() -> None:
    global np
    try:
        import numpy as np_module
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is required for navigation evaluation. Run this script in the "
            "ProtoMotions/CrowdSim Python environment, or install numpy there."
        ) from exc
    np = np_module


def parse_runs(args: argparse.Namespace) -> list[NavigationRun]:
    if args.runs:
        runs = []
        for spec in args.runs:
            if "=" not in spec:
                raise ValueError(f"Invalid run spec {spec!r}; expected label=trajectory")
            label, paths = spec.split("=", maxsplit=1)
            if ":" in paths:
                trajectory, path_log = paths.split(":", maxsplit=1)
                path_log_path = resolve_path(path_log)
            else:
                trajectory = paths
                path_log_path = None
            runs.append(
                NavigationRun(
                    label=label.strip(),
                    trajectory_path=resolve_path(trajectory.strip()),
                    path_log_path=path_log_path,
                )
            )
        return runs

    trajectory_path = resolve_path(args.trajectory)
    path_log_path = resolve_path(args.path_log) if args.path_log else None
    return [
        NavigationRun(
            label=trajectory_path.stem,
            trajectory_path=trajectory_path,
            path_log_path=path_log_path,
        )
    ]


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (PROJECT_ROOT / path).resolve()


def resolve_output_dir(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def evaluate_run(
    run: NavigationRun,
    agent_type: str,
    goal_tolerance: float,
    max_frame_gap: float,
) -> dict[str, Any]:
    metadata, frames = load_trajectory(run.trajectory_path)
    if not frames:
        raise RuntimeError(f"No frame records found in {run.trajectory_path}")

    path_log_path = run.path_log_path or resolve_metadata_path(
        metadata.get("path_log"), run.trajectory_path
    )
    path_data = load_path_log(path_log_path) if path_log_path is not None else None

    arrays = frames_to_arrays(frames)
    num_humanoids = int(metadata.get("num_humanoids", 0))
    num_cars = int(metadata.get("num_cars", 0))
    num_agents = int(metadata.get("num_agents", arrays["positions"].shape[1]))
    agent_ids = select_agent_ids(agent_type, num_humanoids, num_cars, num_agents)
    if not agent_ids:
        raise RuntimeError(f"No agents selected for agent_type={agent_type}")

    per_agent = []
    for agent_id in agent_ids:
        per_agent.append(
            evaluate_agent(
                agent_id=agent_id,
                arrays=arrays,
                path_data=path_data,
                num_humanoids=num_humanoids,
                goal_tolerance=goal_tolerance,
                max_frame_gap=max_frame_gap,
            )
        )

    aggregate = aggregate_agents(per_agent)
    aggregate.update(
        {
            "label": run.label,
            "trajectory_path": str(run.trajectory_path),
            "path_log_path": None if path_log_path is None else str(path_log_path),
            "num_frames": len(frames),
            "duration_s": float(arrays["times"][-1] - arrays["times"][0]),
            "agent_type": agent_type,
            "num_selected_agents": len(agent_ids),
            "num_humanoids": num_humanoids,
            "num_cars": num_cars,
        }
    )
    return {"label": run.label, "aggregate": aggregate, "per_agent": per_agent}


def load_trajectory(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {}
    frames: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "metadata":
                metadata = record
            elif record.get("type") == "frame":
                frames.append(record)
    return metadata, frames


def resolve_metadata_path(path_like: str | None, trajectory_path: Path) -> Path | None:
    if not path_like:
        return None
    path = Path(path_like).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                (PROJECT_ROOT / path),
                (trajectory_path.parent / path),
                (Path.cwd() / path),
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else None


def load_path_log(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def frames_to_arrays(frames: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "steps": np.asarray([frame.get("step", idx) for idx, frame in enumerate(frames)], dtype=np.int64),
        "times": np.asarray([float(frame.get("time", idx)) for idx, frame in enumerate(frames)], dtype=np.float64),
        "positions": np.asarray([frame["positions_xy"] for frame in frames], dtype=np.float64),
        "velocities": np.asarray([frame["velocities_xy"] for frame in frames], dtype=np.float64),
        "waypoints": np.asarray([frame["current_waypoints_xy"] for frame in frames], dtype=np.float64),
        "goals": np.asarray([frame["goals_xy"] for frame in frames], dtype=np.float64),
        "waypoint_ids": np.asarray([frame["waypoint_ids"] for frame in frames], dtype=np.int64),
        "local_targets": np.asarray([frame["local_targets_xy"] for frame in frames], dtype=np.float64),
        "reached": np.asarray([frame["reached"] for frame in frames], dtype=bool),
        "collision_pairs": [set(tuple(pair) for pair in frame.get("collision_pairs", [])) for frame in frames],
    }


def select_agent_ids(
    agent_type: str, num_humanoids: int, num_cars: int, num_agents: int
) -> list[int]:
    if agent_type == "humanoid":
        return list(range(num_humanoids))
    if agent_type == "car":
        return list(range(num_humanoids, min(num_agents, num_humanoids + num_cars)))
    return list(range(num_agents))


def evaluate_agent(
    agent_id: int,
    arrays: dict[str, Any],
    path_data: dict[str, Any] | None,
    num_humanoids: int,
    goal_tolerance: float,
    max_frame_gap: float,
) -> dict[str, Any]:
    positions = arrays["positions"][:, agent_id]
    velocities = arrays["velocities"][:, agent_id]
    waypoints = arrays["waypoints"][:, agent_id]
    goals = arrays["goals"][:, agent_id]
    local_targets = arrays["local_targets"][:, agent_id]
    waypoint_ids = arrays["waypoint_ids"][:, agent_id]
    reached = arrays["reached"][:, agent_id].copy()
    times = arrays["times"]

    goal_dist = np.linalg.norm(positions - goals, axis=1)
    waypoint_dist = np.linalg.norm(positions - waypoints, axis=1)
    local_target_dist = np.linalg.norm(positions - local_targets, axis=1)
    speed = np.linalg.norm(velocities, axis=1)
    reached = reached | (goal_dist <= goal_tolerance)

    success_events = count_rising_edges(reached)
    collision_events = count_agent_collision_events(agent_id, arrays["collision_pairs"])
    episode_count = infer_episode_count(goals, waypoint_ids)
    path_length = integrate_path_length(positions, times, max_frame_gap)
    net_displacement = float(np.linalg.norm(positions[-1] - positions[0]))

    initial_path_dist = distance_to_initial_path(
        agent_id, positions, goals, path_data, goal_change_tol=1e-3
    )
    initial_path_dist_valid = initial_path_dist[np.isfinite(initial_path_dist)]

    return {
        "agent_id": int(agent_id),
        "agent_type": "humanoid" if agent_id < num_humanoids else "car",
        "local_id": int(agent_id if agent_id < num_humanoids else agent_id - num_humanoids),
        "episode_count": int(episode_count),
        "success_events": int(success_events),
        "collision_events": int(collision_events),
        "success_rate_per_episode": safe_div(success_events, episode_count),
        "collision_rate_per_episode": safe_div(collision_events, episode_count),
        "final_goal_distance_m": float(goal_dist[-1]),
        "min_goal_distance_m": float(np.min(goal_dist)),
        "mean_goal_distance_m": float(np.mean(goal_dist)),
        "median_goal_distance_m": float(np.median(goal_dist)),
        "mean_waypoint_distance_m": float(np.mean(waypoint_dist)),
        "p95_waypoint_distance_m": percentile(waypoint_dist, 95),
        "mean_local_target_distance_m": float(np.mean(local_target_dist)),
        "p95_local_target_distance_m": percentile(local_target_dist, 95),
        "mean_speed_m_s": float(np.mean(speed)),
        "max_speed_m_s": float(np.max(speed)),
        "path_length_m": float(path_length),
        "net_displacement_m": net_displacement,
        "net_displacement_over_path_length": safe_div(net_displacement, path_length),
        "mean_initial_path_distance_m": nan_stat(initial_path_dist_valid, np.mean),
        "p95_initial_path_distance_m": nan_stat(initial_path_dist_valid, lambda x: np.percentile(x, 95)),
    }


def count_rising_edges(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    previous = np.concatenate([[False], values[:-1]])
    return int(np.count_nonzero(values & ~previous))


def count_agent_collision_events(agent_id: int, collision_pairs_by_frame: list[set[tuple[int, int]]]) -> int:
    count = 0
    previous_pairs: set[tuple[int, int]] = set()
    for pairs in collision_pairs_by_frame:
        agent_pairs = {pair for pair in pairs if agent_id in pair}
        count += len(agent_pairs - previous_pairs)
        previous_pairs = agent_pairs
    return int(count)


def infer_episode_count(goals: np.ndarray, waypoint_ids: np.ndarray) -> int:
    return len(infer_episode_ranges(goals, waypoint_ids))


def infer_episode_ranges(goals: np.ndarray, waypoint_ids: np.ndarray) -> list[tuple[int, int]]:
    if len(goals) == 0:
        return []
    goal_changed = np.linalg.norm(goals[1:] - goals[:-1], axis=1) > 1e-3
    waypoint_reset = waypoint_ids[1:] < waypoint_ids[:-1]
    boundaries = np.flatnonzero(goal_changed | waypoint_reset) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(goals)]])
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end > start]


def integrate_path_length(positions: np.ndarray, times: np.ndarray, max_frame_gap: float) -> float:
    if len(positions) < 2:
        return 0.0
    dt = np.diff(times)
    positive_dt = dt[dt > 1e-9]
    median_dt = float(np.median(positive_dt)) if positive_dt.size else 0.0
    valid = np.ones(len(dt), dtype=bool)
    if median_dt > 0:
        valid &= dt <= median_dt * max_frame_gap
    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return float(deltas[valid].sum())


def distance_to_initial_path(
    agent_id: int,
    positions: np.ndarray,
    goals: np.ndarray,
    path_data: dict[str, Any] | None,
    goal_change_tol: float,
) -> np.ndarray:
    distances = np.full((len(positions),), np.nan, dtype=np.float64)
    if path_data is None:
        return distances
    agents = path_data.get("agents", [])
    record = next((item for item in agents if int(item.get("agent_id", -1)) == agent_id), None)
    if record is None:
        return distances
    path = np.asarray(record.get("path_xy", []), dtype=np.float64)
    if len(path) < 2:
        return distances
    initial_goal = np.asarray(record.get("goal_xy", goals[0]), dtype=np.float64)
    valid = np.linalg.norm(goals - initial_goal[None, :], axis=1) <= goal_change_tol
    distances[valid] = point_to_polyline_distances(positions[valid], path)
    return distances


def point_to_polyline_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0,), dtype=np.float64)
    starts = polyline[:-1]
    ends = polyline[1:]
    segments = ends - starts
    denom = np.sum(segments * segments, axis=1).clip(min=1e-12)
    diff = points[:, None, :] - starts[None, :, :]
    t = np.sum(diff * segments[None, :, :], axis=2) / denom[None, :]
    t = np.clip(t, 0.0, 1.0)
    closest = starts[None, :, :] + t[:, :, None] * segments[None, :, :]
    dists = np.linalg.norm(points[:, None, :] - closest, axis=2)
    return np.min(dists, axis=1)


def aggregate_agents(per_agent: list[dict[str, Any]]) -> dict[str, Any]:
    episode_count = sum(item["episode_count"] for item in per_agent)
    success_events = sum(item["success_events"] for item in per_agent)
    collision_events = sum(item["collision_events"] for item in per_agent)
    return {
        "episode_count": int(episode_count),
        "success_events": int(success_events),
        "collision_events": int(collision_events),
        "success_rate_per_episode": safe_div(success_events, episode_count),
        "collision_rate_per_episode": safe_div(collision_events, episode_count),
        "agent_collision_probability": safe_div(
            sum(1 for item in per_agent if item["collision_events"] > 0),
            len(per_agent),
        ),
        "agent_success_probability": safe_div(
            sum(1 for item in per_agent if item["success_events"] > 0),
            len(per_agent),
        ),
        "mean_goal_distance_m": mean_key(per_agent, "mean_goal_distance_m"),
        "min_goal_distance_m": mean_key(per_agent, "min_goal_distance_m"),
        "final_goal_distance_m": mean_key(per_agent, "final_goal_distance_m"),
        "mean_waypoint_distance_m": mean_key(per_agent, "mean_waypoint_distance_m"),
        "p95_waypoint_distance_m": mean_key(per_agent, "p95_waypoint_distance_m"),
        "mean_local_target_distance_m": mean_key(per_agent, "mean_local_target_distance_m"),
        "p95_local_target_distance_m": mean_key(per_agent, "p95_local_target_distance_m"),
        "mean_initial_path_distance_m": mean_key(per_agent, "mean_initial_path_distance_m"),
        "p95_initial_path_distance_m": mean_key(per_agent, "p95_initial_path_distance_m"),
        "mean_speed_m_s": mean_key(per_agent, "mean_speed_m_s"),
        "path_length_m": mean_key(per_agent, "path_length_m"),
        "net_displacement_over_path_length": mean_key(per_agent, "net_displacement_over_path_length"),
    }


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def nan_stat(values: np.ndarray, fn) -> float:
    if values.size == 0:
        return float("nan")
    return float(fn(values))


def mean_key(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_run_metrics_csv(path: Path, results: list[dict[str, Any]]) -> None:
    keys = sorted({key for result in results for key in result["aggregate"].keys()})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "metric", "value"])
        writer.writeheader()
        for result in results:
            aggregate = result["aggregate"]
            for key in keys:
                if key in aggregate and isinstance(aggregate[key], (int, float)):
                    writer.writerow({"label": result["label"], "metric": key, "value": aggregate[key]})


def write_per_agent_csv(path: Path, results: list[dict[str, Any]]) -> None:
    keys = sorted({key for result in results for row in result["per_agent"] for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["label", *keys])
        writer.writeheader()
        for result in results:
            for row in result["per_agent"]:
                writer.writerow({"label": result["label"], **row})


def write_plots(
    out_dir: Path,
    results: list[dict[str, Any]],
    plot_format: str,
    plot_agents: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[NavigationEval] matplotlib unavailable; skipping plots: {exc}")
        return

    metrics = [
        ("success_rate_per_episode", "success / episode"),
        ("collision_rate_per_episode", "collision / episode"),
        ("mean_waypoint_distance_m", "waypoint error (m)"),
        ("mean_local_target_distance_m", "local target error (m)"),
        ("mean_initial_path_distance_m", "initial path error (m)"),
        ("mean_speed_m_s", "speed (m/s)"),
    ]
    labels = [result["label"] for result in results]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), squeeze=False)
    for axis, (metric, title) in zip(axes.flatten(), metrics):
        values = [result["aggregate"].get(metric, float("nan")) for result in results]
        axis.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"][: len(values)])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"navigation_eval_bars.{plot_format}", dpi=180)
    plt.close(fig)

    for result in results:
        write_run_trajectory_plot(out_dir, result, plot_format, plot_agents, plt)


def write_run_trajectory_plot(
    out_dir: Path,
    result: dict[str, Any],
    plot_format: str,
    plot_agents: int,
    plt: Any,
) -> None:
    aggregate = result["aggregate"]
    metadata, frames = load_trajectory(Path(aggregate["trajectory_path"]))
    if not frames:
        return
    arrays = frames_to_arrays(frames)
    agent_ids = [row["agent_id"] for row in result["per_agent"][: max(plot_agents, 0)]]
    if not agent_ids:
        return

    path_data = load_path_log(Path(aggregate["path_log_path"])) if aggregate.get("path_log_path") else None
    initial_paths = {}
    if path_data:
        for item in path_data.get("agents", []):
            initial_paths[int(item.get("agent_id", -1))] = item.get("path_xy", [])

    fig, axis = plt.subplots(figsize=(8, 8))
    colors = plt.get_cmap("tab10")
    for idx, agent_id in enumerate(agent_ids):
        color = colors(idx % 10)
        positions = arrays["positions"][:, agent_id]
        goals = arrays["goals"][:, agent_id]
        path = initial_paths.get(agent_id, [])
        if len(path) >= 2:
            path_xy = np.asarray(path, dtype=np.float64)
            axis.plot(
                path_xy[:, 0],
                path_xy[:, 1],
                linestyle="--",
                linewidth=1.0,
                color=color,
                alpha=0.35,
            )
        axis.plot(positions[:, 0], positions[:, 1], linewidth=1.2, color=color, label=f"agent {agent_id}")
        axis.scatter(positions[0, 0], positions[0, 1], marker="o", s=24, color=color)
        axis.scatter(goals[-1, 0], goals[-1, 1], marker="x", s=36, color=color)

    axis.set_title(f"{result['label']} trajectories")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in result["label"])
    fig.savefig(out_dir / f"navigation_eval_trajectories_{safe_label}.{plot_format}", dpi=180)
    plt.close(fig)


def print_short_table(results: list[dict[str, Any]]) -> None:
    keys = [
        "success_rate_per_episode",
        "collision_rate_per_episode",
        "mean_waypoint_distance_m",
        "mean_local_target_distance_m",
        "mean_initial_path_distance_m",
        "mean_speed_m_s",
    ]
    print("\n[NavigationEval] Key metrics")
    print("label," + ",".join(keys))
    for result in results:
        values = [format_value(result["aggregate"].get(key)) for key in keys]
        print(f"{result['label']}," + ",".join(values))


def format_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if math.isnan(float(value)):
            return ""
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):.6g}"


if __name__ == "__main__":
    main()
