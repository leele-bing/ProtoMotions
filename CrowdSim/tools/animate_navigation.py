"""Animate CrowdSim 2D navigation logs.

The trajectory log is written by CrowdNavigationManager as JSONL. The animation
shows the center-cropped occupancy map, planned paths, current agent positions,
global goals, current A* waypoints, SFM local targets, and car speed traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from PIL import Image, ImageOps


HUMANOID_COLORS = ["#e64a19", "#f48fb1", "#ffb74d", "#ab47bc"]
CAR_COLORS = ["#00bcd4", "#03a9f4", "#4dd0e1", "#26a69a"]
START_COLOR = "#28dc50"
GOAL_COLOR = "#ffeb3b"
CURRENT_WAYPOINT_COLOR = "#ffffff"
LOCAL_TARGET_COLOR = "#ff3df2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 2D CrowdSim navigation animation and car speed plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "trajectory_log",
        nargs="?",
        default="output/crowdsim_navigation/trajectory_latest.jsonl",
        help="Trajectory JSONL written by CrowdNavigationManager.",
    )
    parser.add_argument(
        "--path-log",
        default=None,
        help="Path JSON produced by CrowdNavigationManager. Defaults to trajectory metadata.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Animation output path. Use .gif or .mp4. Defaults to trajectory_log with .gif.",
    )
    parser.add_argument(
        "--speed-output",
        default=None,
        help="Standalone car speed PNG. Defaults beside animation output.",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth recorded frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means use all frames.")
    parser.add_argument("--trail-length", type=int, default=80)
    parser.add_argument(
        "--crop-center-pixels",
        type=int,
        default=800,
        help="Center crop size in map pixels. Use 0 to render the full map.",
    )
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory_path = resolve_path(args.trajectory_log)
    metadata, frames = load_trajectory(trajectory_path)
    if not frames:
        raise RuntimeError(f"No trajectory frames found: {trajectory_path}")

    stride = max(1, int(args.stride))
    frames = frames[::stride]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    path_log = resolve_path(args.path_log) if args.path_log else resolve_log_path(
        metadata.get("path_log"), trajectory_path
    )
    path_data = json.loads(path_log.read_text(encoding="utf-8"))
    output_path = resolve_output_path(args.output, trajectory_path)
    speed_output_path = resolve_speed_output_path(args.speed_output, output_path)

    render_animation(
        metadata=metadata,
        path_data=path_data,
        frames=frames,
        output_path=output_path,
        fps=float(args.fps),
        dpi=int(args.dpi),
        trail_length=max(0, int(args.trail_length)),
        crop_center_pixels=max(0, int(args.crop_center_pixels)),
    )
    render_speed_plot(
        metadata=metadata,
        frames=frames,
        output_path=speed_output_path,
        dpi=int(args.dpi),
    )
    print(f"[CrowdSim] Saved navigation animation: {output_path}")
    print(f"[CrowdSim] Saved car speed plot: {speed_output_path}")


def load_trajectory(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {}
    frames: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            item_type = item.get("type")
            if item_type == "metadata":
                metadata = item
            elif item_type == "frame":
                frames.append(item)
    return metadata, frames


def render_animation(
    metadata: dict[str, Any],
    path_data: dict[str, Any],
    frames: list[dict[str, Any]],
    output_path: Path,
    fps: float,
    dpi: int,
    trail_length: int,
    crop_center_pixels: int,
) -> None:
    map_image, extent = load_map_image(path_data, metadata, crop_center_pixels)
    num_humanoids = int(path_data.get("num_humanoids", metadata.get("num_humanoids", 0)))
    num_cars = int(path_data.get("num_cars", metadata.get("num_cars", 0)))
    num_agents = num_humanoids + num_cars
    times, car_speeds = car_speed_series(frames, num_humanoids, num_cars)
    positions = np.asarray([frame["positions_xy"] for frame in frames], dtype=np.float32)

    fig = plt.figure(figsize=(12.0, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0])
    ax_map = fig.add_subplot(grid[0, 0])
    ax_speed = fig.add_subplot(grid[1, 0])

    ax_map.imshow(map_image, cmap="gray", extent=extent, origin="upper")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_title("CrowdSim 2D Navigation")
    ax_map.set_xlabel("world x [m]")
    ax_map.set_ylabel("world y [m]")
    draw_static_paths(ax_map, path_data)
    ax_map.set_xlim(extent[0], extent[1])
    ax_map.set_ylim(extent[2], extent[3])

    agent_artists = []
    local_target_artists = []
    current_waypoint_artists = []
    trail_artists = []
    for agent_id in range(num_agents):
        agent_type, local_id = agent_identity(agent_id, num_humanoids)
        color = agent_color(agent_type, local_id)
        marker = "o" if agent_type == "humanoid" else "s"
        agent_artists.append(
            ax_map.scatter([], [], s=72, marker=marker, c=color, edgecolors="black", linewidths=0.8, zorder=7)
        )
        local_target_artists.append(
            ax_map.scatter([], [], s=76, marker="x", c=LOCAL_TARGET_COLOR, linewidths=2.0, zorder=8)
        )
        current_waypoint_artists.append(
            ax_map.scatter([], [], s=64, marker="+", c=CURRENT_WAYPOINT_COLOR, linewidths=1.8, zorder=8)
        )
        (trail_line,) = ax_map.plot([], [], color=color, linewidth=1.5, alpha=0.7, zorder=6)
        trail_artists.append(trail_line)

    time_text = ax_map.text(
        0.015,
        0.985,
        "",
        transform=ax_map.transAxes,
        ha="left",
        va="top",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "boxstyle": "round,pad=0.25"},
    )
    legend_handles = [
        ax_map.scatter([], [], s=72, marker="o", c=HUMANOID_COLORS[0], edgecolors="black", label="humanoid"),
        ax_map.scatter([], [], s=72, marker="s", c=CAR_COLORS[0], edgecolors="black", label="car"),
        ax_map.scatter([], [], s=76, marker="x", c=LOCAL_TARGET_COLOR, label="SFM local target"),
        ax_map.scatter([], [], s=64, marker="+", c=CURRENT_WAYPOINT_COLOR, label="current A* waypoint"),
        ax_map.scatter([], [], s=95, marker="*", c=GOAL_COLOR, edgecolors="black", label="global goal"),
    ]
    ax_map.legend(handles=legend_handles, loc="upper right", framealpha=0.85)

    for car_idx in range(num_cars):
        ax_speed.plot(times, car_speeds[:, car_idx], color=agent_color("car", car_idx), label=f"car {car_idx}")
    speed_cursor = ax_speed.axvline(times[0], color="black", linewidth=1.2, alpha=0.7)
    ax_speed.set_xlim(float(times[0]), float(times[-1]) if len(times) > 1 else float(times[0] + 1.0))
    max_speed = float(np.max(car_speeds)) if car_speeds.size else 0.0
    ax_speed.set_ylim(0.0, max(1.0, 1.15 * max_speed))
    ax_speed.set_xlabel("time [s]")
    ax_speed.set_ylabel("car speed [m/s]")
    ax_speed.grid(True, alpha=0.25)
    if num_cars > 0:
        ax_speed.legend(loc="upper right", ncols=min(4, num_cars))

    def update(frame_idx: int):
        frame = frames[frame_idx]
        frame_positions = np.asarray(frame["positions_xy"], dtype=np.float32)
        local_targets = np.asarray(frame["local_targets_xy"], dtype=np.float32)
        current_waypoints = np.asarray(frame["current_waypoints_xy"], dtype=np.float32)
        trail_start = max(0, frame_idx - trail_length)

        for agent_id in range(num_agents):
            agent_artists[agent_id].set_offsets(frame_positions[agent_id][None, :])
            local_target_artists[agent_id].set_offsets(local_targets[agent_id][None, :])
            current_waypoint_artists[agent_id].set_offsets(current_waypoints[agent_id][None, :])
            trail = positions[trail_start : frame_idx + 1, agent_id]
            trail_artists[agent_id].set_data(trail[:, 0], trail[:, 1])

        speed_cursor.set_xdata([times[frame_idx], times[frame_idx]])
        time_text.set_text(
            f"frame={frame_idx + 1}/{len(frames)}  "
            f"nav_step={frame.get('step', 0)}  t={float(frame.get('time', 0.0)):.2f}s"
        )
        return [
            *agent_artists,
            *local_target_artists,
            *current_waypoint_artists,
            *trail_artists,
            speed_cursor,
            time_text,
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000.0 / max(fps, 1e-5),
        blit=False,
    )
    if output_path.suffix.lower() == ".mp4":
        animation.save(output_path, writer=FFMpegWriter(fps=fps), dpi=dpi)
    else:
        animation.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def draw_static_paths(ax, path_data: dict[str, Any]) -> None:
    for agent in path_data.get("agents", []):
        agent_type = str(agent.get("agent_type", "agent"))
        local_id = int(agent.get("local_id", 0))
        color = agent_color(agent_type, local_id)
        path = np.asarray(agent.get("path_xy", []), dtype=np.float32)
        if len(path) >= 2:
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.8, alpha=0.45, zorder=2)
        start = np.asarray(agent.get("start_xy", [0.0, 0.0]), dtype=np.float32)
        goal = np.asarray(agent.get("goal_xy", [0.0, 0.0]), dtype=np.float32)
        ax.scatter(start[0], start[1], s=65, c=START_COLOR, marker="o", edgecolors="black", zorder=4)
        ax.scatter(goal[0], goal[1], s=95, c=GOAL_COLOR, marker="*", edgecolors="black", zorder=4)


def render_speed_plot(
    metadata: dict[str, Any],
    frames: list[dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    num_humanoids = int(metadata.get("num_humanoids", 0))
    num_cars = int(metadata.get("num_cars", 0))
    times, car_speeds = car_speed_series(frames, num_humanoids, num_cars)

    fig, ax = plt.subplots(figsize=(10.0, 4.0), constrained_layout=True)
    for car_idx in range(num_cars):
        ax.plot(times, car_speeds[:, car_idx], color=agent_color("car", car_idx), label=f"car {car_idx}")
    ax.set_title("CrowdSim Car Speed")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed [m/s]")
    ax.grid(True, alpha=0.25)
    if num_cars > 0:
        ax.legend(loc="upper right", ncols=min(4, num_cars))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def car_speed_series(
    frames: list[dict[str, Any]], num_humanoids: int, num_cars: int
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray([float(frame.get("time", idx)) for idx, frame in enumerate(frames)], dtype=np.float32)
    speeds = np.zeros((len(frames), num_cars), dtype=np.float32)
    if num_cars <= 0:
        return times, speeds
    for frame_idx, frame in enumerate(frames):
        velocities = np.asarray(frame["velocities_xy"], dtype=np.float32)
        car_velocities = velocities[num_humanoids : num_humanoids + num_cars]
        speeds[frame_idx] = np.linalg.norm(car_velocities, axis=1)
    return times, speeds


def load_map_image(
    path_data: dict[str, Any],
    metadata: dict[str, Any],
    crop_center_pixels: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    map_path = resolve_path(str(path_data.get("map_path") or metadata["map_path"]))
    image = Image.open(map_path).convert("L")
    image = ImageOps.autocontrast(image)

    resolution = float(path_data.get("map_resolution", metadata["map_resolution"]))
    origin_xy = path_data.get("map_origin_xy", metadata.get("map_origin_xy"))
    if origin_xy is None:
        width, height = image.size
        origin_xy = [-0.5 * (width - 1) * resolution, -0.5 * (height - 1) * resolution]
    origin_x, origin_y = float(origin_xy[0]), float(origin_xy[1])
    width, height = image.size

    crop_left, crop_top, crop_right, crop_bottom = centered_crop_box(
        width=width,
        height=height,
        crop_size=crop_center_pixels,
    )
    if crop_right > crop_left and crop_bottom > crop_top:
        image = image.crop((crop_left, crop_top, crop_right, crop_bottom))

    extent = (
        origin_x + crop_left * resolution,
        origin_x + (crop_right - 1) * resolution,
        origin_y + (height - crop_bottom) * resolution,
        origin_y + (height - 1 - crop_top) * resolution,
    )
    array = np.asarray(image)
    return array, extent


def centered_crop_box(
    width: int,
    height: int,
    crop_size: int,
) -> tuple[int, int, int, int]:
    if crop_size <= 0:
        return 0, 0, width, height

    crop_width = min(width, crop_size)
    crop_height = min(height, crop_size)
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return left, top, left + crop_width, top + crop_height


def agent_identity(agent_id: int, num_humanoids: int) -> tuple[str, int]:
    if agent_id < num_humanoids:
        return "humanoid", agent_id
    return "car", agent_id - num_humanoids


def agent_color(agent_type: str, local_id: int) -> str:
    if agent_type == "humanoid":
        return HUMANOID_COLORS[local_id % len(HUMANOID_COLORS)]
    return CAR_COLORS[local_id % len(CAR_COLORS)]


def resolve_log_path(path_like: str | None, trajectory_path: Path) -> Path:
    if not path_like:
        return trajectory_path.with_name("paths_latest.json")
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    candidate = trajectory_path.parent / path
    if candidate.exists():
        return candidate
    return Path.cwd() / path


def resolve_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def resolve_output_path(output: str | None, trajectory_path: Path) -> Path:
    if output:
        return resolve_path(output)
    return trajectory_path.with_suffix(".gif")


def resolve_speed_output_path(output: str | None, animation_output: Path) -> Path:
    if output:
        return resolve_path(output)
    return animation_output.with_name(f"{animation_output.stem}_car_speed.png")


if __name__ == "__main__":
    main()
