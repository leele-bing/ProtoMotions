"""Fast 2D renderer for CrowdSim navigation logs.

This script renders directly onto a center-cropped occupancy map with PIL and
writes video frames with OpenCV. It is much faster than the Matplotlib animation
path when you only need to inspect SFM/local-target behavior.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


BLACK = (0, 0, 0, 230)
WHITE = (255, 255, 255, 235)
LOCAL_TARGET_OUTLINE = (255, 255, 255, 220)
WAYPOINT_OUTLINE = (20, 20, 20, 220)
VEL_ARROW_COLOR = (0, 135, 255, 230)
DESIRED_ARROW_COLOR = (40, 230, 80, 235)
INTERACT_ARROW_COLOR = (255, 50, 210, 225)
REPULSIVE_ARROW_COLOR = (255, 70, 45, 225)
D_VEL_ARROW_COLOR = (0, 0, 0, 230)
YAW_SOURCE_LABELS = {
    "sfm_target": "SFM",
    "waypoint_fallback": "WP",
    "previous": "PREV",
    "reached": "DONE",
    "reset": "RESET",
    "initial": "INIT",
}
YAW_SOURCE_COLORS = {
    "sfm_target": (40, 230, 80, 230),
    "waypoint_fallback": (255, 215, 0, 240),
    "previous": (255, 120, 0, 240),
    "reached": (120, 120, 120, 230),
    "reset": (180, 180, 255, 230),
    "initial": (180, 180, 255, 230),
}


@dataclass
class MapCanvas:
    static_image: Image.Image
    full_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    resolution: float
    origin_xy: tuple[float, float]

    @property
    def size(self) -> tuple[int, int]:
        return self.static_image.size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast PIL/OpenCV renderer for CrowdSim navigation trajectory logs.",
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
        default="output/crowdsim_navigation/trajectory_latest.mp4",
        help="Output path. Defaults to trajectory_log with _fast.mp4.",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth recorded frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means use all frames.")
    parser.add_argument("--trail-length", type=int, default=400)
    parser.add_argument(
        "--crop-center-pixels",
        type=int,
        default=800,
        help="Center crop size in map pixels. Use 0 to render the full map.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.5,
        help="Output image scale after map cropping.",
    )
    parser.add_argument(
        "--sfm-arrow-scale",
        type=float,
        default=1,
        help="World-space scale applied to SFM vector arrows.",
    )
    parser.add_argument("--no-sfm-arrows", action="store_true")
    parser.add_argument(
        "--no-initial-paths",
        action="store_true",
        help="Do not draw full planned agent paths onto the static map background.",
    )
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--agent-size", type=int, default=6)
    parser.add_argument("--target-size", type=int, default=6)
    parser.add_argument(
        "--show-yaw-source-labels",
        action="store_true",
        help="Label every humanoid with its yaw source. By default only non-SFM sources are labeled.",
    )
    parser.add_argument("--no-text", action="store_true")
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

    canvas = make_canvas(
        path_data=path_data,
        metadata=metadata,
        crop_center_pixels=max(0, int(args.crop_center_pixels)),
    )
    render_video(
        canvas=canvas,
        metadata=metadata,
        path_data=path_data,
        frames=frames,
        output_path=output_path,
        fps=float(args.fps),
        trail_length=max(0, int(args.trail_length)),
        scale=max(0.1, float(args.scale)),
        agent_size=max(2, int(args.agent_size)),
        target_size=max(2, int(args.target_size)),
        line_width=max(1, int(args.line_width)),
        arrow_scale=max(0.0, float(args.sfm_arrow_scale)),
        draw_sfm_arrows=not bool(args.no_sfm_arrows),
        draw_planned_paths=not bool(args.no_initial_paths),
        show_yaw_source_labels=bool(args.show_yaw_source_labels),
        show_text=not bool(args.no_text),
    )
    print(f"[CrowdSim] Saved fast navigation render: {output_path}")


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


def make_canvas(
    path_data: dict[str, Any],
    metadata: dict[str, Any],
    crop_center_pixels: int,
) -> MapCanvas:
    map_path = resolve_path(str(path_data.get("map_path") or metadata["map_path"]))
    image = Image.open(map_path).convert("L")
    image = ImageOps.autocontrast(image).convert("RGB")
    width, height = image.size
    crop_box = centered_crop_box(width, height, crop_center_pixels)
    cropped = image.crop(crop_box)

    resolution = float(path_data.get("map_resolution", metadata["map_resolution"]))
    origin_xy_value = path_data.get("map_origin_xy", metadata.get("map_origin_xy"))
    if origin_xy_value is None:
        origin_xy_value = default_center_origin((width, height), resolution)
    origin_xy = (float(origin_xy_value[0]), float(origin_xy_value[1]))

    return MapCanvas(
        static_image=cropped.convert("RGB"),
        full_size=(width, height),
        crop_box=crop_box,
        resolution=resolution,
        origin_xy=origin_xy,
    )


def render_video(
    canvas: MapCanvas,
    metadata: dict[str, Any],
    path_data: dict[str, Any],
    frames: list[dict[str, Any]],
    output_path: Path,
    fps: float,
    trail_length: int,
    scale: float,
    agent_size: int,
    target_size: int,
    line_width: int,
    arrow_scale: float,
    draw_sfm_arrows: bool,
    draw_planned_paths: bool,
    show_yaw_source_labels: bool,
    show_text: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_humanoids = int(path_data.get("num_humanoids", metadata.get("num_humanoids", 0)))
    num_agents = int(path_data.get("num_agents", len(frames[0]["positions_xy"])))
    full_width, full_height = canvas.full_size
    font = load_font()
    positions = np.asarray([frame["positions_xy"] for frame in frames], dtype=np.float32)
    writer = make_video_writer(output_path, canvas.size, fps, scale)
    path_records = current_path_records(path_data, num_agents)

    try:
        for frame_idx, frame in enumerate(frames):
            apply_path_updates(path_records, frame.get("path_updates", []))
            image = canvas.static_image.copy().convert("RGBA")
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay, "RGBA")

            frame_positions = np.asarray(frame["positions_xy"], dtype=np.float32)
            frame_velocities = np.asarray(frame["velocities_xy"], dtype=np.float32)
            local_targets = np.asarray(frame["local_targets_xy"], dtype=np.float32)
            current_waypoints = np.asarray(frame["current_waypoints_xy"], dtype=np.float32)
            desired_velocities = sfm_vector_field(frame, "sfm_desired_velocities_xy", num_agents)
            interact_forces = sfm_vector_field(frame, "sfm_interact_forces_xy", num_agents)
            repulsive_forces = sfm_vector_field(frame, "sfm_repulsive_forces_xy", num_agents)
            d_vel = sfm_vector_field(frame, "sfm_d_vel_xy", num_agents)
            yaw_sources = humanoid_yaw_sources(frame, num_humanoids)
            trail_start = max(0, frame_idx - trail_length)

            for agent_id in range(num_agents):
                color = agent_color(agent_id)
                path_record = path_records[agent_id]
                if draw_planned_paths:
                    path = path_record.get("path_xy", [])
                    if len(path) >= 2:
                        path_pixels = [
                            world_to_crop_pixel(
                                xy,
                                full_width,
                                full_height,
                                canvas.resolution,
                                canvas.origin_xy,
                                canvas.crop_box,
                            )
                            for xy in path
                        ]
                        draw.line(path_pixels, fill=(*color, 135), width=line_width, joint="curve")
                goal_px = world_to_crop_pixel(
                    path_record.get("goal_xy", [0.0, 0.0]),
                    full_width,
                    full_height,
                    canvas.resolution,
                    canvas.origin_xy,
                    canvas.crop_box,
                )
                draw_goal(draw, goal_px, target_size + 2, (*color, 235))

                trail = positions[trail_start : frame_idx + 1, agent_id]
                trail_pixels = [
                    world_to_crop_pixel(xy, full_width, full_height, canvas.resolution, canvas.origin_xy, canvas.crop_box)
                    for xy in trail
                ]
                if len(trail_pixels) >= 2:
                    draw.line(trail_pixels, fill=(*color, 215), width=line_width, joint="curve")

                waypoint_px = world_to_crop_pixel(
                    current_waypoints[agent_id],
                    full_width,
                    full_height,
                    canvas.resolution,
                    canvas.origin_xy,
                    canvas.crop_box,
                )
                target_px = world_to_crop_pixel(
                    local_targets[agent_id],
                    full_width,
                    full_height,
                    canvas.resolution,
                    canvas.origin_xy,
                    canvas.crop_box,
                )
                position_px = world_to_crop_pixel(
                    frame_positions[agent_id],
                    full_width,
                    full_height,
                    canvas.resolution,
                    canvas.origin_xy,
                    canvas.crop_box,
                )

                if draw_sfm_arrows:
                    draw_world_arrow(
                        draw,
                        frame_positions[agent_id],
                        frame_velocities[agent_id],
                        VEL_ARROW_COLOR,
                        canvas,
                        arrow_scale,
                        width=2,
                    )
                    draw_world_arrow(
                        draw,
                        frame_positions[agent_id],
                        desired_velocities[agent_id],
                        DESIRED_ARROW_COLOR,
                        canvas,
                        arrow_scale,
                        width=2,
                    )
                    draw_world_arrow(
                        draw,
                        frame_positions[agent_id],
                        interact_forces[agent_id],
                        INTERACT_ARROW_COLOR,
                        canvas,
                        arrow_scale*0.2,
                        width=2,
                    )
                    draw_world_arrow(
                        draw,
                        frame_positions[agent_id],
                        repulsive_forces[agent_id],
                        REPULSIVE_ARROW_COLOR,
                        canvas,
                        arrow_scale*0.2,
                        width=2,
                    )
                    draw_world_arrow(
                        draw,
                        frame_positions[agent_id],
                        d_vel[agent_id],
                        D_VEL_ARROW_COLOR,
                        canvas,
                        arrow_scale*0.2,
                        width=2,
                    )
                draw_cross(draw, waypoint_px, target_size, (*color, 210), WAYPOINT_OUTLINE)
                draw_x(draw, target_px, target_size + 2, (*color, 235), LOCAL_TARGET_OUTLINE)
                if agent_id < num_humanoids:
                    draw_circle(draw, position_px, agent_size, (*color, 245), BLACK)
                    yaw_source = yaw_sources[agent_id]
                    if show_yaw_source_labels or yaw_source != "sfm_target":
                        draw_yaw_source_label(draw, font, position_px, yaw_source)
                else:
                    draw_square(draw, position_px, agent_size, (*color, 245), BLACK)

            if show_text:
                draw_status_text(
                    draw,
                    font,
                    frame_idx,
                    len(frames),
                    frame,
                    draw_sfm_arrows,
                    yaw_sources,
                )

            rendered = Image.alpha_composite(image, overlay).convert("RGB")
            writer.write(rendered)
    finally:
        writer.close()


def current_path_records(path_data: dict[str, Any], num_agents: int) -> list[dict[str, Any]]:
    records = [dict(agent) for agent in path_data.get("agents", [])]
    while len(records) < num_agents:
        records.append(
            {
                "agent_id": len(records),
                "goal_xy": [0.0, 0.0],
                "path_xy": [],
            }
        )
    return records[:num_agents]


def apply_path_updates(path_records: list[dict[str, Any]], updates: Any) -> None:
    if not updates:
        return
    for update in updates:
        if not isinstance(update, dict):
            continue
        agent_id = int(update.get("agent_id", -1))
        if 0 <= agent_id < len(path_records):
            path_records[agent_id] = dict(update)


class OpenCvVideoWriter:
    def __init__(self, path: Path, size: tuple[int, int], fps: float, scale: float) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required for MP4 output. Run in env_isaaclab or install opencv-python."
            ) from exc

        self.cv2 = cv2
        self.scale = scale
        width, height = size
        self.output_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, float(fps), self.output_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {path}")

    def write(self, image: Image.Image) -> None:
        if image.size != self.output_size:
            image = image.resize(self.output_size, Image.Resampling.BILINEAR)
        frame_rgb = np.asarray(image, dtype=np.uint8)
        self.writer.write(self.cv2.cvtColor(frame_rgb, self.cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.writer.release()


class GifWriter:
    def __init__(self, path: Path, size: tuple[int, int], fps: float, scale: float) -> None:
        self.path = path
        self.scale = scale
        width, height = size
        self.output_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        self.duration_ms = int(round(1000.0 / max(float(fps), 1e-5)))
        self.frames: list[Image.Image] = []

    def write(self, image: Image.Image) -> None:
        if image.size != self.output_size:
            image = image.resize(self.output_size, Image.Resampling.BILINEAR)
        self.frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE))

    def close(self) -> None:
        if not self.frames:
            return
        self.frames[0].save(
            self.path,
            save_all=True,
            append_images=self.frames[1:],
            duration=self.duration_ms,
            loop=0,
        )


def make_video_writer(path: Path, size: tuple[int, int], fps: float, scale: float):
    suffix = path.suffix.lower()
    if suffix == ".gif":
        return GifWriter(path, size, fps, scale)
    return OpenCvVideoWriter(path, size, fps, scale)


def draw_status_text(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    frame_idx: int,
    num_frames: int,
    frame: dict[str, Any],
    show_arrow_legend: bool,
    yaw_sources: list[str],
) -> None:
    lines = [
        (
            f"frame {frame_idx + 1}/{num_frames}  "
            f"nav_step {int(frame.get('step', 0))}  "
            f"t {float(frame.get('time', 0.0)):.2f}s"
        )
    ]
    if yaw_sources:
        lines.append(f"yaw: {format_yaw_source_counts(yaw_sources)}")
    if show_arrow_legend:
        lines.append("arrows: blue=vel green=sfm red=repulse magenta=interact white=d_vel")
    x, y = 12, 10
    boxes = [draw.textbbox((x, y + idx * 20), line, font=font) for idx, line in enumerate(lines)]
    box = (
        min(item[0] for item in boxes),
        min(item[1] for item in boxes),
        max(item[2] for item in boxes),
        max(item[3] for item in boxes),
    )
    pad = 5
    draw.rounded_rectangle(
        (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
        radius=4,
        fill=(0, 0, 0, 150),
    )
    for idx, line in enumerate(lines):
        draw.text((x, y + idx * 20), line, fill=WHITE, font=font)


def sfm_vector_field(
    frame: dict[str, Any],
    key: str,
    num_agents: int,
) -> np.ndarray:
    if key not in frame:
        raise KeyError(
            f"Trajectory frame is missing '{key}'. Re-run CrowdSim to generate "
            "logs with SFM debug fields."
        )
    values = np.asarray(frame[key], dtype=np.float32)
    if values.shape != (num_agents, 2):
        raise ValueError(f"Expected '{key}' to have shape ({num_agents}, 2), got {values.shape}.")
    values[~np.isfinite(values)] = 0.0
    return values


def humanoid_yaw_sources(frame: dict[str, Any], num_humanoids: int) -> list[str]:
    if "humanoid_yaw_source" not in frame:
        raise KeyError(
            "Trajectory frame is missing 'humanoid_yaw_source'. Re-run CrowdSim to "
            "generate logs with yaw-source debug fields."
        )
    sources = [str(value) for value in frame["humanoid_yaw_source"]]
    if len(sources) != num_humanoids:
        raise ValueError(
            f"Expected 'humanoid_yaw_source' length {num_humanoids}, got {len(sources)}."
        )
    return sources


def format_yaw_source_counts(sources: list[str]) -> str:
    ordered = ["sfm_target", "waypoint_fallback", "previous", "reached", "reset", "initial"]
    counts = {source: sources.count(source) for source in set(sources)}
    parts = []
    for source in ordered:
        count = counts.pop(source, 0)
        if count:
            parts.append(f"{YAW_SOURCE_LABELS.get(source, source)}={count}")
    for source, count in sorted(counts.items()):
        parts.append(f"{source}={count}")
    return " ".join(parts) if parts else "none"


def draw_yaw_source_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    position_px: tuple[int, int],
    source: str,
) -> None:
    label = YAW_SOURCE_LABELS.get(source, source)
    fill = YAW_SOURCE_COLORS.get(source, (255, 255, 255, 235))
    x, y = position_px[0] + 9, position_px[1] - 19
    box = draw.textbbox((x, y), label, font=font)
    pad = 3
    draw.rounded_rectangle(
        (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
        radius=3,
        fill=(0, 0, 0, 150),
    )
    draw.text((x, y), label, fill=fill, font=font)


def draw_world_arrow(
    draw: ImageDraw.ImageDraw,
    start_xy: np.ndarray,
    vector_xy: np.ndarray,
    color: tuple[int, int, int, int],
    canvas: MapCanvas,
    scale: float,
    width: int,
) -> None:
    if scale <= 0.0 or not np.all(np.isfinite(vector_xy)):
        return
    norm = float(np.linalg.norm(vector_xy))
    if norm < 1e-5:
        return
    end_xy = np.asarray(start_xy, dtype=np.float32) + np.asarray(vector_xy, dtype=np.float32) * scale
    start = world_to_crop_pixel(
        start_xy,
        canvas.full_size[0],
        canvas.full_size[1],
        canvas.resolution,
        canvas.origin_xy,
        canvas.crop_box,
    )
    end = world_to_crop_pixel(
        end_xy,
        canvas.full_size[0],
        canvas.full_size[1],
        canvas.resolution,
        canvas.origin_xy,
        canvas.crop_box,
    )
    draw.line((start[0], start[1], end[0], end[1]), fill=color, width=width)
    draw_arrow_head(draw, start, end, color, size=max(5, 2 * width + 4))


def draw_arrow_head(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int, int],
    size: int,
) -> None:
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    length = math.hypot(dx, dy)
    if length < 1e-5:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    p1 = (end[0], end[1])
    p2 = (int(round(end[0] - ux * size + px * size * 0.45)), int(round(end[1] - uy * size + py * size * 0.45)))
    p3 = (int(round(end[0] - ux * size - px * size * 0.45)), int(round(end[1] - uy * size - py * size * 0.45)))
    draw.polygon([p1, p2, p3], fill=color)


def world_to_crop_pixel(
    xy: list[float] | tuple[float, float] | np.ndarray,
    full_width: int,
    full_height: int,
    resolution: float,
    origin_xy: tuple[float, float],
    crop_box: tuple[int, int, int, int],
) -> tuple[int, int]:
    origin_x, origin_y = origin_xy
    crop_left, crop_top, _, _ = crop_box
    pixel_x = int(round((float(xy[0]) - origin_x) / resolution))
    pixel_y = int(round((full_height - 1) - (float(xy[1]) - origin_y) / resolution))
    return pixel_x - crop_left, pixel_y - crop_top


def centered_crop_box(width: int, height: int, crop_size: int) -> tuple[int, int, int, int]:
    if crop_size <= 0:
        return 0, 0, width, height
    crop_width = min(width, crop_size)
    crop_height = min(height, crop_size)
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return left, top, left + crop_width, top + crop_height


def default_center_origin(size: tuple[int, int], resolution: float) -> tuple[float, float]:
    width, height = size
    return (-0.5 * (width - 1) * resolution, -0.5 * (height - 1) * resolution)


def draw_circle(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)


def draw_square(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
) -> None:
    x, y = center
    draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)


def draw_x(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
) -> None:
    x, y = center
    draw.line((x - radius, y - radius, x + radius, y + radius), fill=outline, width=4)
    draw.line((x - radius, y + radius, x + radius, y - radius), fill=outline, width=4)
    draw.line((x - radius, y - radius, x + radius, y + radius), fill=fill, width=2)
    draw.line((x - radius, y + radius, x + radius, y - radius), fill=fill, width=2)


def draw_cross(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
) -> None:
    x, y = center
    draw.line((x - radius, y, x + radius, y), fill=outline, width=4)
    draw.line((x, y - radius, x, y + radius), fill=outline, width=4)
    draw.line((x - radius, y, x + radius, y), fill=fill, width=2)
    draw.line((x, y - radius, x, y + radius), fill=fill, width=2)


def draw_goal(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    fill: tuple[int, int, int, int],
) -> None:
    x, y = center
    points = [
        (
            x + int(round(math.cos(math.pi * 0.5 * idx) * radius)),
            y + int(round(math.sin(math.pi * 0.5 * idx) * radius)),
        )
        for idx in range(4)
    ]
    draw.polygon(points, fill=fill, outline=BLACK)


def agent_color(agent_id: int) -> tuple[int, int, int]:
    hue = (0.08 + 0.61803398875 * float(agent_id)) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
    return tuple(int(round(255 * value)) for value in rgb)


def load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        return ImageFont.load_default()


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
    return trajectory_path.with_name(f"{trajectory_path.stem}_fast.mp4")


if __name__ == "__main__":
    main()
