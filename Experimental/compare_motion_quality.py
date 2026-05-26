"""Compare humanoid motion quality metrics across MaskedMimic checkpoints.

This tool runs ProtoMotions' MimicEvaluator with training-time evaluator configs,
not resolved_configs_inference.pt, because the inference configs intentionally
strip evaluation components.  Each checkpoint is evaluated in a separate
subprocess by default so IsaacLab simulator state does not leak across models.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "protomotions").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate ProtoMotions project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch compare ProtoMotions motion-quality evaluation metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "ours=results/smpl_amass/last.ckpt",
            "masked_mimic=data/pretrained_models/masked_mimic/smpl/last.ckpt",
        ],
        help="Model specs as label=checkpoint_path.",
    )
    parser.add_argument(
        "--motion-file",
        default="../Assets/motion/amass_smpl_validation.pt",
        help="Shared motion library for evaluation.",
    )
    parser.add_argument("--simulator", default="isaaclab")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument(
        "--gt-error-threshold",
        type=float,
        default=0.25,
        help="Override gt_error failure threshold for all evaluated models. Use a negative value to keep checkpoint configs.",
    )
    parser.add_argument(
        "--out-dir",
        default="output/motion_quality_test",
        help="Directory for merged comparison outputs.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write summary bar charts after evaluation.",
    )
    parser.add_argument(
        "--plot-format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Plot file format.",
    )
    parser.add_argument(
        "--visualize-only",
        default=None,
        help="Existing motion_quality_summary.json to visualize without rerunning simulation.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run simulator headless.",
    )
    parser.add_argument(
        "--single-process",
        action="store_true",
        help="Evaluate all models in this process. Faster, but less robust for IsaacLab.",
    )

    # Internal subprocess mode.
    parser.add_argument("--single-model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-label", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-output", default=None, help=argparse.SUPPRESS)
    return parser


parser = create_parser()
args, _unknown = parser.parse_known_args()

if args.visualize_only is None:
    from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

    AppLauncher = import_simulator_before_torch(args.simulator)

    import torch  # noqa: E402
    from lightning.fabric import Fabric  # noqa: E402

    from protomotions.agents.evaluators.config import MimicEvaluatorConfig  # noqa: E402
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator  # noqa: E402
    from protomotions.simulator.factory import update_simulator_config_for_test  # noqa: E402
    from protomotions.utils.component_builder import build_all_components  # noqa: E402
    from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
    from protomotions.utils.hydra_replacement import get_class  # noqa: E402
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes  # noqa: E402
else:
    AppLauncher = None


def main() -> None:
    parsed = parser.parse_args()
    if parsed.visualize_only is not None:
        summary_path = resolve_path(parsed.visualize_only)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        out_dir = resolve_output_dir(parsed.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_plots(out_dir, payload["results"], parsed.plot_format)
        print(f"[MotionQuality] Plots written to: {out_dir}")
        return

    if parsed.single_model is not None:
        if parsed.single_label is None or parsed.single_output is None:
            raise ValueError("--single-model requires --single-label and --single-output")
        result = evaluate_checkpoint(
            label=parsed.single_label,
            checkpoint=resolve_path(parsed.single_model),
            motion_file=resolve_path(parsed.motion_file),
            simulator_name=parsed.simulator,
            num_envs=parsed.num_envs,
            headless=parsed.headless,
            max_eval_steps=parsed.max_eval_steps,
            gt_error_threshold=parsed.gt_error_threshold,
        )
        write_json(Path(parsed.single_output), result)
        return

    model_specs = [parse_model_spec(spec) for spec in parsed.models]
    out_dir = resolve_output_dir(parsed.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if parsed.single_process:
        results = [
            evaluate_checkpoint(
                label=label,
                checkpoint=resolve_path(checkpoint),
                motion_file=resolve_path(parsed.motion_file),
                simulator_name=parsed.simulator,
                num_envs=parsed.num_envs,
                headless=parsed.headless,
                max_eval_steps=parsed.max_eval_steps,
                gt_error_threshold=parsed.gt_error_threshold,
            )
            for label, checkpoint in model_specs
        ]
    else:
        results = run_isolated_evaluations(parsed, model_specs, out_dir)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "motion_file": str(resolve_path(parsed.motion_file)),
        "simulator": parsed.simulator,
        "num_envs": parsed.num_envs,
        "max_eval_steps": parsed.max_eval_steps,
        "gt_error_threshold": parsed.gt_error_threshold,
        "headless": parsed.headless,
        "results": results,
        "comparison": build_comparison(results),
    }

    summary_path = out_dir / "motion_quality_summary.json"
    metrics_path = out_dir / "motion_quality_metrics.csv"
    comparison_path = out_dir / "motion_quality_comparison.csv"
    write_json(summary_path, payload)
    write_metrics_csv(metrics_path, results)
    write_comparison_csv(comparison_path, payload["comparison"])
    if parsed.plot:
        write_plots(out_dir, results, parsed.plot_format)

    print(f"[MotionQuality] Summary: {summary_path}")
    print(f"[MotionQuality] Metric rows: {metrics_path}")
    print(f"[MotionQuality] Pairwise comparison: {comparison_path}")
    if parsed.plot:
        print(f"[MotionQuality] Plots: {out_dir / ('motion_quality_tracking.' + parsed.plot_format)}")
    print_short_table(results)


def parse_model_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        path = spec.strip()
        return Path(path).parent.name or Path(path).stem, path
    label, path = spec.split("=", maxsplit=1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid model spec: {spec!r}")
    return label, path


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


def run_isolated_evaluations(
    parsed: argparse.Namespace,
    model_specs: list[tuple[str, str]],
    out_dir: Path,
) -> list[dict[str, Any]]:
    results = []
    for label, checkpoint in model_specs:
        single_output = out_dir / f"single_{sanitize_label(label)}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single-model",
            str(resolve_path(checkpoint)),
            "--single-label",
            label,
            "--single-output",
            str(single_output),
            "--motion-file",
            str(resolve_path(parsed.motion_file)),
            "--simulator",
            parsed.simulator,
            "--num-envs",
            str(parsed.num_envs),
            "--headless" if parsed.headless else "--no-headless",
        ]
        if parsed.max_eval_steps is not None:
            command.extend(["--max-eval-steps", str(parsed.max_eval_steps)])
        command.extend(["--gt-error-threshold", str(parsed.gt_error_threshold)])

        print(f"[MotionQuality] Evaluating {label}: {resolve_path(checkpoint)}")
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        results.append(json.loads(single_output.read_text(encoding="utf-8")))
    return results


def sanitize_label(label: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in label)


def evaluate_checkpoint(
    label: str,
    checkpoint: Path,
    motion_file: Path,
    simulator_name: str,
    num_envs: int,
    headless: bool,
    max_eval_steps: int | None,
    gt_error_threshold: float,
) -> dict[str, Any]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not motion_file.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    config_path = checkpoint.parent / "resolved_configs.pt"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Training resolved config not found: {config_path}. "
            "This comparison requires resolved_configs.pt, not inference configs."
        )

    print(f"[CrowdSim] Loading training configs: {config_path}")
    resolved_configs = torch.load(config_path, map_location="cpu", weights_only=False)
    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]

    current_simulator = simulator_config._target_.split(".")[-3]
    if simulator_name != current_simulator:
        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=simulator_name,
            robot_config=robot_config,
        )

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)
    simulator_config.num_envs = int(num_envs)
    simulator_config.headless = bool(headless)
    motion_lib_config.motion_file = str(motion_file)

    if max_eval_steps is not None:
        agent_config.evaluator.max_eval_steps = int(max_eval_steps)

    if not isinstance(agent_config.evaluator, MimicEvaluatorConfig):
        raise TypeError(
            f"{checkpoint} does not have a MimicEvaluatorConfig in resolved_configs.pt: "
            f"{type(agent_config.evaluator).__name__}"
        )
    if not agent_config.evaluator.evaluation_components:
        raise ValueError(f"{checkpoint} evaluator has no evaluation_components.")
    if gt_error_threshold >= 0:
        set_evaluation_threshold(
            agent_config.evaluator,
            component_name="gt_error",
            threshold=gt_error_threshold,
        )

    fabric_config = FabricConfig(
        accelerator="cpu" if simulator_name == "mujoco" else "gpu",
        devices=1,
        num_nodes=1,
        loggers=[],
        callbacks=[],
    )
    fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    simulator_extra_params = {}
    if simulator_name == "isaaclab":
        if AppLauncher is None:
            raise RuntimeError("IsaacLab AppLauncher is unavailable.")
        app_launcher = AppLauncher({"headless": headless, "device": str(fabric.device)})
        simulator_extra_params["simulation_app"] = app_launcher.app

    env = None
    try:
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
            **simulator_extra_params,
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
        agent.evaluator.eval_count = 0
        agent.evaluator.config.save_predicted_motion_lib_every = None
        if hasattr(agent.evaluator, "_update_motion_sampling_weights"):
            agent.evaluator._update_motion_sampling_weights = lambda: None
        metrics, score = agent.evaluator.evaluate()
        metrics = {key: float(value) for key, value in sorted(metrics.items())}
        return {
            "label": label,
            "checkpoint": str(checkpoint),
            "config": str(config_path),
            "motion_file": str(motion_file),
            "score": None if score is None else float(score),
            "gt_error_threshold": None if gt_error_threshold < 0 else float(gt_error_threshold),
            "metrics": metrics,
        }
    finally:
        if env is not None and hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()


def build_comparison(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(results) < 2:
        return []
    baseline = results[0]
    rows = []
    baseline_metrics = baseline["metrics"]
    for result in results[1:]:
        keys = sorted(set(baseline_metrics) | set(result["metrics"]))
        for key in keys:
            base_value = baseline_metrics.get(key)
            value = result["metrics"].get(key)
            delta = None if base_value is None or value is None else value - base_value
            rows.append(
                {
                    "baseline": baseline["label"],
                    "model": result["label"],
                    "metric": key,
                    "baseline_value": base_value,
                    "model_value": value,
                    "model_minus_baseline": delta,
                }
            )
    return rows


def set_evaluation_threshold(evaluator_config, component_name: str, threshold: float) -> None:
    component = evaluator_config.evaluation_components.get(component_name)
    if component is None:
        raise KeyError(f"Missing evaluation component '{component_name}'")
    params = getattr(component, "static_params", None)
    if params is not None:
        params["threshold"] = float(threshold)
        return
    if hasattr(component, "threshold"):
        component.threshold = float(threshold)
        return
    raise TypeError(f"Cannot set threshold on evaluation component '{component_name}'")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_metrics_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "checkpoint", "metric", "value"])
        writer.writeheader()
        for result in results:
            for metric, value in result["metrics"].items():
                writer.writerow(
                    {
                        "label": result["label"],
                        "checkpoint": result["checkpoint"],
                        "metric": metric,
                        "value": value,
                    }
                )


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "baseline",
        "model",
        "metric",
        "baseline_value",
        "model_value",
        "model_minus_baseline",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_short_table(results: list[dict[str, Any]]) -> None:
    preferred = [
        "eval/success_rate",
        "eval/gt_error/mean",
        "eval/max_joint_error/mean",
        "eval/gr_error/mean",
        "eval/normalized_jerk_mean",
        "eval/high_jerk_frame_percentage_mean",
        "eval/action_delta_mean_rad",
        "eval/action_rate_mean_rad_s",
    ]
    print("\n[MotionQuality] Key metrics")
    print("model," + ",".join(preferred))
    for result in results:
        values = [format_metric(result["metrics"].get(metric)) for metric in preferred]
        print(f"{result['label']}," + ",".join(values))


def format_metric(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


def write_plots(out_dir: Path, results: list[dict[str, Any]], plot_format: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[MotionQuality] matplotlib unavailable; skipping plots: {exc}")
        return

    plot_groups = {
        "tracking": [
            ("eval/success_rate", "success rate"),
            ("eval/gt_error/mean", "mean body pos error (m)"),
            ("eval/max_joint_error/mean", "mean max body pos error (m)"),
            ("eval/gr_error/mean", "mean body rot error (rad)"),
        ],
        "smoothness": [
            ("eval/normalized_jerk_mean", "normalized jerk"),
            ("eval/high_jerk_frame_percentage_mean", "high jerk windows (%)"),
            ("eval/action_delta_mean_rad", "action delta mean (rad)"),
            ("eval/action_rate_mean_rad_s", "action rate mean (rad/s)"),
        ],
    }

    for group_name, metrics in plot_groups.items():
        available = [
            (metric_key, label)
            for metric_key, label in metrics
            if any(metric_key in result["metrics"] for result in results)
        ]
        if not available:
            continue

        fig, axes = plt.subplots(
            1,
            len(available),
            figsize=(4.0 * len(available), 3.2),
            squeeze=False,
        )
        model_labels = [result["label"] for result in results]
        for axis, (metric_key, metric_label) in zip(axes[0], available):
            values = [
                result["metrics"].get(metric_key, float("nan"))
                for result in results
            ]
            axis.bar(model_labels, values, color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"][: len(values)])
            axis.set_title(metric_label)
            axis.tick_params(axis="x", rotation=25)
            axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"motion_quality_{group_name}.{plot_format}", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
