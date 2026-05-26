# Code Log for Future Development

This file is written for an LLM or developer who continues the evaluation work.

## Repository Context

Project root:

```text
/home/pcl/amp/ProtoMotions
```

Experimental evaluation files live in:

```text
Experimental/
```

Current scripts:

```text
Experimental/compare_motion_quality.py
Experimental/evaluate_navigation.py
```

The user wants experimental scripts to stay outside core code:

- Do not move these scripts into `protomotions/`.
- Avoid coupling them to `CrowdSim/` training/runtime unless absolutely necessary.
- Core CrowdSim files may have user changes; do not revert them.

Known dirty files when this log was created:

```text
CrowdSim/config/env.yaml
CrowdSim/crowd_sim.py
CrowdSim/train_robot_ppo.py
CrowdSim/utils/sensor_stream.py
Experimental/
```

## Motion Quality Script

File:

```text
Experimental/compare_motion_quality.py
```

Purpose:

- Batch-compare ProtoMotions motion quality metrics.
- Default models:
  - `ours=results/smpl_amass/last.ckpt`
  - `masked_mimic=data/pretrained_models/masked_mimic/smpl/last.ckpt`
- Uses ProtoMotions `MimicEvaluator`.
- Uses `resolved_configs.pt`, not `resolved_configs_inference.pt`, because inference configs strip evaluator components.
- Evaluates each checkpoint in an isolated subprocess by default to avoid Isaac simulator state leakage.

Important CLI:

```bash
conda run -n env_isaaclab python Experimental/compare_motion_quality.py \
  --out-dir output/motion_quality_test \
  --gt-error-threshold 0.25
```

Visualization-only mode:

```bash
conda run -n env_isaaclab python Experimental/compare_motion_quality.py \
  --visualize-only output/motion_quality_test/motion_quality_summary.json \
  --out-dir output/motion_quality_test
```

Outputs:

```text
motion_quality_summary.json
motion_quality_metrics.csv
motion_quality_comparison.csv
motion_quality_tracking.<fmt>
motion_quality_smoothness.<fmt>
motion_quality_success.<fmt>
```

Metric interpretation:

- `gt_error`: global translation/body-position error. It is MPJPE-like when computed over rigid body positions.
- `gr_error`: global rotation/body-orientation error.
- `max_joint_error`: maximum rigid body position error, despite the name.
- `success_rate`: evaluator-defined success under configured thresholds.
- `action_delta_*`: frame-to-frame action change, used as a smoothness/control jerk proxy.

Important decision:

- `gt_error_threshold` default was unified to `0.25`.
- Passing `--gt-error-threshold -1` preserves checkpoint thresholds.

Fairness caveat:

- `results/smpl_amass` and `data/pretrained_models/masked_mimic/smpl` do not have identical input conditioning.
- `results/smpl_amass` resembles full mimic/tracking policy input.
- MaskedMimic uses masked targets, masks, time offsets, history, VAE noise, etc.
- Treat results as "same evaluator, same threshold" comparison, not a perfectly homogeneous architecture comparison.

## Navigation Evaluation Script

File:

```text
Experimental/evaluate_navigation.py
```

Purpose:

- Offline evaluator for CrowdSim navigation logs.
- Does not launch Isaac.
- Does not import ProtoMotions or CrowdSim runtime.
- Reads `trajectory_*.jsonl` and optional `paths_*.json`.

The script delays NumPy import until after CLI parsing so `--help` works even in a minimal Python environment.

Default command:

```bash
conda run -n env_isaaclab python Experimental/evaluate_navigation.py \
  --trajectory output/crowdsim_navigation/trajectory_latest.jsonl \
  --out-dir output/navigation_eval_test
```

Multi-run comparison:

```bash
conda run -n env_isaaclab python Experimental/evaluate_navigation.py \
  --runs \
  masked_mimic=output/crowdsim_navigation/trajectory_xxx.jsonl:output/crowdsim_navigation/paths_xxx.json \
  ours=output/crowdsim_navigation/trajectory_yyy.jsonl:output/crowdsim_navigation/paths_yyy.json \
  --out-dir output/navigation_eval_compare
```

Main CLI options:

```text
--agent-type humanoid|car|all
--goal-tolerance 0.75
--max-frame-gap 5.0
--plot / --no-plot
--plot-format png|pdf|svg
--plot-agents 8
```

Outputs:

```text
navigation_eval_summary.json
navigation_eval_metrics.csv
navigation_eval_per_agent.csv
navigation_eval_bars.<fmt>
navigation_eval_trajectories_<label>.<fmt>
```

Trajectory JSONL fields used:

```text
positions_xy
velocities_xy
current_waypoints_xy
goals_xy
waypoint_ids
local_targets_xy
reached
collision_pairs
```

Metadata fields used:

```text
num_humanoids
num_cars
num_agents
path_log
```

Per-agent metrics:

```text
episode_count
success_events
collision_events
success_rate_per_episode
collision_rate_per_episode
final_goal_distance_m
min_goal_distance_m
mean_goal_distance_m
median_goal_distance_m
mean_waypoint_distance_m
p95_waypoint_distance_m
mean_local_target_distance_m
p95_local_target_distance_m
mean_speed_m_s
max_speed_m_s
path_length_m
net_displacement_m
net_displacement_over_path_length
mean_initial_path_distance_m
p95_initial_path_distance_m
```

Aggregate metrics:

```text
episode_count
success_events
collision_events
success_rate_per_episode
collision_rate_per_episode
agent_collision_probability
agent_success_probability
mean_goal_distance_m
min_goal_distance_m
final_goal_distance_m
mean_waypoint_distance_m
p95_waypoint_distance_m
mean_local_target_distance_m
p95_local_target_distance_m
mean_initial_path_distance_m
p95_initial_path_distance_m
mean_speed_m_s
path_length_m
net_displacement_over_path_length
```

Episode inference:

- No explicit `episode_id` exists in current logs.
- Episode boundaries are inferred by:
  - goal coordinate change, or
  - `waypoint_id` decreasing.
- This is good enough for approximate event rates but not perfect.

Collision counting:

- `collision_pairs` is treated as a per-frame set of pairs.
- A collision event is counted when a pair involving the agent appears after not being present in the previous frame.
- This avoids counting the same continuous collision every frame.

Success counting:

- `reached` flags are used.
- Additionally, success is inferred if distance to goal is less than or equal to `--goal-tolerance`.
- Default tolerance is `0.75`.

Path-following precision:

- `mean_waypoint_distance_m` and `mean_local_target_distance_m` are reliable because they use per-frame logged values.
- `mean_initial_path_distance_m` is computed as point-to-polyline distance to the initial path from `paths_*.json`.
- Initial path distance is only valid while the current goal matches the initial path goal.

Known limitation:

- Current `paths_*.json` logs only the initial full path.
- After reset/replan, the full new path is not logged.
- Therefore, do not over-interpret `mean_initial_path_distance_m` over long multi-episode logs.

Recommended future CrowdSim logging additions:

```text
episode_id
path_id
full_path_xy after every replan
reset_reason: success|collision|timeout|manual_reset
goal_sample_step
goal_reached_step
collision_step
```

These additions would make success/collision/path metrics exact instead of inferred.

## Verification Already Run

Syntax checks:

```bash
python -m py_compile Experimental/compare_motion_quality.py
python -m py_compile Experimental/evaluate_navigation.py
```

Navigation test command:

```bash
conda run -n env_isaaclab python Experimental/evaluate_navigation.py \
  --trajectory output/crowdsim_navigation/trajectory_latest.jsonl \
  --out-dir output/navigation_eval_test
```

Observed key metrics on the local sample log:

```text
success_rate_per_episode: 0.663043
collision_rate_per_episode: 0.0869565
mean_waypoint_distance_m: 0.78635
mean_local_target_distance_m: 1.38799
mean_initial_path_distance_m: 0.113026
mean_speed_m_s: 1.20447
```

Generated files:

```text
output/navigation_eval_test/navigation_eval_summary.json
output/navigation_eval_test/navigation_eval_metrics.csv
output/navigation_eval_test/navigation_eval_per_agent.csv
output/navigation_eval_test/navigation_eval_bars.png
output/navigation_eval_test/navigation_eval_trajectories_trajectory_latest.png
```

## Diversity Evaluation TODO

FID is a reasonable direction, but it needs a fixed motion feature space.

Possible feature choices:

```text
root-normalized joint positions
joint velocities
root linear/angular velocity
foot contact pattern
fixed-window motion encoder embedding
```

Recommended implementation plan:

1. Export generated trajectories and reference AMASS/train motions into a common representation.
2. Normalize root translation and heading.
3. Slice motions into fixed-length windows.
4. Extract feature vectors.
5. Compute mean/covariance for generated and reference features.
6. Compute Fréchet distance.

Avoid using the policy being evaluated as the feature encoder. If using a learned encoder, keep it fixed.

