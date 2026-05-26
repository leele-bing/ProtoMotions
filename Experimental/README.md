# Experimental Evaluation Notes

这个目录用于存放外围实验与评估脚本，尽量不修改 `protomotions/` 核心代码，也不和 `CrowdSim/` 的核心训练、仿真逻辑强绑定。

当前关注三类实验：

- 运动质量：mimic/tracking 是否稳定、平滑、接近 target motion。
- 导航性能：在 CrowdSim USD 场景中是否能沿路径到达目标，是否碰撞。
- 多样性：生成动作分布是否接近训练数据，后续可考虑 FID/FVD/feature distribution 等指标。

## 1. 运动质量评估

脚本：

```bash
Experimental/compare_motion_quality.py
```

默认比较：

- `ours=results/smpl_amass/last.ckpt`
- `masked_mimic=data/pretrained_models/masked_mimic/smpl/last.ckpt`

推荐运行：

```bash
conda run -n env_isaaclab python Experimental/compare_motion_quality.py \
  --out-dir output/motion_quality_test \
  --gt-error-threshold 0.25
```

只重新画图，不重新跑仿真：

```bash
conda run -n env_isaaclab python Experimental/compare_motion_quality.py \
  --visualize-only output/motion_quality_test/motion_quality_summary.json \
  --out-dir output/motion_quality_test
```

主要输出：

- `motion_quality_summary.json`：完整结果。
- `motion_quality_metrics.csv`：每个模型的指标表。
- `motion_quality_comparison.csv`：模型间差值/比值。
- `motion_quality_tracking.png`、`motion_quality_smoothness.png` 等图。

关键指标解释：

- `gt_error`：global translation / rigid body position error，全身 rigid bodies 的全局位置误差，接近 MPJPE 的含义。
- `gr_error`：global rotation error，全身 rigid bodies 的旋转误差。
- `max_joint_error`：最大 rigid body position error，不是平均误差，名字里 joint 容易误导。
- `success_rate`：ProtoMotions evaluator 按阈值判断的 tracking 成功率。当前统一把 `gt_error` failure threshold 设为 `0.25`，保证两个模型对比口径一致。
- `action_delta_*`：相邻 step action 的变化量，用于衡量动作控制平滑性；越大代表控制变化越剧烈。

重要问题：

- `results/smpl_amass` 和 `data/pretrained_models/masked_mimic/smpl` 的输入形式不同。
- `results/smpl_amass` 更接近标准 mimic/tracking policy，输入包括 target poses、max coords obs、previous actions 等。
- `masked_mimic` 是 MaskedMimic 模型，包含 masked target、mask、time offsets、history、VAE noise 等条件输入。它不是完全相同的 full-tracking 设置。
- 所以这组比较能说明“在同一 evaluator 和同一阈值下的表现”，但不能完全等价为同构模型公平对比。

## 2. 导航性能评估

脚本：

```bash
Experimental/evaluate_navigation.py
```

这是离线评估器，只读取 CrowdSim 记录的日志，不启动 Isaac，不导入 `protomotions` 或 `CrowdSim` runtime。

输入日志通常来自：

```bash
output/crowdsim_navigation/trajectory_*.jsonl
output/crowdsim_navigation/paths_*.json
```

评估最新日志：

```bash
conda run -n env_isaaclab python Experimental/evaluate_navigation.py \
  --trajectory output/crowdsim_navigation/trajectory_latest.jsonl \
  --out-dir output/navigation_eval_test
```

对比多次运行：

```bash
conda run -n env_isaaclab python Experimental/evaluate_navigation.py \
  --runs \
  masked_mimic=output/crowdsim_navigation/trajectory_xxx.jsonl:output/crowdsim_navigation/paths_xxx.json \
  ours=output/crowdsim_navigation/trajectory_yyy.jsonl:output/crowdsim_navigation/paths_yyy.json \
  --out-dir output/navigation_eval_compare
```

主要输出：

- `navigation_eval_summary.json`：完整结果。
- `navigation_eval_metrics.csv`：每个 run 的聚合指标。
- `navigation_eval_per_agent.csv`：每个 agent 的指标。
- `navigation_eval_bars.png`：关键指标柱状图。
- `navigation_eval_trajectories_*.png`：轨迹图，实线为实际轨迹，虚线为初始 path。

关键指标解释：

- `success_rate_per_episode`：成功事件数 / 推断 episode 数。
- `collision_rate_per_episode`：碰撞事件数 / 推断 episode 数。
- `agent_collision_probability`：至少碰撞过一次的 agent 比例。
- `agent_success_probability`：至少成功到达过一次目标的 agent 比例。
- `mean_waypoint_distance_m`：agent 到当前 waypoint 的平均距离。
- `mean_local_target_distance_m`：agent 到 local target 的平均距离。
- `mean_initial_path_distance_m`：agent 到初始 A* path 的平均横向距离。
- `mean_speed_m_s`：平均速度。
- `path_length_m`：日志期间实际走过的路径长度。

当前日志限制：

- `paths_*.json` 只可靠记录初始 path。
- reset/replan 之后的新完整 path 没有显式保存。
- 因此 `mean_initial_path_distance_m` 只适合衡量初始 path 未变化时的跟随误差。
- 更稳的导航精度指标是 `mean_waypoint_distance_m` 和 `mean_local_target_distance_m`，因为它们来自每一帧日志。

如果后续要做严格导航评估，建议在 CrowdSim 日志里增加：

- `episode_id`
- `path_id`
- 每次 replan 后的 `full_path_xy`
- reset reason：`success`、`collision`、`timeout`、`manual_reset`
- 每个 agent 的 goal sampled time 和 reached time

## 3. 后续多样性评估方向

FID/FVD 类指标是合理的，但需要先定义 motion feature extractor。

可行路线：

- 从 generated motion 和 AMASS/train motion 中抽取同一组 features。
- features 可以是 joint positions、velocities、root velocity、contact pattern，或使用预训练 motion encoder 的 embedding。
- 对两组 feature 计算均值和协方差，再计算 Fréchet distance。

注意：

- 直接对 raw joint position 做 FID 可实现，但受坐标系、朝向、速度分布影响大。
- 更推荐先做 root-normalized、heading-normalized、fixed window 的 motion feature。
- 如果使用 learned encoder，需要固定 encoder，不能用正在评估的 policy 本身作为 encoder。

