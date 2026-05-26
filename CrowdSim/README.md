# CrowdSim

CrowdSim 是基于 `NVlabs/ProtoMotions` 和 IsaacLab 扩展出的导航仿真模块，用来构建 humanoid 与移动机器人共存的室内导航环境。

当前主要目标是：

- 在同一个全局 USD 场景中加载多个 SMPL humanoid 和多台移动机器人。
- 从 Isaac Sim Occupancy Map 导出的地图 YAML 中读取地图、分辨率和原点信息。
- 在可通行区域内采样起点和终点，并用 A* 规划全局路径。
- 使用 SFM 作为所有 agent 的局部参考规划器。
- humanoid 由 MaskedMimic 控制，目标由导航模块生成的 pelvis future targets 接入。
- car 可以使用 SFM + 差速控制，也可以切换到 PPO 策略训练。
- 记录轨迹、局部目标、速度和碰撞信息，并支持 2D 动画回放。

`CrowdSim/backup/` 中是开发过程中的临时脚本，当前主流程不依赖这些文件。

## 目录结构

- `crowd_sim.py`：正常仿真入口，加载场景、humanoid、car、导航、传感器和可视化。
- `train_robot_ppo.py`：小车 PPO 导航训练入口。
- `sim_agent.py`：加载 ProtoMotions checkpoint、motion file、env 和 agent。
- `sim_world.py`：USD 场景、机器人 USD、传感器和 scene entity 的创建。
- `nav_task.py`：地图读取、起终点采样、A* 全局路径规划、路径/目标 marker。
- `navigation.py`：CrowdSim 导航主逻辑，负责 SFM local target、reset、collision、日志记录和控制调度。
- `robot_rl_navigation.py`：小车 RL 观测、奖励、done、episode reset。
- `robot_ppo.py`：轻量 PPO actor-critic 实现，支持 vector observation + CNN 局部地图。
- `differential_control.py`：手写差速控制器，把小车局部目标转换成左右轮速度。
- `plan/planning.py`：A* 路径搜索。
- `plan/sfm.py`：Social Force Model 局部规划。
- `tools/filter_scene_usd.py`：离线过滤 USD，按关键词 deactivate 或 remove prim。
- `tools/visualize_paths.py`：将路径 JSON 画成静态 PNG。
- `tools/animate_navigation.py`：使用 Matplotlib 生成 2D 导航动画和小车速度曲线。
- `tools/render_navigation_fast.py`：使用 PIL/OpenCV 快速渲染轨迹 MP4。
- `utils/sensor_stream.py`：小车相机数据录制。
- `utils/humanoid_state_recorder.py`：humanoid root、joint、action 状态录制。
- `utils/map_metadata.py`：读取 occupancy map YAML，并统一地图坐标转换信息。

## 配置文件

主配置分为两份：

- `CrowdSim/config/env.yaml`：控制仿真、场景、humanoid、car、传感器、导航和记录。
- `CrowdSim/config/ppo.yaml`：控制小车 PPO 策略、观测、奖励、网络和训练超参数。

常用场景配置：

```yaml
scene:
  prim_path: /World/Scene
  z_offset: 0.0
  scene_usd: ../Assets/Warehouse/warehouse_removed.usd
  scene_map: ../Assets/Warehouse/warehouse.yaml
```

`scene_map` 应该指向 Isaac Sim Occupancy Map 导出的 YAML。CrowdSim 会从 YAML 中读取地图图片、`resolution`、`origin`、`negate`、`free_thresh` 等信息，不再从 `env.yaml` 中单独配置地图分辨率。

常用导航配置：

```yaml
car:
  rl_policy: true
  policy_checkpoint: output/crowdsim_robot_ppo/latest/robot_ppo_latest.pt

navigation:
  enabled: true
  path:
    seed: 7
    planning_step_size: 0.5
    planning_clearance: 0.25
    min_start_goal_distance: 7.0
    max_start_goal_distance: 10.0
    min_spawn_spacing: 1.5

  goal_tolerance: 0.75
  waypoint_tolerance: 0.5

  local:
    method: sfm
    target_timestep: 1
    max_speed: 2
    agent_radius: 0.35
    safe_distance: 0.6
    neighbor_radius: 4.0

  update_hz: 10.0
```

含义：

- `planning_step_size`：A* 搜索和输出路径 waypoint 的物理间距，单位 m。
- `planning_clearance`：规划时对障碍物膨胀的安全距离。
- `min_start_goal_distance` / `max_start_goal_distance`：reset 或初始化时，终点在起点周围的采样距离范围。
- `min_spawn_spacing`：初始化采样起点时，不同 agent 起点之间的最小距离。
- `goal_tolerance`：agent 距离终点多近算 reach。
- `waypoint_tolerance`：agent 距离当前 A* waypoint 多近时推进到下一个 waypoint。
- `local.target_timestep`：SFM local target 的预测时间，目前通常设为 1 秒。
- `local.neighbor_radius`：SFM 和 RL 邻居观测使用的邻居搜索范围。
- `update_hz`：导航逻辑更新频率，包含 waypoint 更新、SFM local target、collision 检测和记录。

## 运行普通仿真

```bash
python CrowdSim/crowd_sim.py \
  --env-config CrowdSim/config/env.yaml \
  --ppo-config CrowdSim/config/ppo.yaml \
  --num-envs 4
```

常用参数：

- `--num-envs`：humanoid 数量。当前设计中通常每个 humanoid 对应一台 car，因此总 agent 数约为 `num_humanoids + num_robots`。
- `--headless`：无 GUI 运行；不加这个参数时打开 viewer。`crowd_sim.py` 和 `train_robot_ppo.py` 都只使用这个参数控制是否可视化。
- `--scene-physics`：给全局场景启用 physics scene 相关设置。
- `--full-eval`：显式走 ProtoMotions full evaluation。默认运行不会开启 humanoid evaluator，而是使用手写 MaskedMimic inference loop。

如果要加载小车，需要在 `env.yaml` 中设置：

```yaml
car:
  usd: nova_carter
  prim_name: Car
  z: 0.0
```

`car.usd` 可以是：

- `nova_carter`
- `jetbot`
- 本地 USD 路径
- Omniverse URI

当前更推荐用 `nova_carter`，尺寸和速度参数更接近真实室内移动平台。

## USD 场景预处理

如果场景里有天花板、无关家具、遮挡物等，可以先离线生成 filtered USD：

```bash
python CrowdSim/tools/filter_scene_usd.py \
  --input-usd ../Assets/Warehouse/warehouse.usd \
  --output-usd ../Assets/Warehouse/warehouse_removed.usd \
  --keywords ceiling roof light \
  --mode deactivate
```

说明：

- `--keywords` 会匹配 prim path。
- `--mode deactivate` 会写入 `active=false`，比删除更容易回退。
- `--root-prim` 可以不传，脚本会自动寻找场景根节点。
- 生成后把 `env.yaml -> scene.scene_usd` 指向新的 USD。

## 导航逻辑

每个 agent 都有：

- 起点 `start_xy`
- 终点 `goal_xy`
- A* 全局路径 `path_xy`
- 当前 A* waypoint
- SFM local target
- reach / collision / timeout reset 状态

初始化时：

1. 从 occupancy map 的白色可通行区域采样起点。
2. 在起点周围 `[min_start_goal_distance, max_start_goal_distance]` 范围内采样终点。
3. 对每个 agent 规划一条 A* 路径。

运行时：

1. 根据当前位置推进当前 A* waypoint。
2. SFM 根据当前位置、速度、邻居和路径方向计算 1 秒后的 local target。
3. humanoid 使用 MaskedMimic，导航模块把 SFM local target 和 A* path 采样点转换成 pelvis future targets。
4. car 在 `car.rl_policy: true` 时由 PPO 直接根据全局目标等 RL observation 输出线速度/角速度。
5. SFM 仍会为 car 计算 local target 并写入日志，但不会向 car 下发速度控制；`car.rl_policy: false` 时小车保持静止。
6. 如果 reach、collision 或 timeout，agent reset 后重新采样目标并重新规划路径。

## Humanoid Env 与 Navigation Env 的关系

当前系统里其实有两层“环境”概念，需要分清：

```text
ProtoMotions / MaskedMimic humanoid env
  负责 humanoid 的物理仿真、动作策略、motion library、reset_buf 和 dones

CrowdSim navigation manager
  负责地图采样、A* 路径、SFM local target、car 控制、导航 done 和轨迹记录
```

它们不是两个独立的 Isaac Sim world，而是在同一个仿真世界里协作：

- humanoid 是 ProtoMotions env 的主 robot，每个 `num-envs` 对应一个 humanoid。
- car 是 CrowdSim 额外 patch 到 IsaacLab scene 里的 articulation，通常每个 humanoid env 对应一台 car。
- navigation manager 把 humanoid 和 car 都视为平面 agent，统一编号为：

```text
agent_id = 0 ... num_humanoids - 1
  humanoid

agent_id = num_humanoids ... num_humanoids + num_robots - 1
  car
```

运行时的衔接流程是：

1. `crowd_sim.py` 或 `train_robot_ppo.py` 先创建 ProtoMotions runtime，得到 `env` 和 MaskedMimic `agent`。
2. CrowdSim 根据地图为 humanoid 和 car 采样起点，并把 humanoid 起点写入 ProtoMotions env，把 car 起点写入额外的 robot articulation。
3. `navigation.attach(env)` 会 hook `env.reset()` 和 `env.step()`。
4. 每次 `env.step(humanoid_action)` 前后，navigation manager 会按 `navigation.update_hz` 更新 waypoint、SFM local target、collision 和轨迹日志。
5. humanoid action 始终来自 MaskedMimic policy；navigation 只修改 MaskedMimic 的 pelvis future target，不直接给 humanoid 下发速度。
6. car 在 `car.rl_policy: true` 时由 PPO 输出归一化线速度/角速度指令，再经过差速公式转成轮速。
7. SFM 仍然为 humanoid 和日志计算 local target，但不会作为 car RL observation 输入。

reset 也分成两条链路：

```text
humanoid reset
  ProtoMotions env reset_buf / dones
  + CrowdSim 导航 done 主动写 reset_buf
  -> 下一次 env.reset(done_indices) 真正重置 humanoid
  -> CrowdSim hook reset 后重新采样 goal 并重规划路径

car RL reset
  CrowdSim robot_done = reach | collision | timeout
  -> nav_manager.reset_robot_rl_episodes(robot_done)
  -> 直接重置 car root pose / velocity / joint state
  -> 重新采样 goal 并重规划路径
```

因此，humanoid 的 timeout 主要仍来自 motion clip / ProtoMotions env；reach 和 collision 由 CrowdSim 导航层检测后写入 humanoid `reset_buf`。car 的 reach、collision、timeout 则由 CrowdSim 的 robot RL episode 逻辑管理。

默认运行不会调用 `runtime.agent.evaluator.simple_test_policy()`，也不会收集 humanoid evaluation metrics。只有显式传入 `--full-eval` 时，才会调用 ProtoMotions 的 evaluator。

## Humanoid 与 Motion File

CrowdSim 仍然需要加载 `humanoid.motion_file`，原因是 MaskedMimic 本身不是一个只接受目标点的纯导航控制器。

motion file 主要提供：

- humanoid 初始姿态和动作片段。
- motion clip 的时间推进。
- masked future target 中未被导航覆盖的身体目标，例如 head、hands、feet、knees 等。

导航模块主要覆盖 pelvis translation 和 pelvis rotation，使 humanoid 能朝导航目标移动；其他身体部位目标仍来自 motion library，用来保持动作自然性和稳定性。

因此，不能简单地只传入一个 pelvis 目标后完全丢掉 motion file。除非重新训练一个只基于 pelvis waypoint 的策略，否则当前 checkpoint 仍然依赖 motion library 提供的动作先验。

## 小车差速控制

`differential_control.py` 负责把 RL 输出的归一化线速度、角速度转换成差速轮速度。

当前差速控制使用固定机器人参数，例如 Nova Carter：

- Max Linear Speed: `2.0 m/s`
- Max Angular Speed: `3.0 rad/s`
- Wheel Distance: `0.413 m`
- Wheel Radius: `0.14 m`

小车的 RL action 是归一化的 `[linear_velocity, angular_velocity]`。差速控制器先把它映射到真实速度范围，再用差速公式转换为左右轮速度。

`nova_carter` 的 wheel joint 名称固定在 `differential_control.py` 中。当前 CrowdSim 通过 IsaacLab `Articulation.set_joint_velocity_target()` 下发轮速；在这条路径下，直行对应左右 wheel joint 同号。

## 小车 PPO 训练

训练入口：

```bash
python CrowdSim/train_robot_ppo.py \
  --env-config CrowdSim/config/env.yaml \
  --ppo-config CrowdSim/config/ppo.yaml \
  --num-envs 8 \
  --headless
```

如果想带 viewer 调试：

```bash
python CrowdSim/train_robot_ppo.py \
  --env-config CrowdSim/config/env.yaml \
  --ppo-config CrowdSim/config/ppo.yaml \
  --num-envs 4
```

训练脚本和普通仿真一样，只用 `--headless` 控制 viewer。加 `--headless` 就关闭 viewer；不加则打开 viewer。

训练脚本会在内存中强制：

```yaml
navigation:
  enabled: true
  local:
    method: sfm
car:
  rl_policy: true
```

也就是说，训练时 car 的动作来自 PPO；SFM 仍然为 humanoid 和日志计算 local target，但不作为 car RL observation 输入，也不直接下发给小车。

### RL Observation

小车 PPO 观测由 vector observation 和可选 CNN map observation 组成。

当前 vector observation 包含：

- global goal，机器人坐标系下 2 维。
- 当前小车平面速度，机器人坐标系下 2 维。
- 当前 yaw 的 `sin(yaw)` 和 `cos(yaw)`。
- 到 global goal 的距离。
- 当前 robot episode 进度。
- 最近的 `num_neighbors` 个邻居 agent 状态。

邻居 agent 不区分 humanoid 和 car。每个邻居使用 4 维：

```text
[relative_x, relative_y, relative_vx, relative_vy]
```

这些量都在当前小车坐标系下表达，并按 `neighbor_radius` 归一化位置、按 `max_speed` 归一化速度。

如果开启局部地图：

```yaml
rl:
  map_size: 24
  map_extent: 8.0
```

则额外拼接一个机器人坐标系下的 `24 x 24` 局部占据图，作为 CNN 输入。地图中：

- `1` 表示障碍物或地图外区域。
- `0` 表示自由空间。

`map_size` 是局部图的像素尺寸，不是原始地图像素。`map_extent` 是该局部图覆盖的真实世界尺寸，单位 m。默认：

```text
map_size = 24
map_extent = 8.0 m
cell_size = 8.0 / 24 = 0.333 m
```

当前默认维度示例：

```yaml
rl:
  num_neighbors: 4
  map_size: 24
```

则：

```text
vector_obs_dim = 8 + 4 * 4 = 24
map_obs_dim = 24 * 24 = 576
total_obs_dim = 600
```

静态障碍物不再通过 `num_obstacles` 点列表表示，而是完全由局部 occupancy map 表示。动态障碍物，也就是 humanoid 和 car 邻居，仍然通过 `num_neighbors` 的相对位置和相对速度表示。

### RL Reward 和 Done

PPO reward 当前由以下部分组成：

- `time_penalty`：每步时间惩罚。
- `progress_reward_scale * progress`：靠近 action 前缓存的 SFM local target 的进度奖励，用作 path tracking dense reward。
- `goal_reward`：到达终点奖励。
- `collision_penalty`：碰撞惩罚。
- `timeout_penalty`：episode 超时惩罚。

done 条件：

- reach
- collision
- timeout

reset 后会：

1. 重新放置小车。
2. 重新采样 goal。
3. 重新规划 A* path。
4. 清零该小车 PPO episode step 和上一帧距离。

注意：只要 observation 结构变化，旧 PPO checkpoint 通常不能直接加载，需要重新训练。

### 训练日志

训练输出默认按时间戳分 run 保存：

```text
output/crowdsim_robot_ppo/YYYYmmdd_HHMMSS/
```

包括：

- `robot_ppo_latest.pt`
- `robot_ppo_step_*.pt`
- `tb/`
- `config/env.yaml`
- `config/ppo.yaml`

`output/crowdsim_robot_ppo/latest` 会指向最近一次训练 run。

TensorBoard 当前按四组 tag 前缀组织，每个指标单独成图：

- `loss/*`：`policy_loss`、`value_loss`、`entropy`。
- `reward/*`：`total`、`progress`、`terminal`。
- `outcome/*`：`reached`、`collision`、`timeout`。
- `mean/*`：`return`、`length`、`goal_distance`、`progress_target_distance`、`progress`。

其中 `terminal = reward_goal + reward_collision + reward_timeout`。`reached` / `collision` / `timeout` 已经表达了终止事件比例，所以不再把对应的 reward 分量单独画成三张图；`reward_time` 是每步常数，也不单独记录。

查看 TensorBoard：

```bash
tensorboard --logdir output/crowdsim_robot_ppo
```

恢复训练：

```bash
python CrowdSim/train_robot_ppo.py \
  --env-config CrowdSim/config/env.yaml \
  --ppo-config CrowdSim/config/ppo.yaml \
  --resume output/crowdsim_robot_ppo/latest/robot_ppo_latest.pt \
  --headless
```

## 加载训练好的 PPO 策略

在 `env.yaml` 中设置：

```yaml
car:
  rl_policy: true
  policy_checkpoint: output/crowdsim_robot_ppo/latest/robot_ppo_latest.pt
```

然后运行：

```bash
python CrowdSim/crowd_sim.py \
  --env-config CrowdSim/config/env.yaml \
  --ppo-config CrowdSim/config/ppo.yaml \
  --num-envs 4
```

## 记录与可视化

导航记录由 `navigation.recording` 控制：

```yaml
navigation:
  recording:
    enabled: true
    output_dir: output/crowdsim_navigation
```

运行后会生成：

- `paths_*.json`
- `paths_latest.json`
- `trajectory_*.jsonl`
- `trajectory_latest.jsonl`

`paths_*.json` 保存初始路径和目标信息。

`trajectory_*.jsonl` 保存每帧状态，包括：

- agent 平面位置和速度。
- 当前 A* waypoint。
- global goal。
- SFM local target。
- reach 状态。
- collision pairs。
- SFM debug force / velocity。
- humanoid yaw source。

### Matplotlib 动画

生成 2D 动画和小车速度曲线：

```bash
python CrowdSim/tools/animate_navigation.py \
  output/crowdsim_navigation/trajectory_latest.jsonl \
  --output output/crowdsim_navigation/navigation.gif \
  --fps 10 \
  --stride 2
```

如果要输出 MP4，需要系统里有 ffmpeg：

```bash
python CrowdSim/tools/animate_navigation.py \
  output/crowdsim_navigation/trajectory_latest.jsonl \
  --output output/crowdsim_navigation/navigation.mp4 \
  --fps 10
```

动画中会显示：

- occupancy map。
- agent 当前位置。
- global goal。
- A* path。
- 当前 A* waypoint。
- SFM local target。
- agent 运动轨迹。
- 小车速度变化曲线。

### 快速 MP4 渲染

如果 Matplotlib 动画太慢，可以用快速渲染：

```bash
python CrowdSim/tools/render_navigation_fast.py \
  output/crowdsim_navigation/trajectory_latest.jsonl \
  --output output/crowdsim_navigation/trajectory_fast.mp4 \
  --crop-center-pixels 800 \
  --stride 2
```

### 静态路径图

```bash
python CrowdSim/tools/visualize_paths.py \
  output/crowdsim_navigation/paths_latest.json
```

## 传感器与状态记录

小车相机记录：

```yaml
sensors:
  camera:
    enabled: true
    record_fps: 10.0
    record_dir: output/crowdsim_camera
    record_envs: "0"
    auto_record: false
```

viewer 中按 `Y` 开始或停止录制。

humanoid 状态记录：

```yaml
humanoid:
  state_recording:
    enabled: true
    record_fps: 10
    record_dir: output/crowdsim_humanoid_state
    record_envs: "0"
    auto_record: false
    key: H
```

viewer 中按 `H` 开始或停止录制。

## 常见注意点

- 如果出现 `pxr` import 报错，通常是 Isaac Sim / Omniverse 模块在 `SimulationApp` 创建之前被导入了。相关导入必须放在 AppLauncher 或 SimulationApp 初始化之后。
- 如果两张 GPU 都有显存占用，通常是 Isaac Sim、渲染、torch 或 CUDA context 初始化导致，不一定表示 PPO 或 humanoid 策略在多卡训练。
- `motion_file` 不是可选的导航路径文件，而是 MaskedMimic 的动作库。
- `navigation.enabled: false` 时不会走 CrowdSim 的路径规划和导航 reset 逻辑。
- `car.rl_policy: true` 需要 `env.yaml -> car.policy_checkpoint`，除非运行的是 `train_robot_ppo.py`。
- 修改 PPO observation 结构后，旧 checkpoint 大概率不能加载。
- 场景 USD 的坐标系和 occupancy map 的 `origin/resolution` 必须匹配，否则采样点、路径和实体位置会错位。
- 当前 collision 是基于平面距离的简化检测，不是完整 mesh 碰撞检测。
