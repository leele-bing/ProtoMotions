# CrowdSim

CrowdSim is a small local project layered on top of ProtoMotions for crowd-style
visualization experiments in IsaacLab.

## Scripts

- `crowd_sim.py`: config-driven main entry point for USD scene loading, humanoid
  and Jetbot setup, navigation, sensors, and MaskedMimic inference.
- `sim_agent.py`: ProtoMotions checkpoint/config loading plus env and agent
  construction.
- `sim_world.py`: shared USD scene, spawn, scene patching, and robot scene
  helpers.
- `nav_task.py`: map loading, start/goal sampling, and A* path planning.
- `navigation.py`: Jetbot ORCA/SFM local control and runtime collision checks.
- `filter_scene_usd.py`: offline USD scene filter that removes or deactivates
  keyword-matched prims and saves a `*_removed.usd` scene.
- `train_robot_ppo.py`: PPO training entry point for a learned Jetbot local
  navigation controller.
- `robot_ppo.py`: small actor-critic PPO implementation used by
  `train_robot_ppo.py`.
- `visualize_paths.py`: offline PNG renderer for navigation path JSON logs.
- `sensor_stream.py`: robot camera recording.
- `smpl_mesh_visualizer.py`: visual-only SMPL mesh overlay that follows the
  ProtoMotions SMPL robot while keeping the physical humanoid unchanged.

## Typical Commands

```bash
python CrowdSim/crowd_sim.py --config CrowdSim/config/cfg.yaml
```

Scene, agent, sensor, and navigation settings live in
`CrowdSim/config/cfg.yaml`. Frequently changed runtime settings stay as command
line arguments:

```bash
python CrowdSim/crowd_sim.py \
  --num-envs 4 \
  --scene-physics
```

If `humanoid.spawn_xy` is `null`, CrowdSim samples initial humanoid positions
from `scene.scene_map`. White pixels are free space, dark/gray pixels are
obstacles or unknown area, and the image center is world coordinate `(0, 0)`.
The default map scale assumes the 3999x3999 Isaac Sim occupancy image covers
stage X/Y bounds `[-100, 100]`. For a different USD scene, update
`scene.scene_usd`, `scene.scene_map`, `scene.prim_path`, and `scene.resolution` in
`CrowdSim/config/cfg.yaml`.

`crowd_sim.py` loads the configured USD scene, samples or parses spawn positions, loads the
ProtoMotions checkpoint and motion library, then runs the MaskedMimic policy.

### Humanoid + Navigation Robot

`crowd_sim.py` can also inject one optional navigation robot per humanoid
environment. The robot is a separate IsaacLab scene entity named
`crowdsim_robot`; enabled sensors are exposed on the env as
`crowdsim_robot_camera` and `crowdsim_robot_lidar`.

```bash
python CrowdSim/crowd_sim.py --num-envs 4
```

Use `car.spawn_xy: "x,y,yaw;..."` to force car/robot poses, or leave it `null`
to sample starts from the same PNG map. For a custom USD, set `car.usd` in
`cfg.yaml` to `jetbot`, `nova_carter`, a local USD path, or an Omniverse URI. If
the sensor mount should be below a specific car prim, set `car.mount_prim_path`.
For robots whose drive wheels are not the first two joints, set
`car.wheel_joint_indices` and tune `car.wheel_radius`, `car.wheel_base`, and
`car.max_wheel_speed`.

Press `Y` in the viewer to start/stop robot camera recording. By default it
records env `0` at 10 fps into `output/crowdsim_camera/<timestamp>/`, saving RGB
PNG files and depth tensors. Use `--robot-camera-record-envs all` to record every
robot camera, or set `sensors.camera.auto_record: true` to start immediately.

Scene filtering is an offline preprocessing step. `crowd_sim.py` and
`train_robot_ppo.py` do not remove or deactivate scene prims at runtime. To
create a filtered scene from the configured USD and keywords:

```bash
python CrowdSim/filter_scene_usd.py --config CrowdSim/config/cfg.yaml
```

Then point `scene.scene_usd` to the generated `*_removed.usd`.

### Navigation Loop

Set `navigation.enabled: true` to treat humanoids and Jetbots as CrowdSim agents.
CrowdSim samples random starts/goals from the A* traversable white map area,
plans one A* path per agent, drives Jetbots with ORCA or SFM, leaves humanoids
under MaskedMimic control, reports pairwise distance collisions, and shows
navigation debug markers in the viewer. Green spheres are humanoid starts, blue
spheres are car/Jetbot starts, and yellow arrows show current agent velocity
direction. If A* cannot find a path, CrowdSim raises an error instead of using a
straight-line fallback.

```bash
python CrowdSim/crowd_sim.py --num-envs 4 --scene-physics
```

With `--num-envs 4` and `car.usd: jetbot`, the navigation loop creates 8
logical agents: 4 humanoids and 4 Jetbots. Switch the local robot controller
with `navigation.local_controller: sfm`. Collision checks are distance based and
controlled by `navigation.collision_distance`. A* start/goal sampling and path
thinning parameters live under `navigation.path`.

CrowdSim disables ProtoMotions projectile cubes by setting the simulator
projectile pool size to zero during runtime setup.

### Robot PPO Navigation

`train_robot_ppo.py` trains only the Jetbot local controller. Humanoids still run
the loaded MaskedMimic policy, while the navigation manager exposes fixed-size
robot observations, distance/progress rewards, collision penalties, and
robot-only episode resets.

```bash
python CrowdSim/train_robot_ppo.py \
  --config CrowdSim/config/cfg.yaml \
  --num-envs 8 \
  --total-steps 200000 \
  --headless
```

The script forces `navigation.local_controller: rl` in memory and saves
checkpoints to `output/crowdsim_robot_ppo/robot_ppo_latest.pt`. It also writes
CSV metrics to `output/crowdsim_robot_ppo/train_metrics.csv` and TensorBoard
events to `output/crowdsim_robot_ppo/tb/`. Tune the robot observation and reward
terms under `navigation.rl` in `cfg.yaml`.

```bash
tensorboard --logdir output/crowdsim_robot_ppo/tb
```

To run a trained policy through the normal CrowdSim entry point, set:

```yaml
navigation:
  local_controller: rl
  rl:
    policy_checkpoint: output/crowdsim_robot_ppo/robot_ppo_latest.pt
```

Render the latest navigation path log on top of the occupancy map:

```bash
python CrowdSim/visualize_paths.py output/crowdsim_navigation/paths_latest.json
```

The standard ProtoMotions inference entry also supports the CrowdSim overlay:

```bash
python protomotions/inference_agent.py \
  --checkpoint results/smpl_amass/last.ckpt \
  --simulator isaaclab \
  --num-envs 1 \
  --motion-file /home/pcl/amp/Assets/motion/amass_smpl_test.pt \
  --human-mesh
```
