# CrowdSim

CrowdSim is a small local project layered on top of ProtoMotions for crowd-style
visualization experiments in IsaacLab.

## Scripts

- `office_four_smpl_kinematic.py`: kinematic replay of multiple SMPL humanoids in
  one shared global Office USD scene.
- `office_masked_mimic.py`: integrated entry point for Office USD loading,
  PNG-map spawn sampling, and MaskedMimic policy inference.
- `masked_mimic_office_global.py`: MaskedMimic inference in one shared global
  Office USD scene. Kept as the older explicit script.
- `smpl_mesh_visualizer.py`: visual-only SMPL mesh overlay that follows the
  ProtoMotions SMPL robot while keeping the physical humanoid unchanged.

## Typical Commands

```bash
python CrowdSim/office_four_smpl_kinematic.py \
  --motion-file /home/pcl/amp/Assets/motion/amass_smpl_test.pt \
  --office-usd /home/pcl/amp/Assets/Office/office.usd
```

```bash
python CrowdSim/office_masked_mimic.py \
  --checkpoint results/smpl_amass/last.ckpt \
  --motion-file /home/pcl/amp/Assets/motion/amass_smpl_test.pt \
  --office-usd /home/pcl/amp/Assets/Office/office.usd \
  --office-map /home/pcl/amp/Assets/Office/office.png \
  --num-envs 4 \
  --human-mesh
```

If `--spawn-xy` is omitted, CrowdSim samples initial humanoid positions from
`/home/pcl/amp/Assets/Office/office.png`. White pixels are free space, dark/gray
pixels are obstacles or unknown area, and the image center is world coordinate
`(0, 0)`. The default map scale assumes the 3999x3999 Isaac Sim occupancy image
covers stage X/Y bounds `[-100, 100]`. Use `--spawn-xy "x,y;..."` only when you
want to force manual positions.

`office_masked_mimic.py` is the full moving-humanoid entry point: it
loads the Office USD, samples spawn positions from the PNG map, loads the
ProtoMotions checkpoint and motion library, then runs the MaskedMimic policy.
`office_scene.py` only contains shared helper functions.

### Humanoid + Navigation Robot

`office_masked_mimic.py` can also inject one optional navigation robot per
humanoid environment. The robot is a separate IsaacLab scene entity named
`crowdsim_robot`; enabled sensors are exposed on the env as
`crowdsim_robot_camera` and `crowdsim_robot_lidar`.

```bash
python CrowdSim/office_masked_mimic.py \
  --checkpoint results/smpl_amass/last.ckpt \
  --motion-file /home/pcl/amp/Assets/motion/amass_smpl_test.pt \
  --office-usd /home/pcl/amp/Assets/Office/office.usd \
  --office-map /home/pcl/amp/Assets/Office/office.png \
  --num-envs 4 \
  --robot-usd jetbot \
  --enable-robot-camera \
  --enable-robot-lidar
```

Use `--robot-spawn-xy "x,y,yaw;..."` to force robot poses, or omit it to sample
robot start positions from the same PNG map with `--robot-radius`,
`--robot-spacing`, and `--robot-spawn-seed`. For a custom USD, replace
`--robot-usd jetbot` with a local USD path or Omniverse URI. If the sensor
mount should be below a specific robot prim, pass it with
`--robot-mount-prim-path base_link`.

Press `Y` in the viewer to start/stop robot camera recording. By default it
records env `0` at 10 fps into `output/crowdsim_camera/<timestamp>/`, saving RGB
PNG files and depth tensors. Use `--robot-camera-record-envs all` to record every
robot camera, or `--auto-record-robot-camera` to start immediately.

For easier viewing inside closed USD scenes, pass `--hide-office-ceiling` to hide
Office prims whose paths contain `ceiling`, `roof`, or `top`. Override the
keywords with `--hide-office-keywords Ceiling Roof`.

### Navigation Loop

Use `--enable-navigation` to treat humanoids and Jetbots as CrowdSim agents.
CrowdSim samples random starts/goals in white map pixels, plans one A* path per
agent, drives Jetbots with ORCA or SFM, leaves humanoids under MaskedMimic
control, and reports pairwise distance collisions.

```bash
python CrowdSim/office_masked_mimic.py \
  --checkpoint data/pretrained_models/masked_mimic/smpl/last.ckpt \
  --motion-file /home/pcl/amp/Assets/amass_motion/amass_smpl_test.pt \
  --office-usd /home/pcl/amp/Assets/Office/office.usd \
  --office-map /home/pcl/amp/Assets/Office/office.png \
  --num-envs 4 \
  --robot-usd jetbot \
  --enable-navigation \
  --nav-local-controller orca
```

With `--num-envs 4 --robot-usd jetbot`, the navigation loop creates 8 logical
agents: 4 humanoids and 4 Jetbots. Switch the local robot controller with
`--nav-local-controller sfm`. Collision checks are distance based and controlled
by `--nav-collision-distance`.

The standard ProtoMotions inference entry also supports the CrowdSim overlay:

```bash
python protomotions/inference_agent.py \
  --checkpoint results/smpl_amass/last.ckpt \
  --simulator isaaclab \
  --num-envs 1 \
  --motion-file /home/pcl/amp/Assets/motion/amass_smpl_test.pt \
  --human-mesh
```
