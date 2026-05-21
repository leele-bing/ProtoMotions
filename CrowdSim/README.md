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

The standard ProtoMotions inference entry also supports the CrowdSim overlay:

```bash
python protomotions/inference_agent.py \
  --checkpoint results/smpl_amass/last.ckpt \
  --simulator isaaclab \
  --num-envs 1 \
  --motion-file /home/pcl/amp/Assets/motion/amass_smpl_test.pt \
  --human-mesh
```
