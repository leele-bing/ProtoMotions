"""Visual-only SMPL mesh overlay for ProtoMotions IsaacLab humanoids."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from protomotions.utils.rotations import (
    exp_map_to_quat,
    matrix_to_quaternion,
    quat_to_exp_map,
    quaternion_to_matrix,
)


SMPL_JOINT_NAMES: list[str] = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]

MJCF_JOINT_NAMES: list[str] = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
]


@dataclass
class HumanMeshConfig:
    model_dir: str
    num_envs: int
    device: str
    prim_root: str = "/World/CrowdSim/HumanMesh"
    color: tuple[float, float, float] = (0.0, 0.8, 1.0)
    opacity: float = 1.0
    hide_humanoid: bool = False


class ProtoMotionsHumanMeshAdapter:
    """Adapter between ProtoMotions IsaacLab state and the SMPL mesh overlay."""

    def __init__(self, simulator, visualizer: "SMPLMeshVisualizer") -> None:
        self.simulator = simulator
        self.visualizer = visualizer
        self.pose_mapper = SMPLRobotPoseMapper()

    @classmethod
    def from_simulator(cls, simulator) -> "ProtoMotionsHumanMeshAdapter":
        project_root = Path(__file__).resolve().parents[1]
        model_dir = os.environ.get(
            "CROWDSIM_SMPL_MODEL_DIR", str(project_root / "data" / "smpl")
        )
        cfg = HumanMeshConfig(
            model_dir=model_dir,
            num_envs=int(getattr(simulator, "num_envs", 1)),
            device=str(getattr(simulator, "device", "cuda:0")),
            hide_humanoid=os.environ.get("CROWDSIM_HIDE_HUMANOID", "0") == "1",
        )
        return cls(simulator=simulator, visualizer=SMPLMeshVisualizer.from_config(cfg))

    def create(self) -> None:
        self.visualizer.create()
        if self.visualizer.cfg.hide_humanoid:
            self._hide_humanoid_visuals()

    def update(self) -> None:
        robot = self.simulator._robot
        body_pose = self.pose_mapper(robot.data.joint_pos, list(robot.data.joint_names))
        self.visualizer.update(
            body_pose=body_pose,
            root_pos=robot.data.root_pos_w,
            root_quat_wxyz=robot.data.root_quat_w,
        )

    def _hide_humanoid_visuals(self) -> None:
        from pxr import UsdGeom

        stage = self.visualizer.stage
        if stage is None:
            return

        for env_id in range(self.visualizer.cfg.num_envs):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot")
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()


class SMPLRobotPoseMapper:
    """Convert ProtoMotions SMPL robot exp-map DOFs into SMPL body_pose."""

    def __init__(self) -> None:
        self.smpl_to_robot = smpl_to_robot_matrix()
        self.body_names = [name for name in SMPL_JOINT_NAMES if name != "Pelvis"]

    def __call__(self, joint_pos: torch.Tensor, joint_names: Sequence[str]) -> torch.Tensor:
        if joint_pos.ndim != 2:
            raise ValueError(f"joint_pos must be [N, D], got {tuple(joint_pos.shape)}")

        by_name = self._map_by_name(joint_pos, joint_names)
        if by_name is not None:
            return by_name
        return self._map_by_mjcf_order(joint_pos)

    def _map_by_name(
        self, joint_pos: torch.Tensor, joint_names: Sequence[str]
    ) -> Optional[torch.Tensor]:
        rotvecs: list[torch.Tensor] = []
        for body_name in self.body_names:
            indices = self._body_triplet_indices(body_name, joint_names)
            if indices is None:
                return None
            robot_expmap = joint_pos[:, indices]
            rotvecs.append(robot_expmap_to_smpl_rotvec(robot_expmap, self.smpl_to_robot))
        return torch.cat(rotvecs, dim=-1).contiguous()

    def _map_by_mjcf_order(self, joint_pos: torch.Tensor) -> torch.Tensor:
        expected_dofs = 3 * (len(MJCF_JOINT_NAMES) - 1)
        if joint_pos.shape[1] < expected_dofs:
            raise RuntimeError(
                f"Expected at least {expected_dofs} SMPL robot DOFs, got {joint_pos.shape[1]}"
            )

        body_to_rotvec: dict[str, torch.Tensor] = {}
        cursor = 0
        for body_name in MJCF_JOINT_NAMES:
            if body_name == "Pelvis":
                continue
            robot_expmap = joint_pos[:, cursor : cursor + 3]
            body_to_rotvec[body_name] = robot_expmap_to_smpl_rotvec(
                robot_expmap, self.smpl_to_robot
            )
            cursor += 3

        return torch.cat([body_to_rotvec[name] for name in self.body_names], dim=-1)

    @staticmethod
    def _body_triplet_indices(
        body_name: str, joint_names: Sequence[str]
    ) -> Optional[list[int]]:
        axis_to_index: dict[str, int] = {}
        prefix = f"{body_name}_"
        for index, joint_name in enumerate(joint_names):
            if joint_name.startswith(prefix):
                axis_to_index[joint_name[len(prefix) :].lower()] = index

        if set(axis_to_index) != {"x", "y", "z"}:
            return None
        return [axis_to_index["x"], axis_to_index["y"], axis_to_index["z"]]


class SMPLMeshVisualizer:
    """Create and update batched UsdGeom.Mesh prims from SMPL vertices."""

    def __init__(self, cfg: HumanMeshConfig, smpl_model, faces: np.ndarray) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.smpl_model = smpl_model.to(self.device).eval()
        self.faces = np.asarray(faces, dtype=np.int64)
        self.stage = None
        self._points_attrs = []
        self._created = False

    @classmethod
    def from_config(cls, cfg: HumanMeshConfig) -> "SMPLMeshVisualizer":
        install_smpl_pickle_compat()
        try:
            import smplx
        except ImportError as exc:
            raise ImportError("Install smplx and place SMPL_*.pkl files in data/smpl.") from exc

        smpl_model = smplx.create(
            resolve_smpl_model_path(cfg.model_dir),
            model_type="smpl",
            gender="neutral",
            num_betas=10,
            use_pca=False,
            batch_size=cfg.num_envs,
        )
        return cls(cfg, smpl_model, np.asarray(smpl_model.faces, dtype=np.int64))

    def create(self) -> None:
        if self._created:
            return

        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

        self.stage = omni.usd.get_context().get_stage()
        if self.stage is None:
            raise RuntimeError("Could not get active USD stage.")

        UsdGeom.Xform.Define(self.stage, self.cfg.prim_root)
        material = UsdShade.Material.Define(self.stage, f"{self.cfg.prim_root}/Material")
        shader = UsdShade.Shader.Define(self.stage, f"{self.cfg.prim_root}/Material/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*self.cfg.color)
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(self.cfg.opacity))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        num_vertices = int(getattr(self.smpl_model, "v_template").shape[0])
        zero_points = np.zeros((num_vertices, 3), dtype=np.float32)
        face_vertex_counts = [3] * int(self.faces.shape[0])
        face_vertex_indices = self.faces.reshape(-1).astype(np.int64).tolist()

        for env_id in range(self.cfg.num_envs):
            mesh_path = f"{self.cfg.prim_root}/env_{env_id}/Body"
            UsdGeom.Xform.Define(self.stage, f"{self.cfg.prim_root}/env_{env_id}")
            mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
            mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
            mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*self.cfg.color)]))
            mesh.CreateDisplayOpacityAttr([float(self.cfg.opacity)])
            self._points_attrs.append(mesh.CreatePointsAttr(to_vt_vec3f(zero_points)))
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)

        self._created = True

    @torch.no_grad()
    def update(
        self,
        body_pose: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat_wxyz: torch.Tensor,
    ) -> None:
        if not self._created:
            self.create()

        body_pose = body_pose.to(self.device)
        root_pos = root_pos.to(self.device)
        root_quat_wxyz = root_quat_wxyz.to(self.device)
        zeros = torch.zeros((self.cfg.num_envs, 3), device=self.device, dtype=body_pose.dtype)
        betas = torch.zeros((self.cfg.num_envs, 10), device=self.device, dtype=body_pose.dtype)

        out = self.smpl_model(
            betas=betas,
            global_orient=zeros,
            body_pose=body_pose,
            transl=zeros,
            return_verts=True,
        )
        vertices = out.vertices - out.joints[:, 0:1, :]
        vertices = torch.matmul(vertices, smpl_to_robot_matrix(self.device, vertices.dtype).T)
        vertices = quat_apply_wxyz(root_quat_wxyz[:, None, :], vertices) + root_pos[:, None, :]
        vertices = vertices.detach().cpu().numpy().astype(np.float32)

        for env_id, points_attr in enumerate(self._points_attrs):
            points_attr.Set(to_vt_vec3f(vertices[env_id]))


def smpl_to_robot_matrix(
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fixed SMPL local frame -> ProtoMotions SMPL robot local frame transform."""
    rx90 = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    yaw90 = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return yaw90 @ rx90


def robot_expmap_to_smpl_rotvec(
    robot_expmap: torch.Tensor,
    smpl_to_robot: torch.Tensor,
) -> torch.Tensor:
    robot_quat = exp_map_to_quat(robot_expmap, w_last=False)
    robot_rot = quaternion_to_matrix(robot_quat, w_last=False)
    transform = smpl_to_robot.to(device=robot_expmap.device, dtype=robot_expmap.dtype)
    smpl_rot = transform.T @ robot_rot @ transform
    smpl_quat = matrix_to_quaternion(smpl_rot, w_last=False)
    smpl_quat = smpl_quat / torch.clamp(smpl_quat.norm(dim=-1, keepdim=True), min=1e-8)
    return quat_to_exp_map(smpl_quat, w_last=False)


def quat_apply_wxyz(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat = quat_wxyz / torch.clamp(quat_wxyz.norm(dim=-1, keepdim=True), min=1e-8)
    q_vec = quat[..., 1:4]
    q_w = quat[..., 0:1]
    t = 2.0 * torch.cross(q_vec.expand_as(vec), vec, dim=-1)
    return vec + q_w * t + torch.cross(q_vec.expand_as(vec), t, dim=-1)


def resolve_smpl_model_path(model_dir: str) -> str:
    path = Path(model_dir).expanduser()
    if path.is_file():
        return str(path)

    flat_file = path / "SMPL_NEUTRAL.pkl"
    if flat_file.exists():
        return str(flat_file)
    return str(path)


def install_smpl_pickle_compat() -> None:
    """Patch Python 3.11 / NumPy aliases used by older SMPL pickle files."""
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec

    for name, value in {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def to_vt_vec3f(points: np.ndarray):
    from pxr import Gf, Vt

    points = np.asarray(points, dtype=np.float32)
    try:
        return Vt.Vec3fArray.FromNumpy(points)
    except Exception:
        return Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
