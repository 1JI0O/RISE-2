import argparse
import builtins
import contextlib
import os

import numpy as np
import open3d as o3d
import yaml
import airexo.helpers.urdf_robot as robot_helper

import eval_maskaware_json_260314 as base
from easydict import EasyDict as edict
from eval_debug_agent import EvalDebugAgent
from airexo.helpers.constants import (
    O3D_RENDER_TRANSFORMATION,
    ROBOT_PREDEFINED_TRANSFORMATION,
)
from utils.transformation import xyz_rot_transform


_GRIPPER_KEYWORDS = ("finger", "knuckle", "robotiq", "flange")


class SingleArmNoGripperRenderer:
    def __init__(
        self,
        joint_cfgs,
        cam_to_base,
        intrinsic,
        width=1280,
        height=720,
        near_plane=0.01,
        far_plane=100.0,
        urdf_file=None,
    ):
        if urdf_file is None:
            urdf_file = os.path.join(
                "airexo", "airexo", "urdf_models", "robot_old", "left_robot.urdf"
            )

        self.joint_cfgs = joint_cfgs
        self.cam_to_base = np.asarray(cam_to_base, dtype=np.float64)
        self.urdf_file = str(urdf_file)

        self.renderer = o3d.visualization.rendering.OffscreenRenderer(
            int(width), int(height)
        )
        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = "defaultLit"

        cur_transforms, self.visuals_map = robot_helper.forward_kinematic_single(
            joint=np.zeros((self.joint_cfgs.num_joints,), dtype=np.float32),
            joint_cfgs=self.joint_cfgs,
            is_rad=True,
            urdf_file=self.urdf_file,
            with_visuals_map=True,
        )

        self.model_meshes = {}
        self.last_transforms = {}
        for link, transform in cur_transforms.items():
            if not self._should_keep_link(link):
                continue
            for visual in self.visuals_map[link]:
                if visual.geom_param is None:
                    continue
                mesh_name = f"single///{link}///{visual.geom_param}"
                tf = self._compose_tf(self.cam_to_base, transform, visual.offset)
                mesh = o3d.io.read_triangle_mesh(
                    os.path.join(os.path.dirname(self.urdf_file), visual.geom_param)
                )
                mesh.transform(tf)
                mesh.compute_vertex_normals()
                self.model_meshes[mesh_name] = mesh
                self.last_transforms[mesh_name] = tf
                self.renderer.scene.add_geometry(mesh_name, mesh, self.material)

        self.renderer.scene.camera.set_projection(
            np.asarray(intrinsic, dtype=np.float64),
            float(near_plane),
            float(far_plane),
            float(width),
            float(height),
        )

    @staticmethod
    def _should_keep_link(link_name):
        return not any(
            keyword in str(link_name).lower() for keyword in _GRIPPER_KEYWORDS
        )

    @staticmethod
    def _compose_tf(cam_to_base, transform, offset):
        return (
            np.asarray(O3D_RENDER_TRANSFORMATION, dtype=np.float64)
            @ np.asarray(cam_to_base, dtype=np.float64)
            @ np.asarray(ROBOT_PREDEFINED_TRANSFORMATION, dtype=np.float64)
            @ transform.matrix()
            @ offset.matrix()
        )

    def update_joints(self, joint):
        cur_transforms = robot_helper.forward_kinematic_single(
            joint=np.asarray(joint, dtype=np.float32),
            joint_cfgs=self.joint_cfgs,
            is_rad=True,
            urdf_file=self.urdf_file,
            with_visuals_map=False,
        )

        self.renderer.scene.clear_geometry()
        for link, transform in cur_transforms.items():
            if not self._should_keep_link(link):
                continue
            for visual in self.visuals_map[link]:
                if visual.geom_param is None:
                    continue
                mesh_name = f"single///{link}///{visual.geom_param}"
                tf = self._compose_tf(self.cam_to_base, transform, visual.offset)
                self.model_meshes[mesh_name].transform(
                    tf @ np.linalg.inv(self.last_transforms[mesh_name])
                )
                self.model_meshes[mesh_name].compute_vertex_normals()
                self.last_transforms[mesh_name] = tf
                self.renderer.scene.add_geometry(
                    mesh_name, self.model_meshes[mesh_name], self.material
                )

    def render_depth(self):
        return np.asarray(
            self.renderer.render_to_depth_image(z_in_view_space=True), dtype=np.float32
        )

    def render_mask(self):
        depth = self.render_depth()
        mask = np.zeros(depth.shape, dtype=np.uint8)
        mask[np.isfinite(depth)] = 255
        return mask


class EvalDebugAgentSingle(EvalDebugAgent):
    def _load_joint_from_h5(self, frame_id):
        self._ensure_h5_open()
        idx = int(np.argmin(np.abs(self._h5_ts - int(frame_id))))
        suffix = self.left_suffix if self.left_suffix is not None else self.right_suffix
        if suffix is None:
            suffix = sorted(
                key.split("joint_position_rad_")[-1]
                for key in self._h5_cache.keys()
                if key.startswith("joint_position_rad_")
            )[0]
        joint7 = np.asarray(
            self._h5_cache[f"joint_position_rad_{suffix}"][idx], dtype=np.float32
        )
        gripper = float(
            np.asarray(self._h5_cache[f"ee_state_{suffix}"][idx]).reshape(-1)[0]
        )
        return np.concatenate([joint7, np.array([gripper], dtype=np.float32)], axis=0)

    def get_proprio(
        self, rotation_rep="rotation_6d", rotation_rep_convention=None, with_joint=False
    ):
        if self._last_frame_id is None:
            self._last_frame_id = self._select_frame_id()

        lowdim = self._load_lowdim(self._last_frame_id)
        if self.robot_type == "single":
            tcp = np.asarray(lowdim["robot"], dtype=np.float32).reshape(-1)[:7]
            gripper = self._get_gripper_value(lowdim["gripper"])
            tcp = xyz_rot_transform(
                tcp,
                from_rep="quaternion",
                to_rep=rotation_rep,
                to_convention=rotation_rep_convention,
            )
            proprio = np.concatenate([tcp, [gripper]], axis=0)
            if not with_joint:
                return proprio
            proprio_joint = self._load_joint_from_h5(self._last_frame_id).astype(
                np.float32
            )
            return proprio, proprio_joint

        return super().get_proprio(
            rotation_rep=rotation_rep,
            rotation_rep_convention=rotation_rep_convention,
            with_joint=with_joint,
        )


def _is_single_agent(agent):
    if agent is None:
        return False
    robot_type = getattr(agent, "robot_type", None)
    if robot_type is not None:
        return str(robot_type) == "single"
    return hasattr(agent, "robot") and not hasattr(agent, "left_robot")


def _patched_init_mask_renderer(mask_cfg, agent):
    if (not mask_cfg.enabled) or agent is None:
        return None

    if not _is_single_agent(agent):
        return _ORIG_INIT_MASK_RENDERER(mask_cfg, agent)

    json_cfg = mask_cfg.json_mask
    single_json = getattr(json_cfg, "single_json", None)
    if single_json is None:
        single_json = getattr(json_cfg, "left_json", None)
    single_urdf = getattr(json_cfg, "single_urdf", None)
    if single_urdf is None:
        single_urdf = getattr(json_cfg, "left_urdf", None)

    if single_json is None or single_urdf is None:
        base._warn(
            "json mask enabled for single arm but single_json/single_urdf is missing"
        )
        return None

    try:
        single_json_pose = base._load_json_pose(single_json)
        cam_to_base = base._cam_base_from_json(
            single_json_pose, json_cfg.cam_base_mode
        ).astype(np.float32)
        intrinsic, intrinsic_source = base._load_agent_intrinsic(agent, json_cfg)

        with open(
            os.path.join("airexo", "airexo", "configs", "joint", "left", "robot.yaml"),
            "r",
            encoding="utf-8",
        ) as f:
            joint_cfg = edict(yaml.safe_load(f))

        render_width = int(json_cfg.width)
        render_height = int(json_cfg.height)
        if intrinsic.shape[0] >= 2 and intrinsic.shape[1] >= 3:
            cx = float(intrinsic[0, 2])
            cy = float(intrinsic[1, 2])
            inferred_width = int(round(cx * 2.0))
            inferred_height = int(round(cy * 2.0))
            if inferred_width > 0 and inferred_height > 0:
                render_width = inferred_width
                render_height = inferred_height

        renderer = SingleArmNoGripperRenderer(
            joint_cfgs=joint_cfg,
            cam_to_base=cam_to_base,
            intrinsic=intrinsic,
            width=render_width,
            height=render_height,
            near_plane=float(json_cfg.near_plane),
            far_plane=float(json_cfg.far_plane),
            urdf_file=str(single_urdf),
        )
        print(
            "[mask-aware/json-single] renderer initialized "
            f"(cam_mode={json_cfg.cam_base_mode}, intrinsic={intrinsic_source}, "
            f"single_json={single_json}, single_urdf={single_urdf})"
        )
        return renderer
    except Exception as exc:
        base._warn(f"single-arm renderer init failed: {exc}")
        return None


def _patched_infer_mask(agent, depth_shape, mask_cfg, step, robot_type):
    if robot_type != "single":
        return _ORIG_INFER_MASK(agent, depth_shape, mask_cfg, step, robot_type)

    if base._arm_renderer is None:
        return None, "renderer_unavailable"
    joints = base._extract_joint_pair(agent, robot_type)
    if joints is None:
        return None, "joint_unavailable"

    try:
        single_joint, _ = joints
        base._arm_renderer.update_joints(single_joint)
        raw_mask = base._arm_renderer.render_mask()
        return base._normalize_mask(raw_mask, depth_shape, mask_cfg), None
    except Exception as exc:
        return None, f"single_render_failed: {exc}"


def _patched_extract_joint_pair(agent, robot_type):
    if robot_type != "single":
        return _ORIG_EXTRACT_JOINT_PAIR(agent, robot_type)

    if agent is None:
        return None

    if hasattr(agent, "robot") and hasattr(agent, "gripper"):
        try:
            joint_pos = np.asarray(
                agent.robot.get_joint_pos(), dtype=np.float32
            ).reshape(-1)
            gripper_state = agent.gripper.get_states()
            gripper_width = np.asarray(
                gripper_state["width"], dtype=np.float32
            ).reshape(-1)
            if joint_pos.size < 7:
                base._warn(f"single-arm joint length mismatch: joint={joint_pos.size}")
                return None
            if gripper_width.size < 1:
                base._warn("single-arm gripper width length mismatch: width=0")
                return None
            single_joint = np.concatenate([joint_pos[:7], gripper_width[:1]], axis=0)
            return single_joint, None
        except Exception as exc:
            base._warn(f"failed to read single-arm joints from raw agent API: {exc}")

    return _ORIG_EXTRACT_JOINT_PAIR(agent, robot_type)


@contextlib.contextmanager
def _auto_input_when_debug(debug_enabled):
    if not debug_enabled:
        yield
        return

    orig_input = builtins.input
    builtins.input = lambda *args, **kwargs: ""
    try:
        yield
    finally:
        builtins.input = orig_input


_ORIG_INIT_MASK_RENDERER = base._init_mask_renderer
_ORIG_INFER_MASK = base._infer_mask
_ORIG_EXTRACT_JOINT_PAIR = base._extract_joint_pair
base._init_mask_renderer = _patched_init_mask_renderer
base._infer_mask = _patched_infer_mask
base._extract_joint_pair = _patched_extract_joint_pair
base.EvalDebugAgent = EvalDebugAgentSingle


def evaluate(args_override):
    debug_enabled = args_override.get("debug_scene_dir") is not None
    with _auto_input_when_debug(debug_enabled):
        return base.evaluate(args_override)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type", action="store", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument(
        "--calib_airexo", action="store", type=str, required=False, default=None
    )
    parser.add_argument("--calib_rise2", action="store", type=str, required=True)
    parser.add_argument("--config", action="store", type=str, required=True)
    parser.add_argument(
        "--ckpt", action="store", type=str, required=False, default=None
    )
    parser.add_argument(
        "--host", action="store", type=str, required=False, default="127.0.0.1"
    )
    parser.add_argument(
        "--port", action="store", type=int, required=False, default=8000
    )
    parser.add_argument(
        "--debug-scene-dir", action="store", type=str, required=False, default=None
    )
    parser.add_argument(
        "--debug-camera-dir", action="store", type=str, required=False, default=None
    )
    parser.add_argument(
        "--debug-frame-id", action="store", type=int, required=False, default=None
    )
    parser.add_argument("--debug-stream", action="store_true")
    parser.add_argument("--debug-loop", action="store_true")
    parser.add_argument(
        "--debug-h5", action="store", type=str, required=False, default=None
    )
    parser.add_argument(
        "--debug-left-suffix", action="store", type=str, required=False, default=None
    )
    parser.add_argument(
        "--debug-right-suffix", action="store", type=str, required=False, default=None
    )
    evaluate(vars(parser.parse_args()))
