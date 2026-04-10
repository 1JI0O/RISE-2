import os
import json
import yaml
import torch
import argparse
import numpy as np
import open3d as o3d
import torchvision.transforms as T
import airexo.helpers.urdf_robot as robot_helper

from copy import deepcopy
from easydict import EasyDict as edict
from transforms3d.quaternions import quat2mat

from airexo.helpers.constants import (
    O3D_RENDER_TRANSFORMATION,
    ROBOT_PREDEFINED_TRANSFORMATION,
)

from utils.training import set_seed
from utils.ensemble import EnsembleBuffer
from remote_eval import WebsocketClientPolicy
from eval_debug_agent import EvalDebugAgent
from dataset.data_utils import resize_image, ImageProcessor
from dataset.projector import SingleArmProjector, DualArmProjector

import cv2


default_args = edict(
    {
        "type": "local",
        "calib_rise2": "calib_rise2/",
        "calib_airexo": "calib_airexo/",
        "config": "config/dual_teleop_dino.yaml",
        "ckpt": "logs/collect_toys",
        "host": "127.0.0.1",
        "port": 8000,
        "debug_scene_dir": None,
        "debug_camera_dir": None,
        "debug_frame_id": None,
        "debug_stream": False,
        "debug_loop": False,
        "debug_h5": None,
        "debug_left_suffix": None,
        "debug_right_suffix": None,
    }
)


_arm_renderer = None
_GRIPPER_KEYWORDS = ("finger", "knuckle", "robotiq", "flange")


def _warn(msg):
    print(f"[mask-aware][warn] {msg}")


def _build_mask_aware_cfg(config):
    default_json_mask_cfg = {
        "left_json": None,
        "right_json": None,
        "cam_base_mode": "base_to_cam__predef",
        "use_agent_intrinsics": True,
        "intrinsics_npy": None,
        "intrinsic_selector": "first",
        "left_urdf": os.path.join(
            "airexo", "airexo", "urdf_models", "robot", "left_robot_inhand.urdf"
        ),
        "right_urdf": os.path.join(
            "airexo", "airexo", "urdf_models", "robot", "right_robot_inhand.urdf"
        ),
        "width": 1280,
        "height": 720,
        "near_plane": 0.01,
        "far_plane": 100.0,
    }
    default_cfg = {
        "enabled": False,
        "enable_3d_filter": True,
        "enable_2d_reweight": True,
        "mask_threshold": 0,
        "mask_white_is_untrusted": True,
        "dilate_radius": 10,
        "debug_save_mask": False,
        "debug_save_every": 10,
        "debug_dir": "mask_debug_real",
        "debug_print_stats": True,
        "json_mask": default_json_mask_cfg,
    }

    raw_cfg = getattr(config, "mask_aware", {})
    raw_cfg = dict(raw_cfg) if raw_cfg is not None else {}
    merged_cfg = deepcopy(default_cfg)

    for key, value in raw_cfg.items():
        if value is None or key not in merged_cfg:
            continue
        if key == "json_mask" and isinstance(value, dict):
            json_cfg = deepcopy(default_json_mask_cfg)
            json_cfg.update({k: v for k, v in value.items() if v is not None})
            merged_cfg[key] = json_cfg
        else:
            merged_cfg[key] = value

    merged_cfg["enabled"] = bool(merged_cfg["enabled"])
    merged_cfg["enable_3d_filter"] = bool(merged_cfg["enable_3d_filter"])
    merged_cfg["enable_2d_reweight"] = bool(merged_cfg["enable_2d_reweight"])
    merged_cfg["mask_threshold"] = float(merged_cfg["mask_threshold"])
    merged_cfg["mask_white_is_untrusted"] = bool(merged_cfg["mask_white_is_untrusted"])
    merged_cfg["dilate_radius"] = max(0, int(merged_cfg["dilate_radius"]))
    merged_cfg["debug_save_mask"] = bool(merged_cfg["debug_save_mask"])
    merged_cfg["debug_save_every"] = max(1, int(merged_cfg["debug_save_every"]))
    merged_cfg["debug_dir"] = str(merged_cfg["debug_dir"])
    merged_cfg["debug_print_stats"] = bool(merged_cfg["debug_print_stats"])

    json_cfg = deepcopy(default_json_mask_cfg)
    json_cfg.update(
        {k: v for k, v in dict(merged_cfg["json_mask"]).items() if v is not None}
    )
    json_cfg["left_json"] = (
        None if json_cfg["left_json"] is None else str(json_cfg["left_json"])
    )
    json_cfg["right_json"] = (
        None if json_cfg["right_json"] is None else str(json_cfg["right_json"])
    )
    json_cfg["cam_base_mode"] = str(json_cfg["cam_base_mode"])
    json_cfg["use_agent_intrinsics"] = bool(json_cfg["use_agent_intrinsics"])
    json_cfg["intrinsics_npy"] = (
        None if json_cfg["intrinsics_npy"] is None else str(json_cfg["intrinsics_npy"])
    )
    json_cfg["intrinsic_selector"] = str(json_cfg["intrinsic_selector"])
    json_cfg["left_urdf"] = str(json_cfg["left_urdf"])
    json_cfg["right_urdf"] = str(json_cfg["right_urdf"])
    json_cfg["width"] = int(json_cfg["width"])
    json_cfg["height"] = int(json_cfg["height"])
    json_cfg["near_plane"] = float(json_cfg["near_plane"])
    json_cfg["far_plane"] = float(json_cfg["far_plane"])
    merged_cfg["json_mask"] = edict(json_cfg)
    return edict(merged_cfg)


def _log_mask_aware_summary(mask_cfg):
    print(
        "[mask-aware] enabled={} 3d_filter={} 2d_reweight={} threshold={} dilate_radius={} debug_save_mask={} debug_every={} debug_dir={}".format(
            mask_cfg.enabled,
            mask_cfg.enable_3d_filter,
            mask_cfg.enable_2d_reweight,
            mask_cfg.mask_threshold,
            mask_cfg.dilate_radius,
            mask_cfg.debug_save_mask,
            mask_cfg.debug_save_every,
            mask_cfg.debug_dir,
        )
    )


def _save_mask_debug(colors, mask01, step, mask_cfg):
    out_dir = os.path.abspath(mask_cfg.debug_dir)
    os.makedirs(out_dir, exist_ok=True)

    mask_bool = mask01 > 0.5
    mask_u8 = mask_bool.astype(np.uint8) * 255
    overlay = np.asarray(colors, dtype=np.uint8).copy()
    if np.any(mask_bool):
        overlay[mask_bool] = np.clip(
            0.6 * overlay[mask_bool].astype(np.float32)
            + 0.4 * np.array([255, 0, 0], dtype=np.float32),
            0,
            255,
        ).astype(np.uint8)

    cv2.imwrite(os.path.join(out_dir, f"{int(step):06d}_mask.png"), mask_u8)
    cv2.imwrite(
        os.path.join(out_dir, f"{int(step):06d}_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )

    if mask_cfg.debug_print_stats:
        ys, xs = np.where(mask_bool)
        if xs.size == 0:
            print(f"[mask-debug] step={step} pixels=0 bbox=empty dir={out_dir}")
        else:
            print(
                "[mask-debug] step={} pixels={} bbox=({}, {})-({}, {}) dir={}".format(
                    step,
                    int(mask_bool.sum()),
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()),
                    int(ys.max()),
                    out_dir,
                )
            )


class NoGripperSeparateRobotRenderer:
    def __init__(
        self,
        left_joint_cfgs,
        right_joint_cfgs,
        cam_to_left_base,
        cam_to_right_base,
        intrinsic,
        width=1280,
        height=720,
        near_plane=0.01,
        far_plane=100.0,
        urdf_file=None,
    ):
        if urdf_file is None:
            urdf_file = {
                "left": os.path.join(
                    "airexo", "airexo", "urdf_models", "robot", "left_robot_inhand.urdf"
                ),
                "right": os.path.join(
                    "airexo",
                    "airexo",
                    "urdf_models",
                    "robot",
                    "right_robot_inhand.urdf",
                ),
            }

        self.left_joint_cfgs = left_joint_cfgs
        self.right_joint_cfgs = right_joint_cfgs
        self.cam_to_left_base = np.asarray(cam_to_left_base, dtype=np.float64)
        self.cam_to_right_base = np.asarray(cam_to_right_base, dtype=np.float64)
        self.urdf_file = {
            "left": str(urdf_file["left"]),
            "right": str(urdf_file["right"]),
        }

        self.renderer = o3d.visualization.rendering.OffscreenRenderer(
            int(width), int(height)
        )
        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = "defaultLit"

        cur_transforms_left, self.visuals_map_left = (
            robot_helper.forward_kinematic_single(
                joint=np.zeros((self.left_joint_cfgs.num_joints,), dtype=np.float32),
                joint_cfgs=self.left_joint_cfgs,
                is_rad=True,
                urdf_file=self.urdf_file["left"],
                with_visuals_map=True,
            )
        )
        cur_transforms_right, self.visuals_map_right = (
            robot_helper.forward_kinematic_single(
                joint=np.zeros((self.right_joint_cfgs.num_joints,), dtype=np.float32),
                joint_cfgs=self.right_joint_cfgs,
                is_rad=True,
                urdf_file=self.urdf_file["right"],
                with_visuals_map=True,
            )
        )

        self.model_meshes_left = {}
        self.last_transforms_left = {}
        self.model_meshes_right = {}
        self.last_transforms_right = {}
        self._init_side_meshes(
            "left",
            cur_transforms_left,
            self.visuals_map_left,
            self.cam_to_left_base,
            self.urdf_file["left"],
            self.model_meshes_left,
            self.last_transforms_left,
        )
        self._init_side_meshes(
            "right",
            cur_transforms_right,
            self.visuals_map_right,
            self.cam_to_right_base,
            self.urdf_file["right"],
            self.model_meshes_right,
            self.last_transforms_right,
        )

        self.renderer.scene.camera.set_projection(
            np.asarray(intrinsic, dtype=np.float64),
            float(near_plane),
            float(far_plane),
            float(width),
            float(height),
        )

    @staticmethod
    def _should_keep_link(link_name):
        link_name = str(link_name).lower()
        return not any(keyword in link_name for keyword in _GRIPPER_KEYWORDS)

    def _compose_tf(self, cam_to_base, transform, offset):
        return (
            np.asarray(O3D_RENDER_TRANSFORMATION, dtype=np.float64)
            @ np.asarray(cam_to_base, dtype=np.float64)
            @ np.asarray(ROBOT_PREDEFINED_TRANSFORMATION, dtype=np.float64)
            @ transform.matrix()
            @ offset.matrix()
        )

    def _init_side_meshes(
        self,
        side,
        cur_transforms,
        visuals_map,
        cam_to_base,
        urdf_file,
        mesh_store,
        tf_store,
    ):
        for link, transform in cur_transforms.items():
            if not self._should_keep_link(link):
                continue
            for visual in visuals_map[link]:
                if visual.geom_param is None:
                    continue
                mesh_name = f"{side}///{link}///{visual.geom_param}"
                tf = self._compose_tf(cam_to_base, transform, visual.offset)
                mesh = o3d.io.read_triangle_mesh(
                    os.path.join(os.path.dirname(urdf_file), visual.geom_param)
                )
                mesh.transform(tf)
                mesh.compute_vertex_normals()
                mesh_store[mesh_name] = mesh
                tf_store[mesh_name] = tf
                self.renderer.scene.add_geometry(mesh_name, mesh, self.material)

    def update_joints(self, left_joint, right_joint):
        cur_transforms_left = robot_helper.forward_kinematic_single(
            joint=np.asarray(left_joint, dtype=np.float32),
            joint_cfgs=self.left_joint_cfgs,
            is_rad=True,
            urdf_file=self.urdf_file["left"],
            with_visuals_map=False,
        )
        cur_transforms_right = robot_helper.forward_kinematic_single(
            joint=np.asarray(right_joint, dtype=np.float32),
            joint_cfgs=self.right_joint_cfgs,
            is_rad=True,
            urdf_file=self.urdf_file["right"],
            with_visuals_map=False,
        )

        self.renderer.scene.clear_geometry()

        for link, transform in cur_transforms_left.items():
            if not self._should_keep_link(link):
                continue
            for visual in self.visuals_map_left[link]:
                if visual.geom_param is None:
                    continue
                mesh_name = f"left///{link}///{visual.geom_param}"
                tf = self._compose_tf(self.cam_to_left_base, transform, visual.offset)
                self.model_meshes_left[mesh_name].transform(
                    tf @ np.linalg.inv(self.last_transforms_left[mesh_name])
                )
                self.model_meshes_left[mesh_name].compute_vertex_normals()
                self.last_transforms_left[mesh_name] = tf
                self.renderer.scene.add_geometry(
                    mesh_name, self.model_meshes_left[mesh_name], self.material
                )

        for link, transform in cur_transforms_right.items():
            if not self._should_keep_link(link):
                continue
            for visual in self.visuals_map_right[link]:
                if visual.geom_param is None:
                    continue
                mesh_name = f"right///{link}///{visual.geom_param}"
                tf = self._compose_tf(self.cam_to_right_base, transform, visual.offset)
                self.model_meshes_right[mesh_name].transform(
                    tf @ np.linalg.inv(self.last_transforms_right[mesh_name])
                )
                self.model_meshes_right[mesh_name].compute_vertex_normals()
                self.last_transforms_right[mesh_name] = tf
                self.renderer.scene.add_geometry(
                    mesh_name, self.model_meshes_right[mesh_name], self.material
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


def _invert_T(T):
    T = np.asarray(T, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def _pose7_wxyz_to_mat(pose7):
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    mat = np.eye(4, dtype=np.float64)
    quat = pose7[3:]
    quat = quat / np.linalg.norm(quat)
    mat[:3, :3] = quat2mat(quat)
    mat[:3, 3] = pose7[:3]
    return mat


def _load_json_pose(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _pose7_wxyz_to_mat(data["pose_in_link"])


def _load_intrinsic_from_npy(path, selector="first"):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()
    if selector == "first":
        selector = sorted(data.keys())[0]
    elif selector == "mean":
        mats = np.stack(
            [np.asarray(v, dtype=np.float32) for v in data.values()], axis=0
        )
        return mats.mean(axis=0).astype(np.float32)
    return np.asarray(data[selector], dtype=np.float32)


def _cam_base_from_json(T_json, mode):
    robot_predef_inv = _invert_T(
        np.asarray(ROBOT_PREDEFINED_TRANSFORMATION, dtype=np.float64)
    )
    if mode == "base_to_cam__predef":
        return _invert_T(T_json) @ robot_predef_inv
    if mode == "base_to_cam__raw":
        return _invert_T(T_json)
    if mode == "cam_to_base__predef":
        return T_json @ robot_predef_inv
    if mode == "cam_to_base__raw":
        return T_json
    raise ValueError(f"invalid cam_base_mode: {mode}")


def _load_agent_intrinsic(agent, json_cfg):
    if json_cfg.use_agent_intrinsics:
        if not hasattr(agent, "intrinsics"):
            raise RuntimeError("agent does not provide intrinsics")
        return np.asarray(agent.intrinsics, dtype=np.float32), "agent"
    if json_cfg.intrinsics_npy is None:
        raise RuntimeError("intrinsics_npy is required when use_agent_intrinsics=False")
    return _load_intrinsic_from_npy(
        json_cfg.intrinsics_npy, json_cfg.intrinsic_selector
    ), json_cfg.intrinsics_npy


def _init_mask_renderer(mask_cfg, agent):
    if (not mask_cfg.enabled) or agent is None:
        return None

    json_cfg = mask_cfg.json_mask
    if json_cfg.left_json is None or json_cfg.right_json is None:
        _warn("json mask enabled but left_json/right_json is missing; disable renderer")
        return None

    try:
        left_json = _load_json_pose(json_cfg.left_json)
        right_json = _load_json_pose(json_cfg.right_json)
        left_cam_to_base = _cam_base_from_json(
            left_json, json_cfg.cam_base_mode
        ).astype(np.float32)
        right_cam_to_base = _cam_base_from_json(
            right_json, json_cfg.cam_base_mode
        ).astype(np.float32)
        intrinsic, intrinsic_source = _load_agent_intrinsic(agent, json_cfg)

        left_cfg = edict(
            yaml.safe_load(
                open(
                    os.path.join(
                        "airexo", "airexo", "configs", "joint", "left", "robot.yaml"
                    )
                )
            )
        )
        right_cfg = edict(
            yaml.safe_load(
                open(
                    os.path.join(
                        "airexo", "airexo", "configs", "joint", "right", "robot.yaml"
                    )
                )
            )
        )

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

        renderer = NoGripperSeparateRobotRenderer(
            left_joint_cfgs=left_cfg,
            right_joint_cfgs=right_cfg,
            cam_to_left_base=left_cam_to_base,
            cam_to_right_base=right_cam_to_base,
            intrinsic=intrinsic,
            width=render_width,
            height=render_height,
            near_plane=float(json_cfg.near_plane),
            far_plane=float(json_cfg.far_plane),
            urdf_file={
                "left": str(json_cfg.left_urdf),
                "right": str(json_cfg.right_urdf),
            },
        )
        print(
            "[mask-aware/json] renderer initialized "
            f"(cam_mode={json_cfg.cam_base_mode}, intrinsic={intrinsic_source}, "
            f"left_json={json_cfg.left_json}, right_json={json_cfg.right_json})"
        )
        return renderer
    except Exception as exc:
        _warn(f"renderer init failed: {exc}")
        return None


def _extract_joint_pair(agent, robot_type):
    if agent is None:
        return None

    if (
        robot_type == "dual"
        and hasattr(agent, "left_robot")
        and hasattr(agent, "right_robot")
    ):
        try:
            left_joint_pos = np.asarray(
                agent.left_robot.get_joint_pos(), dtype=np.float32
            ).reshape(-1)
            right_joint_pos = np.asarray(
                agent.right_robot.get_joint_pos(), dtype=np.float32
            ).reshape(-1)
            left_gripper_width = np.asarray(
                agent.left_gripper.get_states()["width"], dtype=np.float32
            ).reshape(-1)
            right_gripper_width = np.asarray(
                agent.right_gripper.get_states()["width"], dtype=np.float32
            ).reshape(-1)

            if left_joint_pos.size < 7 or right_joint_pos.size < 7:
                _warn(
                    "dual-arm joint length mismatch: "
                    f"left={left_joint_pos.size}, right={right_joint_pos.size}"
                )
                return None
            if left_gripper_width.size < 1 or right_gripper_width.size < 1:
                _warn(
                    "dual-arm gripper width length mismatch: "
                    f"left={left_gripper_width.size}, right={right_gripper_width.size}"
                )
                return None

            left_joint = np.concatenate(
                [left_joint_pos[:7], left_gripper_width[:1]], axis=0
            )
            right_joint = np.concatenate(
                [right_joint_pos[:7], right_gripper_width[:1]], axis=0
            )
            return left_joint, right_joint
        except Exception as exc:
            _warn(f"failed to read dual-arm joints from raw agent API: {exc}")
            return None

    if not hasattr(agent, "get_proprio"):
        return None
    try:
        _, proprio_joint = agent.get_proprio(with_joint=True)
    except Exception as exc:
        _warn(f"failed to read joints from agent: {exc}")
        return None

    proprio_joint = np.asarray(proprio_joint, dtype=np.float32).reshape(-1)
    if robot_type == "single":
        if proprio_joint.size < 8:
            return None
        return proprio_joint[:8], None
    if proprio_joint.size < 16:
        return None
    return proprio_joint[:8], proprio_joint[8:16]


def _normalize_mask(mask, depth_shape, mask_cfg):
    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        if mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        elif mask_np.shape[-1] == 1:
            mask_np = mask_np[..., 0]
        else:
            raise ValueError(f"unsupported mask shape: {mask_np.shape}")
    if mask_np.ndim != 2:
        raise ValueError(f"unsupported mask shape: {mask_np.shape}")

    target_h, target_w = int(depth_shape[0]), int(depth_shape[1])
    if mask_np.shape != (target_h, target_w):
        mask_np = cv2.resize(
            mask_np.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

    if mask_cfg.mask_white_is_untrusted:
        mask01 = (mask_np > float(mask_cfg.mask_threshold)).astype(np.float32)
    else:
        mask01 = (mask_np <= float(mask_cfg.mask_threshold)).astype(np.float32)

    if mask_cfg.dilate_radius > 0 and np.any(mask01 > 0):
        ks = 2 * int(mask_cfg.dilate_radius) + 1
        kernel = np.ones((ks, ks), np.uint8)
        mask01 = cv2.dilate(mask01.astype(np.uint8), kernel, iterations=1).astype(
            np.float32
        )
    return mask01


def _infer_mask(agent, depth_shape, mask_cfg, step, robot_type):
    if _arm_renderer is None:
        return None, "renderer_unavailable"
    joints = _extract_joint_pair(agent, robot_type)
    if joints is None:
        return None, "joint_unavailable"

    try:
        if robot_type == "single":
            return None, "single_robot_json_mask_not_supported"
        left_joint, right_joint = joints
        _arm_renderer.update_joints(left_joint, right_joint)
        raw_mask = _arm_renderer.render_mask()
        return _normalize_mask(raw_mask, depth_shape, mask_cfg), None
    except Exception as exc:
        return None, f"render_failed: {exc}"


def _build_image_mask_weight(mask01, image_processor):
    mask_tensor = torch.from_numpy(mask01[np.newaxis].astype(np.float32))
    mask_tensor = resize_image(
        mask_tensor, image_processor.img_size, interpolation=T.InterpolationMode.NEAREST
    )
    mask_ratio = image_processor.image_coord_pooling(mask_tensor)
    image_mask_weight = (1.0 - mask_ratio).clamp(0.0, 1.0).to(torch.float32)
    if not torch.isfinite(image_mask_weight).all():
        raise RuntimeError("image_mask_weight contains non-finite values")
    return image_mask_weight


def create_point_cloud(
    colors, depths, intrinsics, config, depth_scale=1000.0, rescale_factor=1
):
    if rescale_factor != 1:
        H, W = depths.shape
        h, w = int(H * rescale_factor), int(W * rescale_factor)
        colors = colors.transpose([2, 0, 1]).astype(np.float32)
        colors = torch.from_numpy(colors)
        colors = np.ascontiguousarray(
            resize_image(colors, [h, w]).numpy().transpose([1, 2, 0])
        )
        depths = depths.astype(np.float32)
        depths = torch.from_numpy(depths[np.newaxis])
        depths = resize_image(
            depths, [h, w], interpolation=T.InterpolationMode.NEAREST
        )[0]
        depths = depths.numpy()

    h, w = depths.shape
    fx, fy = intrinsics[0, 0] * rescale_factor, intrinsics[1, 1] * rescale_factor
    cx, cy = intrinsics[0, 2] * rescale_factor, intrinsics[1, 2] * rescale_factor
    colors = o3d.geometry.Image(colors.astype(np.uint8))
    depths = o3d.geometry.Image(depths.astype(np.float32))
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors, depths, depth_scale, convert_rgb_to_intensity=False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
    bbox3d = o3d.geometry.AxisAlignedBoundingBox(
        config.deploy.workspace.min, config.deploy.workspace.max
    )
    cloud = cloud.crop(bbox3d)
    cloud = cloud.voxel_down_sample(config.data.voxel_size)
    return cloud


def create_input(
    colors, depths, cam_intrinsics, config, depth_scale=1000.0, rescale_factor=1
):
    cloud = create_point_cloud(
        colors,
        depths,
        cam_intrinsics,
        config,
        depth_scale=depth_scale,
        rescale_factor=rescale_factor,
    )
    points = np.asarray(cloud.points)
    coords = np.ascontiguousarray(points / config.data.voxel_size, dtype=np.int32)
    return coords, points, cloud


def create_batch(coords, points):
    import MinkowskiEngine as ME

    coords_batch, feats_batch = ME.utils.sparse_collate(
        [coords], [points.astype(np.float32)]
    )
    return coords_batch, feats_batch


def process_state(state, config, to_control=True):
    if config.robot_type == "single":
        if to_control:
            state[..., 0:3] = (state[..., 0:3] + 1) / 2.0 * (
                config.data.normalization.trans_max
                - config.data.normalization.trans_min
            ) + config.data.normalization.trans_min
            state[..., 9] = (
                (state[..., 9] + 1) / 2.0 * config.data.normalization.max_gripper_width
            )
        else:
            state[..., 0:3] = (
                state[..., 0:3] - config.data.normalization.trans_min
            ) / (
                config.data.normalization.trans_max
                - config.data.normalization.trans_min
            ) * 2.0 - 1
            state[..., 9] = (
                state[..., 9] / config.data.normalization.max_gripper_width * 2.0 - 1
            )
    else:
        if to_control:
            state[..., 0:3] = (state[..., 0:3] + 1) / 2.0 * (
                config.data.normalization.trans_max
                - config.data.normalization.trans_min
            ) + config.data.normalization.trans_min
            state[..., 10:13] = (state[..., 10:13] + 1) / 2.0 * (
                config.data.normalization.trans_max
                - config.data.normalization.trans_min
            ) + config.data.normalization.trans_min
            state[..., 9] = (
                (state[..., 9] + 1) / 2.0 * config.data.normalization.max_gripper_width
            )
            state[..., 19] = (
                (state[..., 19] + 1) / 2.0 * config.data.normalization.max_gripper_width
            )
        else:
            state[..., 0:3] = (
                state[..., 0:3] - config.data.normalization.trans_min
            ) / (
                config.data.normalization.trans_max
                - config.data.normalization.trans_min
            ) * 2.0 - 1
            state[..., 10:13] = (
                state[..., 10:13] - config.data.normalization.trans_min
            ) / (
                config.data.normalization.trans_max
                - config.data.normalization.trans_min
            ) * 2.0 - 1
            state[..., 9] = (
                state[..., 9] / config.data.normalization.max_gripper_width * 2.0 - 1
            )
            state[..., 19] = (
                state[..., 19] / config.data.normalization.max_gripper_width * 2.0 - 1
            )
    return state


def _load_rise2_intrinsic(calib_path, global_cam_serial):
    calib = np.load(calib_path, allow_pickle=True).item()
    serial = str(global_cam_serial)
    if serial.startswith("cam_"):
        serial = serial[4:]
    if serial in calib["intrinsics"]:
        return np.asarray(calib["intrinsics"][serial], dtype=np.float32)
    if global_cam_serial in calib["intrinsics"]:
        return np.asarray(calib["intrinsics"][global_cam_serial], dtype=np.float32)
    raise KeyError(f"camera serial {global_cam_serial} not found in calib intrinsics")


def _build_agent(args, config):
    if args.debug_scene_dir is not None:
        debug_intrinsics = _load_rise2_intrinsic(
            args.calib_rise2, config.deploy.agent.camera_serial
        )
        return EvalDebugAgent(
            scene_dir=args.debug_scene_dir,
            camera_serial=config.deploy.agent.camera_serial,
            camera_dir=args.debug_camera_dir,
            frame_id=args.debug_frame_id,
            stream=bool(args.debug_stream),
            loop=bool(args.debug_loop),
            intrinsics=debug_intrinsics,
            depth_scale=1000.0,
            robot_type=config.robot_type,
            h5_path=args.debug_h5,
            left_suffix=args.debug_left_suffix,
            right_suffix=args.debug_right_suffix,
            gripper_key=getattr(config.deploy.agent, "gripper_key", "width"),
        )

    from eval_agent import DualArmAgent
    from eval_agent_single_device import SingleArmAgent

    Agent = SingleArmAgent if config.robot_type == "single" else DualArmAgent
    return Agent(**config.deploy.agent)


def evaluate(args_override):
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    with open(args.config, "r") as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
    config.data.normalization.trans_min = np.asarray(
        config.data.normalization.trans_min
    )
    config.data.normalization.trans_max = np.asarray(
        config.data.normalization.trans_max
    )
    config.mask_aware = _build_mask_aware_cfg(config)

    set_seed(config.deploy.seed)

    if args.type == "local":
        from policy import RISE2

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading policy ...")
        policy = RISE2(
            num_action=config.data.num_action,
            obs_feature_dim=config.model.obs_feature_dim,
            cloud_enc_dim=config.model.cloud_enc_dim,
            image_enc_dim=config.model.image_enc_dim,
            action_dim=10 if config.robot_type == "single" else 20,
            hidden_dim=config.model.hidden_dim,
            nheads=config.model.nheads,
            num_attn_layers=config.model.num_attn_layers,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            image_enc=config.model.image_enc,
            interp_fn_mode=config.model.interp_fn_mode,
            image_enc_finetune=config.model.image_enc_finetune,
            image_enc_dtype=config.model.image_enc_dtype,
        ).to(device)
        assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
        policy.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
        print(f"Checkpoint {args.ckpt} loaded.")
        policy.eval()
    else:
        print("Connecting to remote server ...")
        policy = WebsocketClientPolicy(host=args.host, port=args.port)
        device = None

    Projector = (
        SingleArmProjector if config.robot_type == "single" else DualArmProjector
    )
    projector = Projector(args.calib_rise2, config.deploy.agent.camera_serial)

    image_enc = config.model.image_enc
    if image_enc == "resnet18":
        img_size = config.data.aligner.img_size_resnet
        img_coord_size = config.data.aligner.img_coord_size_resnet
    elif image_enc.startswith("dinov2"):
        img_size = config.data.aligner.img_size_dinov2
        img_coord_size = config.data.aligner.img_coord_size_dinov2
    elif image_enc.startswith("dinov3"):
        img_size = config.data.aligner.img_size_dinov3
        img_coord_size = config.data.aligner.img_coord_size_dinov3
    else:
        raise ValueError(f"Unknown image encoder: {image_enc}")

    image_processor = ImageProcessor(
        img_size=img_size,
        img_coord_size=img_coord_size,
        voxel_size=config.data.voxel_size,
        img_mean=config.data.normalization.img_mean,
        img_std=config.data.normalization.img_std,
    )

    agent = _build_agent(args, config)
    _log_mask_aware_summary(config.mask_aware)

    global _arm_renderer
    _arm_renderer = (
        _init_mask_renderer(config.mask_aware, agent) if args.type == "local" else None
    )

    ensemble_buffer = EnsembleBuffer(mode=config.deploy.ensemble_mode)

    rollout_steps = config.deploy.max_steps
    if hasattr(agent, "max_steps"):
        rollout_steps = min(int(rollout_steps), int(agent.max_steps))
    if args.debug_scene_dir is None:
        print("Ready for rollout. Press Enter to continue...")
        input()
    else:
        print(f"Ready for debug rollout: steps={rollout_steps}")

    with torch.inference_mode():
        for t in range(rollout_steps):
            if t % config.deploy.num_inference_steps == 0:
                colors, depths = agent.get_global_observation()

                mask01 = None
                if config.mask_aware.enabled:
                    mask01, mask_reason = _infer_mask(
                        agent, depths.shape[:2], config.mask_aware, t, config.robot_type
                    )
                    if mask01 is None:
                        _warn(
                            f"step={t} mask unavailable, continue without mask: {mask_reason}"
                        )
                    elif config.mask_aware.debug_save_mask and (
                        t % config.mask_aware.debug_save_every == 0
                    ):
                        try:
                            _save_mask_debug(colors, mask01, t, config.mask_aware)
                        except Exception as exc:
                            _warn(f"step={t} failed to save mask debug images: {exc}")

                depths_for_cloud = depths
                if (
                    config.mask_aware.enabled
                    and config.mask_aware.enable_3d_filter
                    and mask01 is not None
                ):
                    depths_for_cloud = depths.copy()
                    depths_for_cloud[mask01 > 0.5] = 0

                coords, points, cloud = create_input(
                    colors,
                    depths_for_cloud,
                    cam_intrinsics=agent.intrinsics,
                    config=config,
                    depth_scale=agent.camera.depth_scale,
                    rescale_factor=1.0,
                )

                if points.size > 0 and (not np.isfinite(points).all()):
                    _warn(
                        f"step={t} cloud contains non-finite values after mask filtering; rebuild from original depth"
                    )
                    coords, points, cloud = create_input(
                        colors,
                        depths,
                        cam_intrinsics=agent.intrinsics,
                        config=config,
                        depth_scale=agent.camera.depth_scale,
                        rescale_factor=1.0,
                    )

                if points.shape[0] == 0:
                    if mask01 is not None and config.mask_aware.enable_3d_filter:
                        _warn(
                            f"step={t} cloud becomes empty after mask filtering; rebuild from original depth"
                        )
                        coords, points, cloud = create_input(
                            colors,
                            depths,
                            cam_intrinsics=agent.intrinsics,
                            config=config,
                            depth_scale=agent.camera.depth_scale,
                            rescale_factor=1.0,
                        )
                    if points.shape[0] == 0:
                        _warn(f"step={t} cloud is empty; skip inference")
                        continue

                image_coords = image_processor.get_image_coordinates(
                    depths, agent.intrinsics, agent.camera.depth_scale
                )
                colors, image_coords = image_processor.preprocess_images(
                    colors, image_coords
                )

                image_mask_weight = None
                if (
                    args.type == "local"
                    and config.mask_aware.enabled
                    and config.mask_aware.enable_2d_reweight
                    and mask01 is not None
                ):
                    try:
                        image_mask_weight = _build_image_mask_weight(
                            mask01, image_processor
                        )
                    except Exception as exc:
                        _warn(
                            f"step={t} failed to build image_mask_weight, continue without it: {exc}"
                        )
                        image_mask_weight = None

                if args.type == "local":
                    import MinkowskiEngine as ME

                    coords_batch, feats_batch = create_batch(coords, points)
                    coords_batch = coords_batch.to(device)
                    feats_batch = feats_batch.to(device)
                    cloud_data = ME.SparseTensor(feats_batch, coords_batch)

                    colors = colors.unsqueeze(0).to(device)
                    image_coords = image_coords.unsqueeze(0).to(device)
                    if image_mask_weight is not None:
                        image_mask_weight = image_mask_weight.unsqueeze(0).to(device)

                    pred_raw_action = (
                        policy(
                            cloud_data,
                            colors,
                            image_coords,
                            image_mask_weight=image_mask_weight,
                            actions=None,
                        )
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )
                else:
                    obs_dict = {
                        "coords": coords,
                        "points": points,
                        "colors": colors.numpy(),
                        "image_coords": image_coords.numpy(),
                    }
                    pred_raw_action = deepcopy(policy.infer(obs_dict)["actions"])

                action = process_state(pred_raw_action, config, to_control=True)

                if config.deploy.vis:
                    tcp_vis_list = []
                    for raw_tcp in action:
                        tcp_vis_list.append(
                            o3d.geometry.TriangleMesh.create_sphere(0.01).translate(
                                raw_tcp[:3]
                            )
                        )
                        if config.robot_type == "dual":
                            tcp_vis_list.append(
                                o3d.geometry.TriangleMesh.create_sphere(0.01).translate(
                                    raw_tcp[10:13]
                                )
                            )
                    o3d.visualization.draw_geometries([cloud, *tcp_vis_list])

                if config.robot_type == "single":
                    action_tcp = projector.project_tcp_to_base_coord(
                        action[..., :9], rotation_rep="rotation_6d"
                    )
                    action = np.concatenate([action_tcp, action[..., 9:10]], axis=-1)
                else:
                    action_left_tcp = projector.project_tcp_to_base_coord(
                        action[..., :9], "left", rotation_rep="rotation_6d"
                    )
                    action_right_tcp = projector.project_tcp_to_base_coord(
                        action[..., 10:19], "right", rotation_rep="rotation_6d"
                    )
                    action = np.concatenate(
                        [
                            action_left_tcp,
                            action[..., 9:10],
                            action_right_tcp,
                            action[..., 19:20],
                        ],
                        axis=-1,
                    )

                ensemble_buffer.add_action(action, t)

            step_action = ensemble_buffer.get_action()
            if step_action is None:
                continue

            print(f"Step {t}: executing action {step_action} ...")
            # agent.action(step_action, rotation_rep="rotation_6d")
            input("Press Enter to execute the action...")

    agent.stop()


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
