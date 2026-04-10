#!/usr/bin/env python3
"""
基于左右标定 JSON 渲染 URDF mask，使用合并模型 robot.urdf。

设计说明
--------
模仿 airexo/helpers/renderer.py 的 RobotRenderer，
链条与 CalibrationInfo.get_camera_to_base(real_base=False) 完全对齐。

标定矩阵链条
------------
JSON 的 pose_in_link = T_cam_X_individual_real_base（per-arm）。

  T_from_left  = T_cam_left_real  @ ROBOT_LEFT_REAL_BASE_TO_REAL_BASE   # +135mm→overall
  T_from_right = T_cam_right_real @ ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE  # -135mm→overall
  T_cam_overall_real = average(T_from_left, T_from_right)
  cam_to_base = T_cam_overall_real @ inv(ROBOT_PREDEFINED)

渲染链条（与 RobotRenderer 完全相同）：
  O3D_RENDER @ cam_to_base @ ROBOT_PREDEFINED @ FK_combined @ offset
= O3D_RENDER @ T_cam_overall_real @ FK_combined @ offset

robot.urdf 已将 ±45° 倾斜和 ±135mm 偏移编码在 base2left / base2right 固定关节中。

使用示例
--------
  from render_mask_from_json import JsonRobotRenderer
  import numpy as np

  renderer = JsonRobotRenderer()
  left_joint  = np.array([j1..j7, gripper_width], dtype=np.float32)  # 8-dim, rad + m
  right_joint = np.array([j1..j7, gripper_width], dtype=np.float32)
  renderer.update_joints(left_joint, right_joint)
  mask  = renderer.render_mask()    # uint8 (H, W)
  image = renderer.render_image()   # uint8 (H, W, 3)
  depth = renderer.render_depth()   # float32 (H, W)

CLI：
  python render_mask_from_json.py --output mask.png --save-image render.png \\
      --debug-view debug.png
"""

from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import open3d as o3d
import kinpy as kp
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# 硬编码标定数据
# ---------------------------------------------------------------------------

# 相机内参（serial 104122061850）
HARDCODED_INTRINSIC = np.array([
    [908.4968,   0.0,     640.631 ],
    [  0.0,    907.65515, 351.3301],
    [  0.0,      0.0,       1.0   ],
], dtype=np.float64)

# 默认关节角（来自 airexo configs/tests/renderer/robot_sep.yaml）
# 格式：[joint1..joint7 (rad), gripper_width (m)]
DEFAULT_LEFT_JOINTS = np.array(
    [1.078, -2.201, 1.628, -1.162, -0.958, 0.503, 0.875, 0.05], dtype=np.float32
)
DEFAULT_RIGHT_JOINTS = np.array(
    [1.397, -2.226, 1.835, -1.367, -0.647, 0.781, 0.972, 0.05], dtype=np.float32
)

# 左臂 pose_in_link: [x, y, z, qw, qx, qy, qz]
_LEFT_POSE_IN_LINK = [
    0.0860694267750039, -0.09938092775476133, 0.34225723639609734,
    0.22666579355367836, -0.6846736943308571, 0.6614734633385874, -0.20566333563100633,
]
# 右臂 pose_in_link: [x, y, z, qw, qx, qy, qz]
_RIGHT_POSE_IN_LINK = [
    0.07783932332093665, 0.2078814260418823, 0.34723683952957585,
    0.2273133855008057, -0.6785647482083789, 0.6637673415778982, -0.21746591345696367,
]


# ---------------------------------------------------------------------------
# 变换辅助
# ---------------------------------------------------------------------------

def _quat_wxyz_to_mat3(quat_wxyz) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def _pose_wxyz_to_mat(pose) -> np.ndarray:
    """[x,y,z, qw,qx,qy,qz] → 4×4 float64"""
    arr = np.asarray(pose, dtype=np.float64)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _quat_wxyz_to_mat3(arr[3:])
    mat[:3,  3] = arr[:3]
    return mat


def _average_transforms(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """平均两个 4×4 变换：平移取均值，旋转取四元数均值后归一化。"""
    t = (T1[:3, 3] + T2[:3, 3]) / 2.0
    q1 = Rotation.from_matrix(T1[:3, :3]).as_quat()   # [x,y,z,w]
    q2 = Rotation.from_matrix(T2[:3, :3]).as_quat()
    # 确保两个四元数在同一半球
    if np.dot(q1, q2) < 0:
        q2 = -q2
    q_avg = (q1 + q2) / 2.0
    q_avg /= np.linalg.norm(q_avg)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat(q_avg).as_matrix()
    T[:3,  3] = t
    return T


# ---------------------------------------------------------------------------
# 预定义变换常量（与 airexo/helpers/constants.py 完全相同）
# ---------------------------------------------------------------------------

ROBOT_PREDEFINED_TRANSFORMATION = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)

O3D_RENDER_TRANSFORMATION = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1],
], dtype=np.float64)

# ±135mm Y 偏移：单臂 individual real base → 整体 real base
_REAL_BASE_OFFSET = 0.135
ROBOT_LEFT_REAL_BASE_TO_REAL_BASE = np.array([
    [1, 0, 0,  0               ],
    [0, 1, 0, -_REAL_BASE_OFFSET],
    [0, 0, 1,  0               ],
    [0, 0, 0,  1               ],
], dtype=np.float64)
ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE = np.array([
    [1, 0, 0,  0               ],
    [0, 1, 0, +_REAL_BASE_OFFSET],
    [0, 0, 1,  0               ],
    [0, 0, 0,  1               ],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# 关节状态转换（复用 airexo/helpers/urdf_robot.py 逻辑）
# ---------------------------------------------------------------------------

def _calc_robotiq_state(open_length: float) -> float:
    return 0.725 - math.asin(float(np.clip(open_length, 0.0, 0.1143)) / 0.1143)


def _gripper_joint_states(gripper_width: float) -> Dict[str, float]:
    w = float(np.clip(gripper_width, 0, 0.085))
    s = _calc_robotiq_state(w)
    return {
        "finger_joint":              s,
        "left_outer_finger_joint":   0.0,
        "left_inner_knuckle_joint":  s,
        "left_inner_finger_joint":  -s,
        "right_inner_knuckle_joint": -s,
        "right_inner_finger_joint":  s,
        "right_outer_knuckle_joint": -s,
        "right_outer_finger_joint":  0.0,
    }


def _to_joint_states_combined(
    left_joint: np.ndarray,
    right_joint: np.ndarray,
    is_rad: bool = True,
) -> Dict[str, float]:
    """构建 robot.urdf 所需的 left_/right_ 前缀关节状态字典。"""
    states: Dict[str, float] = {}
    for i in range(7):
        lv = float(left_joint[i])
        rv = float(right_joint[i])
        if not is_rad:
            lv = lv / 180.0 * math.pi
            rv = rv / 180.0 * math.pi
        states[f"left_joint{i+1}"]  = lv
        states[f"right_joint{i+1}"] = rv
    for key, val in _gripper_joint_states(float(left_joint[7])).items():
        states[f"left_{key}"] = val
    for key, val in _gripper_joint_states(float(right_joint[7])).items():
        states[f"right_{key}"] = val
    return states


def _fk_combined(
    left_joint: np.ndarray,
    right_joint: np.ndarray,
    urdf_file: str,
    with_visuals_map: bool = False,
):
    chain = kp.build_chain_from_urdf(open(urdf_file).read().encode("utf-8"))
    js = _to_joint_states_combined(left_joint, right_joint)
    if with_visuals_map:
        return chain.forward_kinematics(js), chain.visuals_map()
    return chain.forward_kinematics(js)


# ---------------------------------------------------------------------------
# 主类：JsonRobotRenderer（对齐 RobotRenderer，使用合并 robot.urdf）
# ---------------------------------------------------------------------------

class JsonRobotRenderer:
    """
    合并 robot.urdf 渲染器，标定值和内参均已硬编码。

    参数
    ----
    intrinsic : np.ndarray, optional  相机内参 3×3
    cam_to_base : np.ndarray, optional  4×4 T_cam_predefined_base；默认由 JSON 计算
    urdf : str, optional  robot.urdf 路径
    width, height : int  渲染分辨率
    near_plane, far_plane : float  深度截断
    """

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _DEFAULT_URDF = str(
        _REPO_ROOT / "airexo" / "airexo" / "urdf_models" / "robot" / "robot.urdf"
    )

    def __init__(
        self,
        intrinsic:    Optional[np.ndarray] = None,
        cam_to_base:  Optional[np.ndarray] = None,
        urdf:         str   = _DEFAULT_URDF,
        width:        int   = 1280,
        height:       int   = 720,
        near_plane:   float = 0.01,
        far_plane:    float = 100.0,
    ):
        # ---- 标定矩阵 ----
        # 链条（对齐 calib_info.get_camera_to_base(real_base=False)）：
        #   T_from_left  = T_cam_left_real  @ LEFT_REAL_TO_REAL
        #   T_from_right = T_cam_right_real @ RIGHT_REAL_TO_REAL
        #   T_overall    = average(T_from_left, T_from_right)
        #   cam_to_base  = T_overall @ inv(ROBOT_PREDEFINED)
        if cam_to_base is not None:
            self.cam_to_base = cam_to_base.astype(np.float64)
        else:
            T_from_left  = _pose_wxyz_to_mat(_LEFT_POSE_IN_LINK)  @ ROBOT_LEFT_REAL_BASE_TO_REAL_BASE
            T_from_right = _pose_wxyz_to_mat(_RIGHT_POSE_IN_LINK) @ ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE
            T_overall    = _average_transforms(T_from_left, T_from_right)
            self.cam_to_base = T_overall @ np.linalg.inv(ROBOT_PREDEFINED_TRANSFORMATION)

        intrinsic = np.asarray(
            intrinsic if intrinsic is not None else HARDCODED_INTRINSIC, dtype=np.float64
        )
        self.intrinsic  = intrinsic
        self.width      = width
        self.height     = height
        self.near_plane = near_plane
        self.far_plane  = far_plane
        self.urdf       = urdf

        # ---- Open3D 离屏渲染器 ----
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = "defaultLit"

        # ---- 初始化 FK（零位）----
        zero = np.zeros(8, dtype=np.float32)
        transforms, self.visuals_map = _fk_combined(zero, zero, urdf, with_visuals_map=True)
        self.last_joints = (zero.copy(), zero.copy())

        # ---- 加载 mesh ----
        self.meshes     = {}
        self.tf_cache   = {}
        urdf_dir = os.path.dirname(urdf)
        for link, transform in transforms.items():
            for v in self.visuals_map[link]:
                if v.geom_param is None:
                    continue
                name = f"{link}///{v.geom_param}"
                tf   = self._build_tf(transform, v)
                mesh = o3d.io.read_triangle_mesh(os.path.join(urdf_dir, v.geom_param))
                mesh.transform(tf)
                mesh.compute_vertex_normals()
                self.meshes[name]   = mesh
                self.tf_cache[name] = tf
                self.renderer.scene.add_geometry(name, mesh, self.material)

        # ---- 相机投影 ----
        self.renderer.scene.camera.set_projection(
            self.intrinsic, self.near_plane, self.far_plane,
            float(self.width), float(self.height),
        )

    # ------------------------------------------------------------------
    # 私有
    # ------------------------------------------------------------------

    def _build_tf(self, transform, visual) -> np.ndarray:
        """O3D_RENDER @ cam_to_base @ ROBOT_PREDEFINED @ FK @ offset（与 RobotRenderer 完全相同）"""
        return (
            O3D_RENDER_TRANSFORMATION
            @ self.cam_to_base
            @ ROBOT_PREDEFINED_TRANSFORMATION
            @ transform.matrix()
            @ visual.offset.matrix()
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def update_joints(self, left_joint: np.ndarray, right_joint: np.ndarray):
        """更新关节角度（8-dim per arm：前7维弧度，第8维夹爪宽度m）。"""
        self.last_joints = (left_joint, right_joint)
        transforms = _fk_combined(left_joint, right_joint, self.urdf)
        self.renderer.scene.clear_geometry()
        for link, transform in transforms.items():
            for v in self.visuals_map[link]:
                if v.geom_param is None:
                    continue
                name = f"{link}///{v.geom_param}"
                tf = self._build_tf(transform, v)
                self.meshes[name].transform(tf @ np.linalg.inv(self.tf_cache[name]))
                self.meshes[name].compute_vertex_normals()
                self.tf_cache[name] = tf
                self.renderer.scene.add_geometry(name, self.meshes[name], self.material)

    def update_cam_to_base(self, cam_to_base: np.ndarray, delay_update: bool = False):
        self.cam_to_base = cam_to_base.astype(np.float64)
        if not delay_update:
            self.update_joints(*self.last_joints)

    def render_image(self) -> np.ndarray:
        return np.asarray(self.renderer.render_to_image(), dtype=np.uint8)

    def render_depth(self) -> np.ndarray:
        return np.asarray(
            self.renderer.render_to_depth_image(z_in_view_space=True), dtype=np.float32
        )

    def render_mask(self, depth: Optional[np.ndarray] = None) -> np.ndarray:
        if depth is None:
            depth = self.render_depth()
        mask = np.zeros(depth.shape, dtype=np.uint8)
        mask[depth < np.inf] = 255
        return mask

    def render_debug_view(self, pad_x: int = 640, pad_y: int = 360) -> np.ndarray:
        """
        渲染扩大视角调试图，绿框标出原始相机 FOV。
        返回 (H+2*pad_y, W+2*pad_x, 3) uint8 RGB 图像。
        """
        import cv2

        new_w = self.width  + 2 * pad_x
        new_h = self.height + 2 * pad_y

        expanded_K = self.intrinsic.copy()
        expanded_K[0, 2] += pad_x
        expanded_K[1, 2] += pad_y

        tmp = o3d.visualization.rendering.OffscreenRenderer(new_w, new_h)
        tmp.scene.set_background([0.2, 0.2, 0.2, 1.0])
        tmp_mat = o3d.visualization.rendering.MaterialRecord()
        tmp_mat.shader = "defaultLit"
        for name, mesh in self.meshes.items():
            tmp.scene.add_geometry(name, copy.deepcopy(mesh), tmp_mat)
        tmp.scene.camera.set_projection(
            expanded_K, self.near_plane, self.far_plane, float(new_w), float(new_h),
        )

        img = np.asarray(tmp.render_to_image(), dtype=np.uint8)
        x0, y0 = pad_x, pad_y
        x1, y1 = pad_x + self.width - 1, pad_y + self.height - 1
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cx = int(round(self.intrinsic[0, 2])) + pad_x
        cy = int(round(self.intrinsic[1, 2])) + pad_y
        cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        return img


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_floats(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split(",")], dtype=np.float32)


def main():
    p = argparse.ArgumentParser(
        description="渲染机械臂 URDF mask（标定值已硬编码，使用合并 robot.urdf）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--left-joints", default=None,
        help="左臂关节角（弧度）+ 夹爪宽度（米），8 个逗号分隔值\n"
             "不传则使用 DEFAULT_LEFT_JOINTS（来自 robot_sep.yaml 测试值）",
    )
    p.add_argument(
        "--right-joints", default=None,
        help="右臂关节角（弧度）+ 夹爪宽度（米），8 个逗号分隔值\n"
             "不传则使用 DEFAULT_RIGHT_JOINTS（来自 robot_sep.yaml 测试值）",
    )
    p.add_argument("--width",       type=int, default=1280)
    p.add_argument("--height",      type=int, default=720)
    p.add_argument("--output",      default="mask.png",  help="输出 mask 路径")
    p.add_argument("--save-image",  default=None,        help="保存彩色渲染图（可选）")
    p.add_argument("--debug-view",  default=None,        help="保存扩大视角调试图（带 FOV 框）")
    p.add_argument("--pad-x",       type=int, default=640)
    p.add_argument("--pad-y",       type=int, default=360)
    args = p.parse_args()

    left_j  = _parse_floats(args.left_joints)  if args.left_joints  else DEFAULT_LEFT_JOINTS
    right_j = _parse_floats(args.right_joints) if args.right_joints else DEFAULT_RIGHT_JOINTS

    renderer = JsonRobotRenderer(width=args.width, height=args.height)
    renderer.update_joints(left_j, right_j)

    depth = renderer.render_depth()
    mask  = renderer.render_mask(depth=depth)

    import cv2
    cv2.imwrite(args.output, mask)
    print(f"[ok] mask saved: {args.output}  ({mask.sum() // 255} pixels)")

    if args.save_image:
        image = renderer.render_image()
        cv2.imwrite(args.save_image, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"[ok] image saved: {args.save_image}")

    if args.debug_view:
        dbg = renderer.render_debug_view(pad_x=args.pad_x, pad_y=args.pad_y)
        cv2.imwrite(args.debug_view, cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))
        print(f"[ok] debug view saved: {args.debug_view}  (pad={args.pad_x}x{args.pad_y})")


if __name__ == "__main__":
    main()
