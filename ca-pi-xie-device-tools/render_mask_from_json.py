#!/usr/bin/env python3
"""
基于左右标定 JSON 渲染 URDF mask（左右臂分别渲染）。

设计说明
--------
直接模仿 airexo/helpers/renderer.py 的 SeparateRobotRenderer，
但去掉 hydra / CalibrationInfo 等依赖，改为内嵌硬编码标定值。

硬编码标定来源
--------------
相机内参（serial: 104122061850）：
  fx=908.4968, fy=907.65515, cx=640.631, cy=351.3301

左臂 pose_in_link → T_cam_left_base:
  [0.0861, -0.0994, 0.3423,  qw=0.2267, qx=-0.6847, qy=0.6615, qz=-0.2057]

右臂 pose_in_link → T_cam_right_base:
  [0.0778,  0.2079, 0.3472,  qw=0.2273, qx=-0.6786, qy=0.6638, qz=-0.2175]

标定矩阵处理（重要）：
  JSON 的 pose_in_link = T_cam_individual_real_base（单臂局部 base，非 URDF base_link）
  ROBOT_LEFT_REAL_BASE_TO_REAL_BASE = Y 方向 -135mm 平移（从局部 base 到 URDF base_link）
  正确链条（参考 build_fake_airexo_robot_calib.py line 101）：
    cam_to_left_base = T_cam_individual @ LEFT_REAL_BASE_TO_REAL_BASE @ inv(PREDEFINED)

  展开完整渲染链条：
    O3D_RENDER
    @ (T_cam_indiv @ LEFT_REAL_BASE_TO_REAL_BASE @ inv(PREDEFINED))
    @ PREDEFINED @ LEFT_PREDEFINED @ FK @ offset
  = O3D_RENDER @ T_cam_urdf_base @ LEFT_PREDEFINED @ FK @ offset  ← 正确

使用示例
--------
  from render_mask_from_json import JsonSeparateRobotRenderer
  import numpy as np

  renderer = JsonSeparateRobotRenderer()
  left_joint  = np.array([j1..j7, gripper_width], dtype=np.float32)  # 8-dim, rad + meters
  right_joint = np.array([j1..j7, gripper_width], dtype=np.float32)
  renderer.update_joints(left_joint, right_joint)
  mask  = renderer.render_mask()    # uint8 (H, W), 255 = robot pixel
  image = renderer.render_image()   # uint8 (H, W, 3)
  depth = renderer.render_depth()   # float32 (H, W)

CLI（直接运行，不传关节角则使用 robot_sep.yaml 测试默认值）：
  python render_mask_from_json.py --output mask.png --save-image render.png
  python render_mask_from_json.py \\
      --left-joints  "1.078,-2.201,1.628,-1.162,-0.958,0.503,0.875,0.05" \\
      --right-joints "1.397,-2.226,1.835,-1.367,-0.647,0.781,0.972,0.05" \\
      --output mask.png --save-image render.png
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import open3d as o3d
import kinpy as kp


# ---------------------------------------------------------------------------
# 硬编码标定数据
# ---------------------------------------------------------------------------

# 相机内参（serial 104122061850，intrinsics.npy 第一个键）
HARDCODED_INTRINSIC = np.array([
    [908.4968,   0.0,     640.631 ],
    [  0.0,    907.65515, 351.3301],
    [  0.0,      0.0,       1.0   ],
], dtype=np.float32)

# ---------------------------------------------------------------------------
# 硬编码示例关节角（来自 airexo configs/tests/renderer/robot_sep.yaml）
# 格式：[joint1..joint7 (rad), gripper_width (m)]
# ---------------------------------------------------------------------------

DEFAULT_LEFT_JOINTS = np.array([
     1.078, -2.201,  1.628, -1.162, -0.958, 0.503, 0.875,
     0.05,
], dtype=np.float32)

DEFAULT_RIGHT_JOINTS = np.array([
     1.397, -2.226,  1.835, -1.367, -0.647, 0.781, 0.972,
     0.05,
], dtype=np.float32)


# 左臂 pose_in_link: [x, y, z, qw, qx, qy, qz]
_LEFT_POSE_IN_LINK  = [
    0.0860694267750039, -0.09938092775476133, 0.34225723639609734,
    0.22666579355367836, -0.6846736943308571, 0.6614734633385874, -0.20566333563100633,
]
# 右臂 pose_in_link: [x, y, z, qw, qx, qy, qz]
_RIGHT_POSE_IN_LINK = [
    0.07783932332093665, 0.2078814260418823, 0.34723683952957585,
    0.2273133855008057, -0.6785647482083789, 0.6637673415778982, -0.21746591345696367,
]


# ---------------------------------------------------------------------------
# pose → 4×4 矩阵转换（逻辑与 build_rise2_calib_from_json.py 一致）
# ---------------------------------------------------------------------------

def _quat_wxyz_to_mat3(quat_wxyz) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = q
    n = float(w*w + x*x + y*y + z*z)
    if n < 1e-12:
        raise ValueError("Quaternion norm is too small")
    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    return np.array([
        [1.0-(yy+zz), xy-wz,     xz+wy    ],
        [xy+wz,       1.0-(xx+zz), yz-wx  ],
        [xz-wy,       yz+wx,     1.0-(xx+yy)],
    ], dtype=np.float32)


def _pose_wxyz_to_mat(pose) -> np.ndarray:
    """[x,y,z, qw,qx,qy,qz] → 4×4 float32"""
    arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 7:
        raise ValueError(f"pose must have 7 values, got {arr.shape[0]}")
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = _quat_wxyz_to_mat3(arr[3:])
    mat[:3,  3] = arr[:3]
    return mat


# ---------------------------------------------------------------------------
# 预定义变换常量（与 airexo/helpers/constants.py 完全相同）
# ---------------------------------------------------------------------------

O3D_RENDER_TRANSFORMATION = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1],
], dtype=np.float32)

ROBOT_PREDEFINED_TRANSFORMATION = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float32)

_sqrt2_2 = float(np.sqrt(2) / 2)

LEFT_ROBOT_PREDEFINED_TRANSFORMATION = np.array([
    [1, 0,        0,       0],
    [0, _sqrt2_2, _sqrt2_2, 0],
    [0, -_sqrt2_2, _sqrt2_2, 0],
    [0, 0,        0,        1],
], dtype=np.float32)

RIGHT_ROBOT_PREDEFINED_TRANSFORMATION = np.array([
    [1, 0,        0,        0],
    [0, _sqrt2_2, -_sqrt2_2, 0],
    [0, _sqrt2_2,  _sqrt2_2, 0],
    [0, 0,        0,         1],
], dtype=np.float32)

# 135mm offset between "individual real base" (what the JSON stores)
# and the "overall real base" that the URDF base_link corresponds to.
# From airexo/helpers/constants.py: ROBOT_REAL_BASE_TO_INDIVIDUAL_REAL_BASE = 0.135
_REAL_BASE_OFFSET = 0.135
ROBOT_LEFT_REAL_BASE_TO_REAL_BASE = np.array([
    [1.0, 0.0, 0.0,  0.0              ],
    [0.0, 1.0, 0.0, -_REAL_BASE_OFFSET],
    [0.0, 0.0, 1.0,  0.0              ],
    [0.0, 0.0, 0.0,  1.0              ],
], dtype=np.float32)
ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE = np.array([
    [1.0, 0.0, 0.0,  0.0              ],
    [0.0, 1.0, 0.0, +_REAL_BASE_OFFSET],
    [0.0, 0.0, 1.0,  0.0              ],
    [0.0, 0.0, 0.0,  1.0              ],
], dtype=np.float32)


# ---------------------------------------------------------------------------
# 关节配置（简化版，无 hydra 依赖）
# ---------------------------------------------------------------------------

class RobotJointCfgs:
    """
    最小化关节配置，对齐 airexo joint yaml 的语义：
      num_joints       = 8  (7 robot DOF + 1 gripper)
      num_robot_joints = 7
    """
    num_joints: int = 8
    num_robot_joints: int = 7


# ---------------------------------------------------------------------------
# 关节状态转换（复用 airexo/helpers/urdf_robot.py 的逻辑）
# ---------------------------------------------------------------------------

def _calc_robotiq_state(open_length: float) -> float:
    return 0.725 - math.asin(np.clip(open_length, 0, 0.1143) / 0.1143)


def _gripper_joint_states(gripper_width: float) -> Dict[str, float]:
    w = float(np.clip(gripper_width, 0, 0.085))
    s = _calc_robotiq_state(w)
    return {
        "finger_joint": s,
        "left_outer_finger_joint": 0.0,
        "left_inner_knuckle_joint": s,
        "left_inner_finger_joint": -s,
        "right_inner_knuckle_joint": -s,
        "right_inner_finger_joint": s,
        "right_outer_knuckle_joint": -s,
        "right_outer_finger_joint": 0.0,
    }


def _to_joint_states_single(joint: np.ndarray, is_rad: bool = True) -> Dict[str, float]:
    """8-dim joint array → kinpy joint_states dict（单臂，无前缀）"""
    states: Dict[str, float] = {}
    for i in range(7):
        val = float(joint[i])
        if not is_rad:
            val = val / 180.0 * math.pi
        states[f"joint{i+1}"] = val
    states.update(_gripper_joint_states(float(joint[7])))
    return states


def _fk_single(
    joint: np.ndarray,
    urdf_file: str,
    is_rad: bool = True,
    with_visuals_map: bool = False,
):
    """单臂正运动学，返回 (transforms) 或 (transforms, visuals_map)"""
    chain = kp.build_chain_from_urdf(open(urdf_file).read().encode("utf-8"))
    js = _to_joint_states_single(joint, is_rad=is_rad)
    if with_visuals_map:
        return chain.forward_kinematics(js), chain.visuals_map()
    return chain.forward_kinematics(js)


# ---------------------------------------------------------------------------
# 主类：JsonSeparateRobotRenderer
# ---------------------------------------------------------------------------

class JsonSeparateRobotRenderer:
    """
    左右臂分别渲染器，标定值和内参均已硬编码。

    参数
    ----
    intrinsic : np.ndarray, optional
        相机内参 3×3；默认使用 HARDCODED_INTRINSIC
    cam_to_left_base : np.ndarray, optional
        4×4 T_cam_left_base；默认由 _LEFT_POSE_IN_LINK 计算
    cam_to_right_base : np.ndarray, optional
        4×4 T_cam_right_base；默认由 _RIGHT_POSE_IN_LINK 计算
    left_urdf : str, optional
        左臂 URDF 路径；默认自动定位到仓库内 left_robot.urdf
    right_urdf : str, optional
        右臂 URDF 路径；默认自动定位到仓库内 right_robot.urdf
    width, height : int
        渲染分辨率
    near_plane, far_plane : float
        深度截断范围
    """

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _DEFAULT_LEFT_URDF  = str(_REPO_ROOT / "airexo" / "airexo" / "urdf_models" / "robot" / "left_robot.urdf")
    _DEFAULT_RIGHT_URDF = str(_REPO_ROOT / "airexo" / "airexo" / "urdf_models" / "robot" / "right_robot.urdf")

    def __init__(
        self,
        intrinsic:         Optional[np.ndarray] = None,
        cam_to_left_base:  Optional[np.ndarray] = None,
        cam_to_right_base: Optional[np.ndarray] = None,
        left_urdf:         str   = _DEFAULT_LEFT_URDF,
        right_urdf:        str   = _DEFAULT_RIGHT_URDF,
        width:             int   = 1280,
        height:            int   = 720,
        near_plane:        float = 0.01,
        far_plane:         float = 100.0,
    ):
        # ---- 标定矩阵（默认使用硬编码值） ----
        # JSON 的 pose_in_link = T_cam_individual_real_base（单臂局部 base）。
        # build_fake_airexo_robot_calib.py 的真实链条（line 101-102）：
        #   cam_to_left_predefined_base = T_cam_left_individual_real_base
        #                                 @ ROBOT_LEFT_REAL_BASE_TO_REAL_BASE
        #                                 @ inv(ROBOT_PREDEFINED)
        # 其中 ROBOT_LEFT_REAL_BASE_TO_REAL_BASE 是 -135mm Y 方向平移，
        # 代表单臂 individual base 到整体/URDF base_link 的偏移。
        _inv_predefined = np.linalg.inv(ROBOT_PREDEFINED_TRANSFORMATION)
        if cam_to_left_base is not None:
            self.cam_to_left_base = cam_to_left_base
        else:
            self.cam_to_left_base = (
                _pose_wxyz_to_mat(_LEFT_POSE_IN_LINK)
                @ ROBOT_LEFT_REAL_BASE_TO_REAL_BASE
                @ _inv_predefined
            )
        if cam_to_right_base is not None:
            self.cam_to_right_base = cam_to_right_base
        else:
            self.cam_to_right_base = (
                _pose_wxyz_to_mat(_RIGHT_POSE_IN_LINK)
                @ ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE
                @ _inv_predefined
            )
        intrinsic = intrinsic if intrinsic is not None else HARDCODED_INTRINSIC
        self.intrinsic   = intrinsic.astype(np.float64)
        self.width       = width
        self.height      = height
        self.near_plane  = near_plane
        self.far_plane   = far_plane

        self.urdf_left  = left_urdf
        self.urdf_right = right_urdf

        # ---- Open3D 离屏渲染器 ----
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = "defaultLit"

        # ---- 初始化正运动学（零位） ----
        cfgs = RobotJointCfgs()
        zero = np.zeros(cfgs.num_joints, dtype=np.float32)

        tf_left,  self.visuals_left  = _fk_single(zero, left_urdf,  with_visuals_map=True)
        tf_right, self.visuals_right = _fk_single(zero, right_urdf, with_visuals_map=True)

        self.last_joints = (zero.copy(), zero.copy())

        # ---- 加载 mesh，建立初始场景 ----
        self.meshes_left   = {}
        self.tf_cache_left = {}
        self._load_meshes(
            transforms=tf_left,
            visuals_map=self.visuals_left,
            cam_to_base=self.cam_to_left_base,
            predefined=LEFT_ROBOT_PREDEFINED_TRANSFORMATION,
            urdf_file=left_urdf,
            side="left",
            mesh_store=self.meshes_left,
            tf_cache=self.tf_cache_left,
        )

        self.meshes_right   = {}
        self.tf_cache_right = {}
        self._load_meshes(
            transforms=tf_right,
            visuals_map=self.visuals_right,
            cam_to_base=self.cam_to_right_base,
            predefined=RIGHT_ROBOT_PREDEFINED_TRANSFORMATION,
            urdf_file=right_urdf,
            side="right",
            mesh_store=self.meshes_right,
            tf_cache=self.tf_cache_right,
        )

        # ---- 相机投影 ----
        self.renderer.scene.camera.set_projection(
            self.intrinsic,
            self.near_plane, self.far_plane,
            float(self.width), float(self.height),
        )

    # ------------------------------------------------------------------
    # 私有工具
    # ------------------------------------------------------------------

    def _build_tf(
        self,
        transform,
        visual,
        cam_to_base: np.ndarray,
        predefined:  np.ndarray,
    ) -> np.ndarray:
        return (
            O3D_RENDER_TRANSFORMATION
            @ cam_to_base
            @ ROBOT_PREDEFINED_TRANSFORMATION
            @ predefined
            @ transform.matrix()
            @ visual.offset.matrix()
        )

    def _load_meshes(
        self,
        transforms,
        visuals_map,
        cam_to_base: np.ndarray,
        predefined:  np.ndarray,
        urdf_file:   str,
        side:        str,
        mesh_store:  dict,
        tf_cache:    dict,
    ):
        urdf_dir = os.path.dirname(urdf_file)
        for link, transform in transforms.items():
            for v in visuals_map[link]:
                if v.geom_param is None:
                    continue
                name = f"{side}///{link}///{v.geom_param}"
                tf   = self._build_tf(transform, v, cam_to_base, predefined)
                mesh = o3d.io.read_triangle_mesh(os.path.join(urdf_dir, v.geom_param))
                mesh.transform(tf)
                mesh.compute_vertex_normals()
                mesh_store[name] = mesh
                tf_cache[name]   = tf
                self.renderer.scene.add_geometry(name, mesh, self.material)

    def _update_side(
        self,
        joint:       np.ndarray,
        transforms,
        visuals_map,
        cam_to_base: np.ndarray,
        predefined:  np.ndarray,
        side:        str,
        mesh_store:  dict,
        tf_cache:    dict,
    ):
        for link, transform in transforms.items():
            for v in visuals_map[link]:
                if v.geom_param is None:
                    continue
                name = f"{side}///{link}///{v.geom_param}"
                tf   = self._build_tf(transform, v, cam_to_base, predefined)
                mesh_store[name].transform(tf @ np.linalg.inv(tf_cache[name]))
                mesh_store[name].compute_vertex_normals()
                tf_cache[name] = tf
                self.renderer.scene.add_geometry(name, mesh_store[name], self.material)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def update_joints(self, left_joint: np.ndarray, right_joint: np.ndarray):
        """更新关节角度并刷新场景。joint 为 8-dim，前 7 维为弧度，第 8 维为夹爪宽度（米）。"""
        self.last_joints = (left_joint, right_joint)

        tf_left  = _fk_single(left_joint,  self.urdf_left,  with_visuals_map=False)
        tf_right = _fk_single(right_joint, self.urdf_right, with_visuals_map=False)

        self.renderer.scene.clear_geometry()
        self._update_side(
            left_joint, tf_left, self.visuals_left,
            self.cam_to_left_base, LEFT_ROBOT_PREDEFINED_TRANSFORMATION,
            "left", self.meshes_left, self.tf_cache_left,
        )
        self._update_side(
            right_joint, tf_right, self.visuals_right,
            self.cam_to_right_base, RIGHT_ROBOT_PREDEFINED_TRANSFORMATION,
            "right", self.meshes_right, self.tf_cache_right,
        )

    def update_camera_to_left_base(self, cam_to_left_base: np.ndarray, delay_update: bool = False):
        self.cam_to_left_base = cam_to_left_base
        if not delay_update:
            self.update_joints(*self.last_joints)

    def update_camera_to_right_base(self, cam_to_right_base: np.ndarray, delay_update: bool = False):
        self.cam_to_right_base = cam_to_right_base
        if not delay_update:
            self.update_joints(*self.last_joints)

    def render_image(self) -> np.ndarray:
        return np.asarray(self.renderer.render_to_image(), dtype=np.uint8)

    def render_depth(self) -> np.ndarray:
        return np.asarray(self.renderer.render_to_depth_image(z_in_view_space=True), dtype=np.float32)

    def render_mask(self, depth: Optional[np.ndarray] = None) -> np.ndarray:
        if depth is None:
            depth = self.render_depth()
        mask = np.zeros(depth.shape, dtype=np.uint8)
        mask[depth < np.inf] = 255
        return mask

    def render_debug_view(self, pad_x: int = 640, pad_y: int = 360) -> np.ndarray:
        """
        渲染扩大视角的调试图，并在其中画出原始相机 FOV 矩形框。

        原理：保持焦距不变，仅向四周扩展画布（pad_x/pad_y 像素），
        通过平移主点实现等效广角视图，无需修改场景几何体。

        返回: (H+2*pad_y, W+2*pad_x, 3) uint8 RGB 图像
        """
        import cv2

        new_w = self.width  + 2 * pad_x
        new_h = self.height + 2 * pad_y

        # 扩展内参：焦距不变，主点偏移
        expanded_K = self.intrinsic.copy()
        expanded_K[0, 2] += pad_x
        expanded_K[1, 2] += pad_y

        # 创建临时渲染器
        tmp_renderer = o3d.visualization.rendering.OffscreenRenderer(new_w, new_h)
        tmp_renderer.scene.set_background([0.2, 0.2, 0.2, 1.0])
        tmp_mat = o3d.visualization.rendering.MaterialRecord()
        tmp_mat.shader = "defaultLit"

        # 将当前所有已变换的 mesh 添加到临时场景
        for name, mesh in {**self.meshes_left, **self.meshes_right}.items():
            tmp_renderer.scene.add_geometry(name, mesh, tmp_mat)

        tmp_renderer.scene.camera.set_projection(
            expanded_K,
            self.near_plane, self.far_plane,
            float(new_w), float(new_h),
        )

        img = np.asarray(tmp_renderer.render_to_image(), dtype=np.uint8)

        # 画出原始相机 FOV 边框（绿色）
        x0, y0 = pad_x,              pad_y
        x1, y1 = pad_x + self.width - 1, pad_y + self.height - 1
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        # 中心十字
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
        description="渲染机械臂 URDF mask（标定值已硬编码）",
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
    p.add_argument("--width",      type=int, default=1280)
    p.add_argument("--height",     type=int, default=720)
    p.add_argument("--output",     default="mask.png", help="输出 mask 文件路径")
    p.add_argument("--save-image", default=None,       help="同时保存彩色渲染图（可选）")
    p.add_argument(
        "--debug-view", default=None,
        help="保存扩大视角调试图（带 FOV 框），例如 debug.png",
    )
    p.add_argument("--pad-x", type=int, default=640, help="debug 视图水平扩展像素（每侧）")
    p.add_argument("--pad-y", type=int, default=360, help="debug 视图垂直扩展像素（每侧）")
    args = p.parse_args()

    left_j  = _parse_floats(args.left_joints)  if args.left_joints  else DEFAULT_LEFT_JOINTS
    right_j = _parse_floats(args.right_joints) if args.right_joints else DEFAULT_RIGHT_JOINTS

    renderer = JsonSeparateRobotRenderer(width=args.width, height=args.height)

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
