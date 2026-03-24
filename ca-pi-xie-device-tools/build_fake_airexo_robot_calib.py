import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

WORKSPACE_ROOT = "/home/haoxiang/rise2_mask_aware"
AIREXO_ROOT = os.path.join(WORKSPACE_ROOT, "airexo")
for p in [WORKSPACE_ROOT, AIREXO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from airexo.helpers.constants import (
    ROBOT_LEFT_CAM_TO_TCP,
    ROBOT_RIGHT_CAM_TO_TCP,
    ROBOT_LEFT_REAL_BASE_TO_REAL_BASE,
    ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE,
    ROBOT_PREDEFINED_TRANSFORMATION,
)
from airexo.helpers.rotation import mat_to_xyz_rot, average_xyz_rot_quat

# =========================
# 用户可修改配置
# =========================
LEFT_JSON = Path("/data/haoxiang/data/task0012_260321/calib/left_global_20260104/result.json")
RIGHT_JSON = Path("/data/haoxiang/data/task0012_260321/calib/right_global_20260104/result.json")
INTRINSICS_NPY = Path("/data/haoxiang/data/task0012_260321/1736320913189/intrinsics.npy")
OUTPUT_DIR = Path("/data/haoxiang/data/task0012_260321/calib")
OUTPUT_TIMESTAMP = "20260104_fake_airexo_robot"
GLOBAL_SERIAL = "104122060902"
GLOBAL_INTRINSIC_KEY = "104122061850"
INHAND_LEFT_SERIAL = "104122064161"
INHAND_RIGHT_SERIAL = "104122061330"


def pose7_wxyz_to_mat(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    t = pose7[:3]
    qw, qx, qy, qz = pose7[3:]
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    mat[:3, 3] = t
    return mat


def load_pose_in_link(json_path: Path) -> np.ndarray:
    data = json.loads(json_path.read_text())
    pose = np.asarray(data["pose_in_link"], dtype=np.float64)
    return pose7_wxyz_to_mat(pose)


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def mat_to_pose7_quat(T: np.ndarray) -> np.ndarray:
    pose = mat_to_xyz_rot(T, rotation_rep="quaternion")
    return np.asarray(pose, dtype=np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 真实语义：这两个 JSON 就是 camera -> left_real_base / camera -> right_real_base
    cam_to_left_real_base = load_pose_in_link(LEFT_JSON)
    cam_to_right_real_base = load_pose_in_link(RIGHT_JSON)

    intrinsics_all = np.load(INTRINSICS_NPY, allow_pickle=True).item()
    intrinsic = np.asarray(intrinsics_all[GLOBAL_INTRINSIC_KEY], dtype=np.float32)

    # 根据 CalibrationInfo(type='robot') 的官方链条，先把左右 real base 升到 overall real base
    cam_to_overall_real_base_from_left = cam_to_left_real_base @ ROBOT_LEFT_REAL_BASE_TO_REAL_BASE
    cam_to_overall_real_base_from_right = cam_to_right_real_base @ ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE
    cam_to_overall_real_base = average_xyz_rot_quat(
        cam_to_overall_real_base_from_left,
        cam_to_overall_real_base_from_right,
        rotation_rep="matrix",
    )

    # renderer 最终吃的是 camera -> urdf/predefined base
    cam_to_predefined_base = cam_to_overall_real_base @ np.linalg.inv(ROBOT_PREDEFINED_TRANSFORMATION)

    # 逆向构造一份兼容 CalibrationInfo(type='robot') 的文件：
    # 令 global / inhand extrinsics 全为单位阵，则：
    #   cam_to_left_base(predefined)  = ROBOT_LEFT_CAM_TO_TCP  @ inv(tcp_pose_left)
    #   cam_to_right_base(predefined) = ROBOT_RIGHT_CAM_TO_TCP @ inv(tcp_pose_right)
    # 又因为 CalibrationInfo.get_camera_to_base() 会把左右相机到 real base 提升后平均，
    # 所以更关键的是让左右局部 base 和整体链条自洽。
    extrinsics = {
        GLOBAL_SERIAL: np.eye(4, dtype=np.float32),
        INHAND_LEFT_SERIAL: np.eye(4, dtype=np.float32),
        INHAND_RIGHT_SERIAL: np.eye(4, dtype=np.float32),
    }

    cam_to_left_predefined_base = cam_to_left_real_base @ ROBOT_LEFT_REAL_BASE_TO_REAL_BASE @ np.linalg.inv(ROBOT_PREDEFINED_TRANSFORMATION)
    cam_to_right_predefined_base = cam_to_right_real_base @ ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE @ np.linalg.inv(ROBOT_PREDEFINED_TRANSFORMATION)

    tcp_pose_left_mat = invert_T(cam_to_left_predefined_base) @ ROBOT_LEFT_CAM_TO_TCP
    tcp_pose_right_mat = invert_T(cam_to_right_predefined_base) @ ROBOT_RIGHT_CAM_TO_TCP

    tcp_pose_left = mat_to_pose7_quat(tcp_pose_left_mat)
    tcp_pose_right = mat_to_pose7_quat(tcp_pose_right_mat)

    calib = {
        "type": "robot",
        "camera_serials": [GLOBAL_SERIAL, INHAND_LEFT_SERIAL, INHAND_RIGHT_SERIAL],
        "camera_serials_global": [GLOBAL_SERIAL],
        "camera_serial_inhand_left": INHAND_LEFT_SERIAL,
        "camera_serial_inhand_right": INHAND_RIGHT_SERIAL,
        "intrinsics": {
            GLOBAL_SERIAL: intrinsic,
            INHAND_LEFT_SERIAL: intrinsic.copy(),
            INHAND_RIGHT_SERIAL: intrinsic.copy(),
        },
        "extrinsics": extrinsics,
        "robot_left": {
            "tcp_pose": tcp_pose_left,
            "joint_pos": np.zeros((8,), dtype=np.float32),
            "tcp_vel": np.zeros((6,), dtype=np.float32),
            "joint_vel": np.zeros((8,), dtype=np.float32),
            "force_torque": np.zeros((6,), dtype=np.float32),
        },
        "robot_right": {
            "tcp_pose": tcp_pose_right,
            "joint_pos": np.zeros((8,), dtype=np.float32),
            "tcp_vel": np.zeros((6,), dtype=np.float32),
            "joint_vel": np.zeros((8,), dtype=np.float32),
            "force_torque": np.zeros((6,), dtype=np.float32),
        },
    }

    output_path = OUTPUT_DIR / f"{OUTPUT_TIMESTAMP}.npy"
    np.save(output_path, calib, allow_pickle=True)

    print(f"saved: {output_path}")
    print("\n[camera_to_left_real_base from json]")
    print(cam_to_left_real_base)
    print("\n[camera_to_right_real_base from json]")
    print(cam_to_right_real_base)
    print("\n[camera_to_overall_real_base_from_left]")
    print(cam_to_overall_real_base_from_left)
    print("\n[camera_to_overall_real_base_from_right]")
    print(cam_to_overall_real_base_from_right)
    print("\n[averaged camera_to_overall_real_base]")
    print(cam_to_overall_real_base)
    print("\n[camera_to_predefined_base for renderer]")
    print(cam_to_predefined_base)
    print("\n[synthetic robot_left tcp_pose]")
    print(tcp_pose_left)
    print("\n[synthetic robot_right tcp_pose]")
    print(tcp_pose_right)


if __name__ == "__main__":
    main()
