#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

from airexo.helpers.constants import (
    ROBOT_LEFT_CAM_TO_TCP,
    ROBOT_RIGHT_CAM_TO_TCP,
    ROBOT_LEFT_REAL_BASE_TO_REAL_BASE,
    ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE,
    ROBOT_PREDEFINED_TRANSFORMATION,
)
from airexo.helpers.rotation import average_xyz_rot_quat, xyz_rot_to_mat


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def pose7_wxyz_to_mat(pose7: Any) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    t = pose7[:3]
    qw, qx, qy, qz = pose7[3:]
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    mat[:3, 3] = t
    return mat


def load_json_pose(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    if "pose_in_link" not in data:
        raise KeyError(f"Missing pose_in_link in {path}")
    return pose7_wxyz_to_mat(data["pose_in_link"])


def load_calib(path: Path) -> Dict[str, Any]:
    data = np.load(str(path), allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()
    if not isinstance(data, dict):
        raise TypeError(f"Calibration file {path} does not contain a dict")
    return data


def renderer_left_cam_to_real_base(calib: Dict[str, Any], global_serial: str) -> np.ndarray:
    tcp_pose = xyz_rot_to_mat(calib["robot_left"]["tcp_pose"], rotation_rep="quaternion").astype(np.float64)
    return (
        np.asarray(calib["extrinsics"][global_serial], dtype=np.float64)
        @ invert_T(np.asarray(calib["extrinsics"][calib["camera_serial_inhand_left"]], dtype=np.float64))
        @ np.asarray(ROBOT_LEFT_CAM_TO_TCP, dtype=np.float64)
        @ invert_T(tcp_pose)
    )


def renderer_right_cam_to_real_base(calib: Dict[str, Any], global_serial: str) -> np.ndarray:
    tcp_pose = xyz_rot_to_mat(calib["robot_right"]["tcp_pose"], rotation_rep="quaternion").astype(np.float64)
    return (
        np.asarray(calib["extrinsics"][global_serial], dtype=np.float64)
        @ invert_T(np.asarray(calib["extrinsics"][calib["camera_serial_inhand_right"]], dtype=np.float64))
        @ np.asarray(ROBOT_RIGHT_CAM_TO_TCP, dtype=np.float64)
        @ invert_T(tcp_pose)
    )


def renderer_shared_cam_to_real_base(calib: Dict[str, Any], global_serial: str) -> np.ndarray:
    left = renderer_left_cam_to_real_base(calib, global_serial) @ np.asarray(ROBOT_LEFT_REAL_BASE_TO_REAL_BASE, dtype=np.float64)
    right = renderer_right_cam_to_real_base(calib, global_serial) @ np.asarray(ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE, dtype=np.float64)
    return np.asarray(average_xyz_rot_quat(left, right, rotation_rep="matrix"), dtype=np.float64)


def renderer_shared_cam_to_predefined_base(calib: Dict[str, Any], global_serial: str) -> np.ndarray:
    return renderer_shared_cam_to_real_base(calib, global_serial) @ invert_T(np.asarray(ROBOT_PREDEFINED_TRANSFORMATION, dtype=np.float64))


def fro_err(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B, ord="fro"))


def trans_err(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3]))


def rot_deg_err(A: np.ndarray, B: np.ndarray) -> float:
    rel = A[:3, :3] @ B[:3, :3].T
    val = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))


def print_matrix(name: str, T: np.ndarray) -> None:
    print(f"\n[{name}]")
    print(np.array2string(T, formatter={"float_kind": lambda x: f"{x: .6f}"}))


def compare_pair(name_a: str, A: np.ndarray, name_b: str, B: np.ndarray) -> None:
    print(f"\n[compare] {name_a}  vs  {name_b}")
    print(f"frobenius_error   : {fro_err(A, B):.9f}")
    print(f"translation_error : {trans_err(A, B):.9f} m")
    print(f"rotation_error    : {rot_deg_err(A, B):.9f} deg")


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 JSON 标定与 Airexo renderer 公式链条是否一致，并排查 inverse / base 层级问题")
    parser.add_argument("--left-json", required=True)
    parser.add_argument("--right-json", required=True)
    parser.add_argument("--airexo-calib", required=True, help="Airexo renderer 实际使用或拟使用的 robot calib .npy")
    parser.add_argument("--global-serial", default=None, help="不填则取 calib 中第一个 global serial")
    parser.add_argument("--print-matrices", action="store_true", help="打印关键矩阵")
    args = parser.parse_args()

    left_json = load_json_pose(Path(args.left_json)).astype(np.float64)
    right_json = load_json_pose(Path(args.right_json)).astype(np.float64)
    calib = load_calib(Path(args.airexo_calib))

    global_serial = args.global_serial
    if global_serial is None:
        global_serial = calib["camera_serials_global"][0]

    left_renderer_real = renderer_left_cam_to_real_base(calib, global_serial)
    right_renderer_real = renderer_right_cam_to_real_base(calib, global_serial)

    left_renderer_real_inv = invert_T(left_renderer_real)
    right_renderer_real_inv = invert_T(right_renderer_real)
    left_json_inv = invert_T(left_json)
    right_json_inv = invert_T(right_json)

    shared_real = renderer_shared_cam_to_real_base(calib, global_serial)
    shared_predefined = renderer_shared_cam_to_predefined_base(calib, global_serial)

    print(f"[info] global_serial: {global_serial}")
    print(f"[info] camera_serial_inhand_left: {calib['camera_serial_inhand_left']}")
    print(f"[info] camera_serial_inhand_right: {calib['camera_serial_inhand_right']}")

    compare_pair("left_json", left_json, "left_renderer_real", left_renderer_real)
    compare_pair("right_json", right_json, "right_renderer_real", right_renderer_real)

    compare_pair("left_json_inv", left_json_inv, "left_renderer_real", left_renderer_real)
    compare_pair("right_json_inv", right_json_inv, "right_renderer_real", right_renderer_real)

    compare_pair("left_json", left_json, "left_renderer_real_inv", left_renderer_real_inv)
    compare_pair("right_json", right_json, "right_renderer_real_inv", right_renderer_real_inv)

    print("\n[hint]")
    print("- 如果 left_json vs left_renderer_real 误差很小，说明 JSON 语义与 renderer 恢复出的 left real base 基本一致。")
    print("- 如果 left_json_inv vs left_renderer_real 更小，说明 JSON 很可能方向写反了。右臂同理。")
    print("- 如果左右单臂都能对上，但共享渲染仍差，问题更可能出在 shared base 合成或 predefined base 对齐。")

    if args.print_matrices:
        print_matrix("left_json", left_json)
        print_matrix("right_json", right_json)
        print_matrix("left_renderer_real", left_renderer_real)
        print_matrix("right_renderer_real", right_renderer_real)
        print_matrix("shared_real", shared_real)
        print_matrix("shared_predefined", shared_predefined)
        print_matrix("ROBOT_PREDEFINED_TRANSFORMATION", np.asarray(ROBOT_PREDEFINED_TRANSFORMATION, dtype=np.float64))


if __name__ == "__main__":
    main()
