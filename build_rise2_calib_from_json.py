#!/usr/bin/env python3
"""
根据左右相机外参 JSON 和一个内参来源，生成 RISE2 风格标定文件。

当前脚本面向本项目当前需求：
- 左右两个 JSON 已经分别给出“global 相机相对于左/右机械臂”的外参；
- pose_in_link 格式为 [x, y, z, qw, qx, qy, qz]；
- 使用一份 intrinsics.npy 中的某个 3x3 内参，映射到目标 global serial；
- 输出为 RISE2 风格 `.npy`，字段与 [`DualArmProjector`](dataset/projector.py:70) 兼容。

严格说明：
- 本脚本不会尝试还原 Airexo 原始标定链条；
- 本脚本直接生成下游训练/读取所需的 RISE2 风格标定。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from transforms3d.quaternions import quat2mat


def pose_wxyz_to_mat(pose: Any) -> np.ndarray:
    arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 7:
        raise ValueError(f"Expected pose with 7 values [x,y,z,qw,qx,qy,qz], got shape {arr.shape}")

    xyz = arr[:3]
    quat_wxyz = arr[3:]
    quat_norm = float(np.linalg.norm(quat_wxyz))
    if quat_norm < 1e-12:
        raise ValueError("Quaternion norm is too small")
    quat_wxyz = quat_wxyz / quat_norm

    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = quat2mat(quat_wxyz).astype(np.float32)
    mat[:3, 3] = xyz.astype(np.float32)
    return mat


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"JSON at {path} is not a dict")
    return data


def load_intrinsics_from_npy(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()
    if not isinstance(data, dict):
        raise TypeError(f"Intrinsics file {path} does not contain a dict")

    intrinsics: Dict[str, np.ndarray] = {}
    for key, value in data.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (3, 3):
            raise ValueError(f"Intrinsic for key {key} has shape {arr.shape}, expected (3,3)")
        intrinsics[str(key)] = arr
    return intrinsics


def choose_intrinsic(intrinsics_dict: Dict[str, np.ndarray], selector: str) -> np.ndarray:
    if selector == "first":
        first_key = sorted(intrinsics_dict.keys())[0]
        return intrinsics_dict[first_key].astype(np.float32)
    if selector == "mean":
        mats = np.stack([intrinsics_dict[k] for k in sorted(intrinsics_dict.keys())], axis=0).astype(np.float32)
        return mats.mean(axis=0).astype(np.float32)
    if selector in intrinsics_dict:
        return intrinsics_dict[selector].astype(np.float32)
    raise KeyError(
        f"Invalid intrinsic selector: {selector}. Available keys={list(intrinsics_dict.keys())}, supported special values=['first', 'mean']"
    )


def build_rise2_calib(
    left_json_path: str,
    right_json_path: str,
    intrinsics_npy_path: str,
    global_serial: str,
    intrinsic_selector: str,
) -> Dict[str, Any]:
    left_data = load_json(left_json_path)
    right_data = load_json(right_json_path)
    intrinsics_dict = load_intrinsics_from_npy(intrinsics_npy_path)

    for side, data in [("left", left_data), ("right", right_data)]:
        if "pose_in_link" not in data:
            raise KeyError(f"Missing key 'pose_in_link' in {side} json")

    intrinsic = choose_intrinsic(intrinsics_dict, intrinsic_selector)

    serial = global_serial
    if serial.startswith("cam_"):
        serial = serial[len("cam_"):]

    calib: Dict[str, Any] = {
        "type": "robot",
        "camera_serials": [serial],
        "camera_serials_global": [serial],
        "camera_serial_inhand_left": "unknown_inhand_left",
        "camera_serial_inhand_right": "unknown_inhand_right",
        "intrinsics": {
            serial: intrinsic.astype(np.float32),
        },
        "camera_to_robot_left": {
            serial: pose_wxyz_to_mat(left_data["pose_in_link"]).astype(np.float32),
        },
        "camera_to_robot_right": {
            serial: pose_wxyz_to_mat(right_data["pose_in_link"]).astype(np.float32),
        },
    }
    return calib


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a RISE2-format calibration npy from left/right extrinsic JSON files.")
    parser.add_argument("--left-json", required=True, help="Path to left result.json")
    parser.add_argument("--right-json", required=True, help="Path to right result.json")
    parser.add_argument("--intrinsics-npy", required=True, help="Path to intrinsics.npy used as intrinsic source")
    parser.add_argument("--global-serial", required=True, help="Global camera serial, e.g. cam_104122060902 or 104122060902")
    parser.add_argument("--intrinsic-selector", default="first", help="Which intrinsic to use from intrinsics.npy: a key, or 'first', or 'mean'")
    parser.add_argument("--output", required=True, help="Output rise2 calibration npy path")
    args = parser.parse_args()

    calib = build_rise2_calib(
        left_json_path=args.left_json,
        right_json_path=args.right_json,
        intrinsics_npy_path=args.intrinsics_npy,
        global_serial=args.global_serial,
        intrinsic_selector=args.intrinsic_selector,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), calib, allow_pickle=True)

    serial = calib["camera_serials_global"][0]
    k = calib["intrinsics"][serial]
    print(f"[ok] wrote: {output_path}")
    print(f"[ok] global serial: {serial}")
    print(f"[ok] intrinsic diag: fx={k[0,0]:.6f}, fy={k[1,1]:.6f}")
    print(f"[ok] intrinsic center: cx={k[0,2]:.6f}, cy={k[1,2]:.6f}")
    print(f"[ok] left translation: {calib['camera_to_robot_left'][serial][:3, 3]}")
    print(f"[ok] right translation: {calib['camera_to_robot_right'][serial][:3, 3]}")


if __name__ == "__main__":
    main()
