#!/usr/bin/env python3
"""
Build a single-arm RISE2 calibration `.npy` from one JSON file.

Expected output schema matches `dataset/projector.py::SingleArmProjector`:
- `camera_serials`
- `camera_serials_global`
- `camera_serial_inhand`
- `intrinsics`
- `camera_to_robot`

Input JSON is expected to contain at least:
- `pose_in_link`: [x, y, z, qw, qx, qy, qz]
- `cam_serial`
- `intrinsics`: 3x3 intrinsic matrix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def quat_wxyz_to_mat3(quat_wxyz: Any) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float32).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"Expected quaternion [w,x,y,z], got shape {q.shape}")

    w, x, y, z = q
    n = float(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError("Quaternion norm is too small")

    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def pose_wxyz_to_mat(pose: Any) -> np.ndarray:
    arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 7:
        raise ValueError(f"Expected pose with 7 values [x,y,z,qw,qx,qy,qz], got shape {arr.shape}")

    xyz = arr[:3]
    quat_wxyz = arr[3:]
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = quat_wxyz_to_mat3(quat_wxyz)
    mat[:3, 3] = xyz.astype(np.float32)
    return mat


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"JSON at {path} is not a dict")
    return data


def build_rise2_single_calib(json_path: str, global_serial: str | None, inhand_serial: str) -> Dict[str, Any]:
    data = load_json(json_path)

    if "pose_in_link" not in data:
        raise KeyError("Missing key 'pose_in_link' in json")
    if "intrinsics" not in data:
        raise KeyError("Missing key 'intrinsics' in json")

    serial = global_serial if global_serial is not None else str(data.get("cam_serial", "")).strip()
    if serial == "":
        raise ValueError("Global serial is empty. Provide --global-serial or make sure json contains cam_serial")
    if serial.startswith("cam_"):
        serial = serial[len("cam_"):]

    intrinsic = np.asarray(data["intrinsics"], dtype=np.float32)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Intrinsic matrix has shape {intrinsic.shape}, expected (3,3)")

    calib: Dict[str, Any] = {
        "type": "robot",
        "camera_serials": [serial],
        "camera_serials_global": [serial],
        "camera_serial_inhand": str(inhand_serial),
        "intrinsics": {
            serial: intrinsic.astype(np.float32),
        },
        "camera_to_robot": {
            serial: pose_wxyz_to_mat(data["pose_in_link"]).astype(np.float32),
        },
    }
    return calib


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single-arm RISE2-format calibration npy from one result.json file.")
    parser.add_argument("--json", required=True, help="Path to result.json")
    parser.add_argument("--global-serial", default=None, help="Override global camera serial, e.g. cam_104422070117 or 104422070117")
    parser.add_argument("--inhand-serial", default="unknown_inhand", help="Placeholder in-hand serial required by SingleArmProjector")
    parser.add_argument("--output", required=True, help="Output rise2 calibration npy path")
    args = parser.parse_args()

    calib = build_rise2_single_calib(
        json_path=args.json,
        global_serial=args.global_serial,
        inhand_serial=args.inhand_serial,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), calib, allow_pickle=True)

    serial = calib["camera_serials_global"][0]
    k = calib["intrinsics"][serial]
    t = calib["camera_to_robot"][serial][:3, 3]
    print(f"[ok] wrote: {output_path}")
    print(f"[ok] global serial: {serial}")
    print(f"[ok] inhand serial placeholder: {calib['camera_serial_inhand']}")
    print(f"[ok] intrinsic diag: fx={k[0,0]:.6f}, fy={k[1,1]:.6f}")
    print(f"[ok] intrinsic center: cx={k[0,2]:.6f}, cy={k[1,2]:.6f}")
    print(f"[ok] camera_to_robot translation: {t}")


if __name__ == "__main__":
    main()
