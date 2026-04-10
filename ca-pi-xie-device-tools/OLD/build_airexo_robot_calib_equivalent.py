#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from scipy.spatial.transform import Rotation as R

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
AIREXO_ROOT = WORKSPACE_ROOT / "airexo"

ROBOT_LEFT_FLANGE_TO_CAM = np.array(
    [
        [-0.01240050, 0.99905890, 0.04157753, -0.09342833],
        [-0.99978572, -0.01307791, 0.01605635, 0.02158097],
        [0.01658444, -0.04136945, 0.99900651, -0.00380356],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
ROBOT_RIGHT_FLANGE_TO_CAM = np.array(
    [
        [-0.10785065, 0.99415284, 0.00529436, -0.08709522],
        [-0.99405175, -0.10791755, 0.01465851, 0.00088501],
        [0.01514411, -0.00368180, 0.99987853, -0.00513650],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
ROBOT_TCP_TO_FLANGE = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -0.170], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float32,
)
ROBOT_LEFT_TCP_TO_CAM = ROBOT_TCP_TO_FLANGE @ ROBOT_LEFT_FLANGE_TO_CAM
ROBOT_RIGHT_TCP_TO_CAM = ROBOT_TCP_TO_FLANGE @ ROBOT_RIGHT_FLANGE_TO_CAM
ROBOT_LEFT_CAM_TO_TCP = np.linalg.inv(ROBOT_LEFT_TCP_TO_CAM).astype(np.float32)
ROBOT_RIGHT_CAM_TO_TCP = np.linalg.inv(ROBOT_RIGHT_TCP_TO_CAM).astype(np.float32)
ROBOT_PREDEFINED_TRANSFORMATION = np.array(
    [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32
)
ROT_Z_POS_90 = np.array(
    [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float32,
)
ROT_Z_NEG_90 = np.array(
    [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float32,
)

TARGET_MODES = [
    "real_base",
    "predefined_base",
    "predefined_base_rz_pos90",
    "predefined_base_rz_neg90",
    "inverse_real_base",
    "inverse_predefined_base",
]


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def pose7_wxyz_to_mat(pose7: Iterable[float]) -> np.ndarray:
    pose7 = np.asarray(list(pose7), dtype=np.float64).reshape(7)
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


def load_csv_pose7(path: Path, row: int = 0) -> np.ndarray:
    rows: List[np.ndarray] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = np.fromstring(line, sep=",", dtype=np.float64)
            if vals.shape[0] != 7:
                raise ValueError(f"Expected 7 columns in {path}, got {vals.shape[0]}")
            rows.append(vals)
    if len(rows) == 0:
        raise ValueError(f"No valid rows in {path}")
    if row < 0 or row >= len(rows):
        raise IndexError(f"Row index {row} out of range for {path}, n={len(rows)}")
    return rows[row].astype(np.float64)


def choose_intrinsic(intrinsics_all: Dict[str, Any], selector: str, explicit_key: str | None) -> tuple[np.ndarray, str]:
    if explicit_key is not None:
        if explicit_key not in intrinsics_all:
            raise KeyError(f"intrinsic_source_key {explicit_key} not found in keys {list(intrinsics_all.keys())}")
        return np.asarray(intrinsics_all[explicit_key], dtype=np.float32), explicit_key

    if selector == "first":
        key = sorted(intrinsics_all.keys())[0]
        return np.asarray(intrinsics_all[key], dtype=np.float32), key

    if selector == "mean":
        mats = np.stack([np.asarray(v, dtype=np.float32) for v in intrinsics_all.values()], axis=0)
        return mats.mean(axis=0).astype(np.float32), "mean"

    if selector in intrinsics_all:
        return np.asarray(intrinsics_all[selector], dtype=np.float32), selector

    raise KeyError(f"Invalid intrinsic selector {selector}, keys={list(intrinsics_all.keys())}")


def map_json_target_to_real_base(target_json_raw: np.ndarray, target_mode: str) -> tuple[np.ndarray, np.ndarray]:
    # 返回 (json_used, real_base_target)
    if target_mode == "real_base":
        return target_json_raw.astype(np.float64), target_json_raw.astype(np.float64)

    if target_mode == "predefined_base":
        return target_json_raw.astype(np.float64), (target_json_raw @ ROBOT_PREDEFINED_TRANSFORMATION).astype(np.float64)

    if target_mode == "predefined_base_rz_pos90":
        return target_json_raw.astype(np.float64), (target_json_raw @ ROT_Z_POS_90 @ ROBOT_PREDEFINED_TRANSFORMATION).astype(np.float64)

    if target_mode == "predefined_base_rz_neg90":
        return target_json_raw.astype(np.float64), (target_json_raw @ ROT_Z_NEG_90 @ ROBOT_PREDEFINED_TRANSFORMATION).astype(np.float64)

    if target_mode == "inverse_real_base":
        inv_json = invert_T(target_json_raw)
        return inv_json.astype(np.float64), inv_json.astype(np.float64)

    if target_mode == "inverse_predefined_base":
        inv_json = invert_T(target_json_raw)
        return inv_json.astype(np.float64), (inv_json @ ROBOT_PREDEFINED_TRANSFORMATION).astype(np.float64)

    raise ValueError(f"Invalid target_mode {target_mode}, expected one of {TARGET_MODES}")


def build_equivalent_calib(
    left_json: Path,
    right_json: Path,
    intrinsics_npy: Path,
    left_pose_b2e_csv: Path,
    right_pose_b2e_csv: Path,
    global_serial: str,
    inhand_left_serial: str,
    inhand_right_serial: str,
    intrinsic_selector: str,
    intrinsic_source_key: str | None,
    left_row: int,
    right_row: int,
    target_mode: str,
) -> Dict[str, Any]:
    left_target_json_raw = load_json_pose(left_json).astype(np.float64)
    right_target_json_raw = load_json_pose(right_json).astype(np.float64)

    left_json_used, left_target_real = map_json_target_to_real_base(left_target_json_raw, target_mode)
    right_json_used, right_target_real = map_json_target_to_real_base(right_target_json_raw, target_mode)

    left_tcp_pose7 = load_csv_pose7(left_pose_b2e_csv, row=left_row)
    right_tcp_pose7 = load_csv_pose7(right_pose_b2e_csv, row=right_row)
    left_tcp_mat = pose7_wxyz_to_mat(left_tcp_pose7)
    right_tcp_mat = pose7_wxyz_to_mat(right_tcp_pose7)

    intrinsics_all = np.load(str(intrinsics_npy), allow_pickle=True)
    if isinstance(intrinsics_all, np.ndarray) and intrinsics_all.shape == ():
        intrinsics_all = intrinsics_all.item()
    if not isinstance(intrinsics_all, dict):
        raise TypeError("intrinsics.npy must contain a dict")

    intrinsic, intrinsic_key = choose_intrinsic(intrinsics_all, intrinsic_selector, intrinsic_source_key)

    extr_global = np.eye(4, dtype=np.float64)
    extr_left_inhand = invert_T(left_target_real @ left_tcp_mat @ invert_T(ROBOT_LEFT_CAM_TO_TCP))
    extr_right_inhand = invert_T(right_target_real @ right_tcp_mat @ invert_T(ROBOT_RIGHT_CAM_TO_TCP))

    calib = {
        "type": "robot",
        "camera_serials": [global_serial, inhand_left_serial, inhand_right_serial],
        "camera_serials_global": [global_serial],
        "camera_serial_inhand_left": inhand_left_serial,
        "camera_serial_inhand_right": inhand_right_serial,
        "intrinsics": {
            global_serial: intrinsic.astype(np.float32),
            inhand_left_serial: intrinsic.copy().astype(np.float32),
            inhand_right_serial: intrinsic.copy().astype(np.float32),
        },
        "extrinsics": {
            global_serial: extr_global.astype(np.float32),
            inhand_left_serial: extr_left_inhand.astype(np.float32),
            inhand_right_serial: extr_right_inhand.astype(np.float32),
        },
        "robot_left": {
            "tcp_pose": left_tcp_pose7.astype(np.float32),
            "joint_pos": np.zeros((8,), dtype=np.float32),
            "tcp_vel": np.zeros((6,), dtype=np.float32),
            "joint_vel": np.zeros((8,), dtype=np.float32),
            "force_torque": np.zeros((6,), dtype=np.float32),
        },
        "robot_right": {
            "tcp_pose": right_tcp_pose7.astype(np.float32),
            "joint_pos": np.zeros((8,), dtype=np.float32),
            "tcp_vel": np.zeros((6,), dtype=np.float32),
            "joint_vel": np.zeros((8,), dtype=np.float32),
            "force_torque": np.zeros((6,), dtype=np.float32),
        },
        "_debug": {
            "left_target_from_json_raw": left_target_json_raw.astype(np.float32),
            "right_target_from_json_raw": right_target_json_raw.astype(np.float32),
            "left_target_json_used": left_json_used.astype(np.float32),
            "right_target_json_used": right_json_used.astype(np.float32),
            "left_target_used_for_solve": left_target_real.astype(np.float32),
            "right_target_used_for_solve": right_target_real.astype(np.float32),
            "intrinsic_source_key": intrinsic_key,
            "left_pose_b2e_row": left_row,
            "right_pose_b2e_row": right_row,
            "target_mode": target_mode,
        },
    }
    return calib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从左右 JSON + pose_b2e.csv 反构造 Airexo 风格 robot calib")
    parser.add_argument("--left-json", required=True)
    parser.add_argument("--right-json", required=True)
    parser.add_argument("--left-pose-b2e", required=True, help="左臂 pose_b2e.csv")
    parser.add_argument("--right-pose-b2e", required=True, help="右臂 pose_b2e.csv")
    parser.add_argument("--intrinsics-npy", required=True)
    parser.add_argument("--global-serial", required=True)
    parser.add_argument("--inhand-left-serial", required=True)
    parser.add_argument("--inhand-right-serial", required=True)
    parser.add_argument("--intrinsic-selector", default="first", help="first / mean / 具体 key")
    parser.add_argument("--intrinsic-source-key", default=None, help="显式指定 intrinsics.npy 里采用哪个 key；不填则由 selector 决定")
    parser.add_argument("--left-row", type=int, default=0, help="左臂 pose_b2e.csv 采用第几行")
    parser.add_argument("--right-row", type=int, default=0, help="右臂 pose_b2e.csv 采用第几行")
    parser.add_argument("--target-mode", choices=TARGET_MODES, default="real_base", help="按哪种层级解释 result.json")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calib = build_equivalent_calib(
        left_json=Path(args.left_json),
        right_json=Path(args.right_json),
        intrinsics_npy=Path(args.intrinsics_npy),
        left_pose_b2e_csv=Path(args.left_pose_b2e),
        right_pose_b2e_csv=Path(args.right_pose_b2e),
        global_serial=args.global_serial,
        inhand_left_serial=args.inhand_left_serial,
        inhand_right_serial=args.inhand_right_serial,
        intrinsic_selector=args.intrinsic_selector,
        intrinsic_source_key=args.intrinsic_source_key,
        left_row=args.left_row,
        right_row=args.right_row,
        target_mode=args.target_mode,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), calib, allow_pickle=True)

    print(f"[ok] wrote: {output_path}")
    print("[ok] structure keys:", list(calib.keys()))
    print("[ok] extrinsics keys:", list(calib["extrinsics"].keys()))
    print("[ok] intrinsic source key:", calib["_debug"]["intrinsic_source_key"])
    print("[ok] target mode:", calib["_debug"]["target_mode"])
    print("[ok] robot_left.tcp_pose:", calib["robot_left"]["tcp_pose"])
    print("[ok] robot_right.tcp_pose:", calib["robot_right"]["tcp_pose"])


if __name__ == "__main__":
    main()
