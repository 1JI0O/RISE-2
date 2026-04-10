#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def find_nearest_h5_index(h5_timestamps: np.ndarray, camera_timestamp: int) -> int:
    return int(np.argmin(np.abs(h5_timestamps - camera_timestamp)))


def fmt_joint_with_gripper(joint7: np.ndarray, gripper: float) -> str:
    joint7 = np.asarray(joint7, dtype=np.float64).reshape(7)
    vals = list(joint7.astype(np.float64)) + [float(gripper)]
    return ",".join(f"{x:.8f}" for x in vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="从 lowdim.h5 中按相机时间戳提取左右 joint+gripper，并打印 render_json_separate_robot_completed.py 的可执行命令")
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--camera-timestamp", type=int, required=True)
    parser.add_argument("--left-json", required=True)
    parser.add_argument("--right-json", required=True)
    parser.add_argument("--intrinsics-npy", required=True)
    parser.add_argument("--intrinsic-selector", default="first")
    parser.add_argument("--left-arm-suffix", default="062703")
    parser.add_argument("--right-arm-suffix", default="062046")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    h5_path = Path(args.h5_path)
    with h5py.File(h5_path, "r") as f:
        h5_timestamps = np.asarray(f["timestamp"][:], dtype=np.int64)
        idx = find_nearest_h5_index(h5_timestamps, args.camera_timestamp)
        matched_ts = int(h5_timestamps[idx])
        diff = abs(matched_ts - args.camera_timestamp)

        left_joint = np.asarray(f[f"joint_position_rad_{args.left_arm_suffix}"][idx], dtype=np.float64)
        right_joint = np.asarray(f[f"joint_position_rad_{args.right_arm_suffix}"][idx], dtype=np.float64)
        left_gripper = float(np.asarray(f[f"ee_state_{args.left_arm_suffix}"][idx]).reshape(-1)[0])
        right_gripper = float(np.asarray(f[f"ee_state_{args.right_arm_suffix}"][idx]).reshape(-1)[0])

    left_joint_arg = fmt_joint_with_gripper(left_joint, left_gripper)
    right_joint_arg = fmt_joint_with_gripper(right_joint, right_gripper)

    command = (
        "python ca-pi-xie-device-tools/render_json_separate_robot_completed.py "
        f"--left-json {args.left_json} "
        f"--right-json {args.right_json} "
        f"--intrinsics-npy {args.intrinsics_npy} "
        f"--intrinsic-selector {args.intrinsic_selector} "
        f"--left-joint {left_joint_arg} "
        f"--right-joint {right_joint_arg} "
        f"--output-dir {args.output_dir}"
    )

    print(f"[info] camera_timestamp={args.camera_timestamp}")
    print(f"[info] matched_h5_timestamp={matched_ts}")
    print(f"[info] timestamp_diff={diff}")
    print(f"[info] left_joint_arg={left_joint_arg}")
    print(f"[info] right_joint_arg={right_joint_arg}")
    print("\n[command]")
    print(command)


if __name__ == "__main__":
    main()
