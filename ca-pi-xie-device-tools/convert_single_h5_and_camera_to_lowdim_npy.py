#!/usr/bin/env python3
"""
Convert per-scene single-arm `lowdim.h5` into RISE2 training `.npy` files.

This script writes files in place under:
  <task_dir>/<split>/scene_xxxx/lowdim/<image_timestamp>.npy

Each output `.npy` is a dict compatible with `dataset/data_utils.py::load_action`
for `robot_type == "single"`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, cast

import h5py
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert single-arm h5 lowdim files to RISE2 npy files in place."
    )
    parser.add_argument(
        "--task-dir",
        required=True,
        help="Dataset root containing train/scene_* directories",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split directory under task-dir, default: train",
    )
    parser.add_argument(
        "--camera-subdir",
        required=True,
        help="Camera directory name, e.g. cam_104422070117",
    )
    parser.add_argument(
        "--arm-suffix", required=True, help="Suffix used in h5 keys, e.g. 062770"
    )
    parser.add_argument(
        "--gripper-source",
        default="ee_state",
        choices=["ee_state", "ee_command"],
        help="Which h5 dataset to use for gripper values",
    )
    parser.add_argument(
        "--skip-pedal-value",
        type=float,
        default=None,
        help="If set, skip samples whose nearest h5 pedal_0 equals this value",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing npy files",
    )
    parser.add_argument(
        "--max-timestamp-diff-ms",
        type=int,
        default=20,
        help="Warn when nearest h5 timestamp differs from image timestamp by more than this value in ms",
    )
    parser.add_argument(
        "--drop-large-timestamp-diff",
        action="store_true",
        help="Skip writing samples whose nearest h5 timestamp differs by more than --max-timestamp-diff-ms",
    )
    return parser.parse_args()


def build_robot_vector(tcp_pose: np.ndarray) -> np.ndarray:
    robot = np.zeros(33, dtype=np.float32)
    robot[:7] = np.asarray(tcp_pose, dtype=np.float32)[:7]
    return robot


def build_gripper_vector(value: float) -> np.ndarray:
    v = np.float32(value)
    return np.array([v, v], dtype=np.float32)


def find_nearest_h5_index(h5_timestamps: np.ndarray, camera_timestamp: int) -> int:
    insert_idx = int(np.searchsorted(h5_timestamps, camera_timestamp))
    if insert_idx <= 0:
        return 0
    if insert_idx >= len(h5_timestamps):
        return len(h5_timestamps) - 1

    left_idx = insert_idx - 1
    right_idx = insert_idx
    left_diff = abs(int(h5_timestamps[left_idx]) - int(camera_timestamp))
    right_diff = abs(int(h5_timestamps[right_idx]) - int(camera_timestamp))
    if left_diff <= right_diff:
        return left_idx
    return right_idx


def list_scene_dirs(split_dir: Path) -> List[Path]:
    return sorted(
        [p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("scene_")]
    )


def list_png_timestamps(color_dir: Path) -> List[int]:
    timestamps = []
    for path in sorted(color_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".png":
            timestamps.append(int(path.stem))
    return timestamps


def process_scene(
    scene_dir: Path,
    camera_subdir: str,
    arm_suffix: str,
    gripper_source: str,
    skip_pedal_value: float | None,
    overwrite: bool,
    max_timestamp_diff_ms: int,
    drop_large_timestamp_diff: bool,
) -> dict | None:
    camera_dir = scene_dir / camera_subdir
    color_dir = camera_dir / "color"
    depth_dir = camera_dir / "depth"
    h5_path = scene_dir / "lowdim" / "lowdim.h5"
    output_lowdim_dir = scene_dir / "lowdim"

    if not color_dir.is_dir():
        print(f"[skip scene] missing color dir: {color_dir}")
        return None
    if not depth_dir.is_dir():
        print(f"[skip scene] missing depth dir: {depth_dir}")
        return None
    if not h5_path.is_file():
        print(f"[skip scene] missing h5 file: {h5_path}")
        return None

    ensure_dir(output_lowdim_dir)
    image_timestamps = list_png_timestamps(color_dir)
    if len(image_timestamps) == 0:
        print(f"[skip scene] empty color dir: {color_dir}")
        return None

    with h5py.File(h5_path, "r") as f:
        h5_timestamps = f["timestamp"][:].astype(np.int64)
        tcp_dataset = f[f"tcp_pose_{arm_suffix}"]
        gripper_dataset = f[f"{gripper_source}_{arm_suffix}"]
        pedal_dataset = (
            f["pedal_0"][:].reshape(-1).astype(np.float32) if "pedal_0" in f else None
        )

        kept_count = 0
        skipped_pedal_count = 0
        skipped_existing_count = 0
        skipped_large_delta_count = 0
        max_timestamp_diff = 0
        large_delta_count = 0
        large_delta_examples = []

        for camera_timestamp in image_timestamps:
            output_lowdim_file = output_lowdim_dir / f"{camera_timestamp}.npy"
            if output_lowdim_file.exists() and not overwrite:
                skipped_existing_count += 1
                continue

            nearest_h5_index = find_nearest_h5_index(h5_timestamps, camera_timestamp)
            nearest_h5_timestamp = int(h5_timestamps[nearest_h5_index])
            timestamp_diff = abs(nearest_h5_timestamp - camera_timestamp)
            max_timestamp_diff = max(max_timestamp_diff, timestamp_diff)
            if timestamp_diff > max_timestamp_diff_ms:
                large_delta_count += 1
                if len(large_delta_examples) < 3:
                    large_delta_examples.append(
                        (camera_timestamp, nearest_h5_timestamp, int(timestamp_diff))
                    )
                if drop_large_timestamp_diff:
                    skipped_large_delta_count += 1
                    continue

            if pedal_dataset is not None and skip_pedal_value is not None:
                pedal_value = float(pedal_dataset[nearest_h5_index])
                if np.isclose(pedal_value, skip_pedal_value):
                    skipped_pedal_count += 1
                    continue

            tcp_pose = np.asarray(tcp_dataset[nearest_h5_index], dtype=np.float32)
            gripper_value = float(
                np.asarray(gripper_dataset[nearest_h5_index]).reshape(-1)[0]
            )
            lowdim_dict = {
                "robot": build_robot_vector(tcp_pose),
                "gripper": build_gripper_vector(gripper_value),
            }
            np.save(output_lowdim_file, cast(Any, lowdim_dict), allow_pickle=True)
            kept_count += 1

    print(
        "[scene] {} images={} wrote={} skipped_existing={} skipped_pedal={} skipped_large_delta={} max_ts_diff={} large_delta_count={}".format(
            scene_dir.name,
            len(image_timestamps),
            kept_count,
            skipped_existing_count,
            skipped_pedal_count,
            skipped_large_delta_count,
            max_timestamp_diff,
            large_delta_count,
        )
    )
    for camera_timestamp, nearest_h5_timestamp, timestamp_diff in large_delta_examples:
        print(
            "[scene warning] {} image_ts={} matched_h5_ts={} diff_ms={}".format(
                scene_dir.name,
                camera_timestamp,
                nearest_h5_timestamp,
                timestamp_diff,
            )
        )
    return {
        "scene": scene_dir.name,
        "images": len(image_timestamps),
        "wrote": kept_count,
        "skipped_existing": skipped_existing_count,
        "skipped_pedal": skipped_pedal_count,
        "skipped_large_delta": skipped_large_delta_count,
        "large_delta_count": large_delta_count,
        "max_timestamp_diff": max_timestamp_diff,
    }


def main() -> None:
    args = parse_args()
    task_dir = Path(args.task_dir)
    split_dir = task_dir / args.split
    scene_dirs = list_scene_dirs(split_dir)

    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split dir does not exist: {split_dir}")

    print(f"task_dir={task_dir}")
    print(f"split_dir={split_dir}")
    print(f"camera_subdir={args.camera_subdir}")
    print(f"arm_suffix={args.arm_suffix}")
    print(f"gripper_source={args.gripper_source}")
    print(f"skip_pedal_value={args.skip_pedal_value}")
    print(f"overwrite={args.overwrite}")
    print(f"max_timestamp_diff_ms={args.max_timestamp_diff_ms}")
    print(f"drop_large_timestamp_diff={args.drop_large_timestamp_diff}")
    print(f"scene_count={len(scene_dirs)}")

    all_results = []
    for scene_dir in scene_dirs:
        result = process_scene(
            scene_dir=scene_dir,
            camera_subdir=args.camera_subdir,
            arm_suffix=args.arm_suffix,
            gripper_source=args.gripper_source,
            skip_pedal_value=args.skip_pedal_value,
            overwrite=args.overwrite,
            max_timestamp_diff_ms=args.max_timestamp_diff_ms,
            drop_large_timestamp_diff=args.drop_large_timestamp_diff,
        )
        if result is not None:
            all_results.append(result)

    total_images = sum(x["images"] for x in all_results)
    total_wrote = sum(x["wrote"] for x in all_results)
    total_skipped_existing = sum(x["skipped_existing"] for x in all_results)
    total_skipped_pedal = sum(x["skipped_pedal"] for x in all_results)
    total_skipped_large_delta = sum(x["skipped_large_delta"] for x in all_results)
    total_large_delta = sum(x["large_delta_count"] for x in all_results)
    worst_scene = (
        None
        if len(all_results) == 0
        else max(all_results, key=lambda x: x["max_timestamp_diff"])
    )
    print("[all scenes summary]")
    print(f"processed_scene_count={len(all_results)}")
    print(f"total_images={total_images}")
    print(f"total_wrote={total_wrote}")
    print(f"total_skipped_existing={total_skipped_existing}")
    print(f"total_skipped_pedal={total_skipped_pedal}")
    print(f"total_skipped_large_delta={total_skipped_large_delta}")
    print(f"total_large_delta_count={total_large_delta}")
    if worst_scene is not None:
        print(
            "worst_scene={} worst_max_ts_diff={}".format(
                worst_scene["scene"],
                worst_scene["max_timestamp_diff"],
            )
        )


if __name__ == "__main__":
    main()
