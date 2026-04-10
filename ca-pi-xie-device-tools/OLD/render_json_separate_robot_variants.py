#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
AIREXO_ROOT = WORKSPACE_ROOT / "airexo"
for p in [WORKSPACE_ROOT, AIREXO_ROOT]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from omegaconf import OmegaConf

from airexo.helpers.constants import ROBOT_PREDEFINED_TRANSFORMATION
from airexo.helpers.renderer import SeparateRobotRenderer


LEFT_JOINT_CFG_PATH = WORKSPACE_ROOT / "airexo/airexo/configs/joint/left/robot.yaml"
RIGHT_JOINT_CFG_PATH = WORKSPACE_ROOT / "airexo/airexo/configs/joint/right/robot.yaml"
LEFT_URDF = str((WORKSPACE_ROOT / "airexo/airexo/urdf_models/robot/left_robot_inhand.urdf").resolve())
RIGHT_URDF = str((WORKSPACE_ROOT / "airexo/airexo/urdf_models/robot/right_robot_inhand.urdf").resolve())


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


def load_intrinsic(path: Path, selector: str) -> np.ndarray:
    data = np.load(str(path), allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()
    if not isinstance(data, dict):
        raise TypeError(f"Intrinsics file {path} does not contain a dict")
    if selector == "first":
        key = sorted(data.keys())[0]
    elif selector == "mean":
        mats = np.stack([np.asarray(v, dtype=np.float32) for v in data.values()], axis=0)
        return mats.mean(axis=0).astype(np.float32)
    else:
        key = selector
    if selector != "mean":
        if key not in data:
            raise KeyError(f"Intrinsic selector {selector} not in keys {list(data.keys())}")
        return np.asarray(data[key], dtype=np.float32)
    raise AssertionError("unreachable")


def build_variant(T: np.ndarray, mode: str) -> np.ndarray:
    predefined_inv = invert_T(np.asarray(ROBOT_PREDEFINED_TRANSFORMATION, dtype=np.float64))
    if mode == "raw":
        return T.astype(np.float64)
    if mode == "raw_predef_inv":
        return (T @ predefined_inv).astype(np.float64)
    if mode == "inv_raw":
        return invert_T(T).astype(np.float64)
    if mode == "inv_raw_predef_inv":
        return (invert_T(T) @ predefined_inv).astype(np.float64)
    raise ValueError(f"Unknown mode {mode}")


def save_matrix_txt(path: Path, title: str, T: np.ndarray) -> None:
    arr_str = np.array2string(T, formatter={"float_kind": lambda x: f"{x: .6f}"})
    path.write_text(f"[{title}]\n{arr_str}\n", encoding="utf-8")


def render_case(
    out_dir: Path,
    intrinsic: np.ndarray,
    left_joint: np.ndarray,
    right_joint: np.ndarray,
    left_cam_to_base: np.ndarray,
    right_cam_to_base: np.ndarray,
    width: int,
    height: int,
) -> None:
    left_joint_cfgs = OmegaConf.load(str(LEFT_JOINT_CFG_PATH))
    right_joint_cfgs = OmegaConf.load(str(RIGHT_JOINT_CFG_PATH))

    renderer = SeparateRobotRenderer(
        left_joint_cfgs=left_joint_cfgs,
        right_joint_cfgs=right_joint_cfgs,
        cam_to_left_base=left_cam_to_base.astype(np.float32),
        cam_to_right_base=right_cam_to_base.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
        width=width,
        height=height,
        near_plane=0.01,
        far_plane=100.0,
        urdf_file={"left": LEFT_URDF, "right": RIGHT_URDF},
    )
    renderer.update_joints(left_joint.astype(np.float32), right_joint.astype(np.float32))
    color = renderer.render_image()
    depth = renderer.render_depth()
    mask = renderer.render_mask(depth=depth)
    del renderer

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "color.png"), color[:, :, ::-1])
    cv2.imwrite(str(out_dir / "mask.png"), mask)
    depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(out_dir / "depth.png"), depth_mm)


def main() -> None:
    parser = argparse.ArgumentParser(description="直接用左右 JSON 枚举 SeparateRobotRenderer 的 cam_to_base 候选写法并输出渲染结果")
    parser.add_argument("--left-json", required=True)
    parser.add_argument("--right-json", required=True)
    parser.add_argument("--intrinsics-npy", required=True)
    parser.add_argument("--intrinsic-selector", default="first")
    parser.add_argument("--left-joint", required=True, help="8 维逗号分隔，单位 rad，例如 1.0,-2.2,...")
    parser.add_argument("--right-joint", required=True, help="8 维逗号分隔，单位 rad，例如 1.3,-2.1,...")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    left_json = load_json_pose(Path(args.left_json)).astype(np.float64)
    right_json = load_json_pose(Path(args.right_json)).astype(np.float64)
    intrinsic = load_intrinsic(Path(args.intrinsics_npy), args.intrinsic_selector)

    left_joint = np.fromstring(args.left_joint, sep=",", dtype=np.float64)
    right_joint = np.fromstring(args.right_joint, sep=",", dtype=np.float64)
    if left_joint.shape != (8,):
        raise ValueError(f"left_joint must have 8 values, got {left_joint.shape}")
    if right_joint.shape != (8,):
        raise ValueError(f"right_joint must have 8 values, got {right_joint.shape}")

    modes = ["raw", "raw_predef_inv", "inv_raw", "inv_raw_predef_inv"]
    variants: Dict[str, np.ndarray] = {f"left_{m}": build_variant(left_json, m) for m in modes}
    variants.update({f"right_{m}": build_variant(right_json, m) for m in modes})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combo_list: Tuple[Tuple[str, str], ...] = (
        ("raw", "raw"),
        ("raw_predef_inv", "raw_predef_inv"),
        ("inv_raw", "inv_raw"),
        ("inv_raw_predef_inv", "inv_raw_predef_inv"),
        ("raw", "raw_predef_inv"),
        ("raw_predef_inv", "raw"),
    )

    for key, T in variants.items():
        save_matrix_txt(output_dir / f"{key}.txt", key, T)

    summary_lines = []
    for left_mode, right_mode in combo_list:
        case_name = f"left_{left_mode}__right_{right_mode}"
        case_dir = output_dir / case_name
        left_T = build_variant(left_json, left_mode)
        right_T = build_variant(right_json, right_mode)
        render_case(
            out_dir=case_dir,
            intrinsic=intrinsic,
            left_joint=left_joint,
            right_joint=right_joint,
            left_cam_to_base=left_T,
            right_cam_to_base=right_T,
            width=args.width,
            height=args.height,
        )
        summary_lines.append(case_name)

    (output_dir / "README.txt").write_text(
        "Rendered variants for SeparateRobotRenderer\n"
        "Modes:\n"
        "- raw: json 原矩阵\n"
        "- raw_predef_inv: json @ inv(ROBOT_PREDEFINED_TRANSFORMATION)\n"
        "- inv_raw: inv(json)\n"
        "- inv_raw_predef_inv: inv(json) @ inv(ROBOT_PREDEFINED_TRANSFORMATION)\n\n"
        "Rendered cases:\n- " + "\n- ".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote renders to: {output_dir}")
    for case_name in summary_lines:
        print(f"[ok] case: {case_name}")


if __name__ == "__main__":
    main()
