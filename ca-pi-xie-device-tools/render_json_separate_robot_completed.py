#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
AIREXO_ROOT = WORKSPACE_ROOT / "airexo"
for p in [WORKSPACE_ROOT, AIREXO_ROOT]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from airexo.helpers.constants import ROBOT_PREDEFINED_TRANSFORMATION
from airexo.helpers.renderer import SeparateRobotRenderer


LEFT_JOINT_CFGS = SimpleNamespace(num_joints=8, num_robot_joints=7)
RIGHT_JOINT_CFGS = SimpleNamespace(num_joints=8, num_robot_joints=7)
LEFT_URDF = str((WORKSPACE_ROOT / "airexo/airexo/urdf_models/robot/left_robot_inhand.urdf").resolve())
RIGHT_URDF = str((WORKSPACE_ROOT / "airexo/airexo/urdf_models/robot/right_robot_inhand.urdf").resolve())


CAM_BASE_MODE_CHOICES = [
    "base_to_cam__predef",
    "base_to_cam__raw",
    "cam_to_base__predef",
    "cam_to_base__raw",
]


PROBE_JOINT_MODES = [
    "identity",
    "flip_all_7",
    "flip_set_1_3_5_7",
    "flip_set_2_4_6",
    "flip_1",
    "flip_2",
    "flip_3",
    "flip_4",
    "flip_5",
    "flip_6",
    "flip_7",
]


PROBE_CAM_MODES = [
    "base_to_cam__predef",
    "base_to_cam__raw",
    "cam_to_base__predef",
    "cam_to_base__raw",
]


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
        return np.asarray(data[key], dtype=np.float32)
    if selector == "mean":
        mats = np.stack([np.asarray(v, dtype=np.float32) for v in data.values()], axis=0)
        return mats.mean(axis=0).astype(np.float32)
    if selector not in data:
        raise KeyError(f"Intrinsic selector {selector} not in keys {list(data.keys())}")
    return np.asarray(data[selector], dtype=np.float32)


def cam_base_from_json(T_json: np.ndarray, mode: str) -> np.ndarray:
    T_json = np.asarray(T_json, dtype=np.float64)
    T_predef_inv = invert_T(np.asarray(ROBOT_PREDEFINED_TRANSFORMATION, dtype=np.float64))

    if mode == "base_to_cam__predef":
        return invert_T(T_json) @ T_predef_inv
    if mode == "base_to_cam__raw":
        return invert_T(T_json)
    if mode == "cam_to_base__predef":
        return T_json @ T_predef_inv
    if mode == "cam_to_base__raw":
        return T_json
    raise ValueError(f"Unknown cam-base mode: {mode}")


def adapt_joint(joint: np.ndarray, mode: str) -> np.ndarray:
    out = np.asarray(joint, dtype=np.float64).copy()
    if out.shape != (8,):
        raise ValueError(f"Joint must be shape (8,), got {out.shape}")

    if mode == "identity":
        return out
    if mode == "flip_all_7":
        out[:7] *= -1.0
        return out

    if mode.startswith("flip_set_"):
        set_str = mode[len("flip_set_"):]
        indices = [int(x) for x in set_str.split("_") if x.strip()]
        for idx in indices:
            if idx < 1 or idx > 7:
                raise ValueError("flip_set_* only allows indices in [1,7]")
            out[idx - 1] *= -1.0
        return out

    if mode.startswith("flip_"):
        idx_str = mode[len("flip_"):]
        idx = int(idx_str)
        if idx < 1 or idx > 7:
            raise ValueError("flip_<i> requires i in [1,7]")
        out[idx - 1] *= -1.0
        return out

    raise ValueError(f"Unknown joint-adapter mode: {mode}")


def save_matrix(path: Path, title: str, T: np.ndarray) -> None:
    arr_str = np.array2string(T, formatter={"float_kind": lambda x: f"{x: .6f}"})
    path.write_text(f"[{title}]\n{arr_str}\n", encoding="utf-8")


def draw_debug_overlay(color: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = color.copy()
    h, w = out.shape[:2]

    border_color = np.array([255, 0, 0], dtype=np.uint8)
    out[:4, :, :] = border_color
    out[-4:, :, :] = border_color
    out[:, :4, :] = border_color
    out[:, -4:, :] = border_color

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return out

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    box_color = np.array([0, 255, 0], dtype=np.uint8)
    out[max(0, y0 - 2):min(h, y0 + 3), x0:x1 + 1, :] = box_color
    out[max(0, y1 - 2):min(h, y1 + 3), x0:x1 + 1, :] = box_color
    out[y0:y1 + 1, max(0, x0 - 2):min(w, x0 + 3), :] = box_color
    out[y0:y1 + 1, max(0, x1 - 2):min(w, x1 + 3), :] = box_color

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    cross_color = np.array([255, 255, 0], dtype=np.uint8)
    out[max(0, cy - 2):min(h, cy + 3), max(0, cx - 20):min(w, cx + 21), :] = cross_color
    out[max(0, cy - 20):min(h, cy + 21), max(0, cx - 2):min(w, cx + 3), :] = cross_color

    return out


def parse_joint(s: str, name: str) -> np.ndarray:
    arr = np.fromstring(s, sep=",", dtype=np.float64)
    if arr.shape != (8,):
        raise ValueError(f"{name} must have 8 comma-separated values, got shape {arr.shape}")
    return arr


def iter_joint_modes(base_mode: str, probe: bool) -> List[str]:
    if not probe:
        return [base_mode]
    return PROBE_JOINT_MODES


def iter_cam_modes(base_mode: str, probe: bool) -> List[str]:
    if not probe:
        return [base_mode]
    return PROBE_CAM_MODES


def render_one(
    intrinsic: np.ndarray,
    left_joint: np.ndarray,
    right_joint: np.ndarray,
    left_cam_to_base: np.ndarray,
    right_cam_to_base: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    renderer = SeparateRobotRenderer(
        left_joint_cfgs=LEFT_JOINT_CFGS,
        right_joint_cfgs=RIGHT_JOINT_CFGS,
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
    return color, depth, mask


def render_and_save_mode(
    mode_path: Path,
    mode_id: int,
    cam_mode: str,
    left_adapter: str,
    right_adapter: str,
    intrinsic: np.ndarray,
    left_joint_raw: np.ndarray,
    right_joint_raw: np.ndarray,
    left_joint: np.ndarray,
    right_joint: np.ndarray,
    left_json: np.ndarray,
    right_json: np.ndarray,
    left_cam_to_base: np.ndarray,
    right_cam_to_base: np.ndarray,
    width: int,
    height: int,
) -> int:
    color, depth, mask = render_one(
        intrinsic=intrinsic,
        left_joint=left_joint,
        right_joint=right_joint,
        left_cam_to_base=left_cam_to_base,
        right_cam_to_base=right_cam_to_base,
        width=width,
        height=height,
    )
    color_debug = draw_debug_overlay(color, mask)
    mask_pixels = int(np.count_nonzero(mask))
    finite_depth = depth[np.isfinite(depth)]
    if finite_depth.size > 0:
        depth_min = float(finite_depth.min())
        depth_max = float(finite_depth.max())
    else:
        depth_min = float("nan")
        depth_max = float("nan")

    Image.fromarray(color).save(mode_path / "color.png")
    Image.fromarray(color_debug).save(mode_path / "color_debug.png")
    Image.fromarray(mask).save(mode_path / "mask.png")
    Image.fromarray(np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)).save(mode_path / "depth.png")

    save_matrix(mode_path / "left_json.txt", "left_json", left_json)
    save_matrix(mode_path / "right_json.txt", "right_json", right_json)
    save_matrix(mode_path / "left_cam_to_renderer_base.txt", "left_cam_to_renderer_base", left_cam_to_base)
    save_matrix(mode_path / "right_cam_to_renderer_base.txt", "right_cam_to_renderer_base", right_cam_to_base)
    save_matrix(
        mode_path / "robot_predefined_inv.txt",
        "robot_predefined_inv",
        invert_T(np.asarray(ROBOT_PREDEFINED_TRANSFORMATION, dtype=np.float64)),
    )

    (mode_path / "joint_debug.txt").write_text(
        "left_raw=" + np.array2string(left_joint_raw, precision=6) + "\n"
        + "right_raw=" + np.array2string(right_joint_raw, precision=6) + "\n"
        + "left_used=" + np.array2string(left_joint, precision=6) + "\n"
        + "right_used=" + np.array2string(right_joint, precision=6) + "\n",
        encoding="utf-8",
    )

    (mode_path / "README.txt").write_text(
        "JSON -> SeparateRobotRenderer diagnostic render\n\n"
        f"mode_id: {mode_id}\n"
        f"cam_mode: {cam_mode}\n"
        f"left_joint_adapter: {left_adapter}\n"
        f"right_joint_adapter: {right_adapter}\n"
        f"mask_pixels: {mask_pixels}\n"
        f"depth_min: {depth_min}\n"
        f"depth_max: {depth_max}\n",
        encoding="utf-8",
    )
    return mask_pixels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="渲染 JSON+joint 到 SeparateRobotRenderer；支持 joint/标定语义探测模式"
    )
    parser.add_argument("--left-json", required=True)
    parser.add_argument("--right-json", required=True)
    parser.add_argument("--intrinsics-npy", required=True)
    parser.add_argument("--intrinsic-selector", default="first")
    parser.add_argument("--left-joint", required=True)
    parser.add_argument("--right-joint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--cam-base-mode",
        default="base_to_cam__predef",
        choices=CAM_BASE_MODE_CHOICES,
        help="JSON 矩阵解释与是否补 predefined 的组合模式",
    )
    parser.add_argument(
        "--left-joint-adapter",
        default="identity",
        help="左臂关节适配模式，如 identity/flip_all_7/flip_1/flip_set_1_3_5_7",
    )
    parser.add_argument(
        "--right-joint-adapter",
        default="identity",
        help="右臂关节适配模式，如 identity/flip_all_7/flip_1/flip_set_1_3_5_7",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="开启后自动枚举 cam-base 与 joint 适配组合并批量导出",
    )
    args = parser.parse_args()

    left_json = load_json_pose(Path(args.left_json)).astype(np.float64)
    right_json = load_json_pose(Path(args.right_json)).astype(np.float64)
    intrinsic = load_intrinsic(Path(args.intrinsics_npy), args.intrinsic_selector)
    left_joint_raw = parse_joint(args.left_joint, "left_joint")
    right_joint_raw = parse_joint(args.right_joint, "right_joint")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[str] = []
    rows.append("mode_id,cam_mode,left_adapter,right_adapter,mask_pixels")

    if not args.probe:
        cam_mode = args.cam_base_mode
        left_adapter = args.left_joint_adapter
        right_adapter = args.right_joint_adapter
        left_cam_to_base = cam_base_from_json(left_json, cam_mode)
        right_cam_to_base = cam_base_from_json(right_json, cam_mode)
        left_joint = adapt_joint(left_joint_raw, left_adapter)
        right_joint = adapt_joint(right_joint_raw, right_adapter)

        mask_pixels = render_and_save_mode(
            mode_path=out_dir,
            mode_id=0,
            cam_mode=cam_mode,
            left_adapter=left_adapter,
            right_adapter=right_adapter,
            intrinsic=intrinsic,
            left_joint_raw=left_joint_raw,
            right_joint_raw=right_joint_raw,
            left_joint=left_joint,
            right_joint=right_joint,
            left_json=left_json,
            right_json=right_json,
            left_cam_to_base=left_cam_to_base,
            right_cam_to_base=right_cam_to_base,
            width=args.width,
            height=args.height,
        )
        rows.append(f"0,{cam_mode},{left_adapter},{right_adapter},{mask_pixels}")
        (out_dir / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"[ok] wrote render to: {out_dir}")
        print(f"[ok] summary: {out_dir / 'summary.csv'}")
        return

    cam_modes = iter_cam_modes(args.cam_base_mode, True)
    left_joint_modes = iter_joint_modes(args.left_joint_adapter, True)
    right_joint_modes = iter_joint_modes(args.right_joint_adapter, True)

    mode_id = 0
    for cam_mode in cam_modes:
        left_cam_to_base = cam_base_from_json(left_json, cam_mode)
        right_cam_to_base = cam_base_from_json(right_json, cam_mode)

        for left_adapter in left_joint_modes:
            left_joint = adapt_joint(left_joint_raw, left_adapter)

            for right_adapter in right_joint_modes:
                right_joint = adapt_joint(right_joint_raw, right_adapter)
                mode_name = f"m{mode_id:03d}__cam-{cam_mode}__l-{left_adapter}__r-{right_adapter}"
                mode_path = out_dir / mode_name
                mode_path.mkdir(parents=True, exist_ok=True)

                mask_pixels = render_and_save_mode(
                    mode_path=mode_path,
                    mode_id=mode_id,
                    cam_mode=cam_mode,
                    left_adapter=left_adapter,
                    right_adapter=right_adapter,
                    intrinsic=intrinsic,
                    left_joint_raw=left_joint_raw,
                    right_joint_raw=right_joint_raw,
                    left_joint=left_joint,
                    right_joint=right_joint,
                    left_json=left_json,
                    right_json=right_json,
                    left_cam_to_base=left_cam_to_base,
                    right_cam_to_base=right_cam_to_base,
                    width=args.width,
                    height=args.height,
                )

                rows.append(f"{mode_id},{cam_mode},{left_adapter},{right_adapter},{mask_pixels}")
                mode_id += 1

    (out_dir / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    (out_dir / "README.txt").write_text(
        "render_json_separate_robot_completed.py diagnostic outputs\n\n"
        f"probe={args.probe}\n"
        f"num_modes={mode_id}\n"
        "Each mode folder contains color/depth/mask and debug transforms.\n"
        "Use color_debug.png for quick visual comparison.\n",
        encoding="utf-8",
    )

    print(f"[ok] wrote renders to: {out_dir}")
    print(f"[ok] total modes: {mode_id}")
    print(f"[ok] summary: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
