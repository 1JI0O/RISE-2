#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
AIREXO_ROOT = WORKSPACE_ROOT / "airexo"
for p in [WORKSPACE_ROOT, AIREXO_ROOT]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from airexo.helpers.constants import (
    ROBOT_LEFT_REAL_BASE_TO_REAL_BASE,
    ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE,
    ROBOT_PREDEFINED_TRANSFORMATION,
)
from airexo.helpers.rotation import average_xyz_rot_quat


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
    parser = argparse.ArgumentParser(description="只依赖 JSON，检查哪些矩阵假设最像 Airexo renderer 需要的 cam_to_base")
    parser.add_argument("--left-json", required=True)
    parser.add_argument("--right-json", required=True)
    parser.add_argument("--print-matrices", action="store_true")
    args = parser.parse_args()

    left_real = load_json_pose(Path(args.left_json)).astype(np.float64)
    right_real = load_json_pose(Path(args.right_json)).astype(np.float64)

    left_real_inv = invert_T(left_real)
    right_real_inv = invert_T(right_real)

    overall_real_from_left = left_real @ np.asarray(ROBOT_LEFT_REAL_BASE_TO_REAL_BASE, dtype=np.float64)
    overall_real_from_right = right_real @ np.asarray(ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE, dtype=np.float64)
    overall_real_avg = np.asarray(
        average_xyz_rot_quat(overall_real_from_left, overall_real_from_right, rotation_rep="matrix"),
        dtype=np.float64,
    )

    predefined_from_left = overall_real_from_left @ invert_T(np.asarray(ROBOT_PREDEFINED_TRANSFORMATION, dtype=np.float64))
    predefined_from_right = overall_real_from_right @ invert_T(np.asarray(ROBOT_PREDEFINED_TRANSFORMATION, dtype=np.float64))
    predefined_avg = overall_real_avg @ invert_T(np.asarray(ROBOT_PREDEFINED_TRANSFORMATION, dtype=np.float64))

    print("[info] 基础假设：JSON 语义先按 camera -> left/right real base 处理")

    compare_pair("overall_real_from_left", overall_real_from_left, "overall_real_from_right", overall_real_from_right)
    compare_pair("predefined_from_left", predefined_from_left, "predefined_from_right", predefined_from_right)
    compare_pair("left_real", left_real, "right_real", right_real)
    compare_pair("left_real_inv", left_real_inv, "right_real_inv", right_real_inv)

    print("\n[hint]")
    print("- 如果 overall_real_from_left 与 overall_real_from_right 很接近，说明左右 JSON 在提升到 shared real base 后是自洽的。")
    print("- 如果这一项误差很大，说明问题先出在 JSON 本身、左右结果不一致，或左右基座定义和 Airexo 常量不一致。")
    print("- predefined_from_left / predefined_from_right 则对应更接近 RobotRenderer 输入层级的候选 cam_to_base。")
    print("- 若你的 notebook 直接拿 left_real/right_real 去渲染，而 predefined_* 明显更合理，说明少做了 predefined base 对齐。")
    print("- 若 left_real 与 right_real 的逆矩阵彼此更接近，也提示 pose_in_link 方向可能理解反了。")

    if args.print_matrices:
        print_matrix("left_real(json)", left_real)
        print_matrix("right_real(json)", right_real)
        print_matrix("overall_real_from_left", overall_real_from_left)
        print_matrix("overall_real_from_right", overall_real_from_right)
        print_matrix("overall_real_avg", overall_real_avg)
        print_matrix("predefined_from_left", predefined_from_left)
        print_matrix("predefined_from_right", predefined_from_right)
        print_matrix("predefined_avg", predefined_avg)
        print_matrix("ROBOT_LEFT_REAL_BASE_TO_REAL_BASE", np.asarray(ROBOT_LEFT_REAL_BASE_TO_REAL_BASE, dtype=np.float64))
        print_matrix("ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE", np.asarray(ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE, dtype=np.float64))
        print_matrix("ROBOT_PREDEFINED_TRANSFORMATION", np.asarray(ROBOT_PREDEFINED_TRANSFORMATION, dtype=np.float64))


if __name__ == "__main__":
    main()
