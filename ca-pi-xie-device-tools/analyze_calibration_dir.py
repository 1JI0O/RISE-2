#!/usr/bin/env python3
"""
只读分析标准采集标定目录中的 calibration 文件结构。

输入目录通常包含：
- extrinsics.npy
- intrinsics.npy
- tcp.npy
- images/

脚本目标：
1. 打印三个 npy 文件的 Python 结构、key、shape、dtype。
2. 对疑似 4x4 齐次变换矩阵 / 3x3 或 4x4 相机内参做启发式分析。
3. 对 tcp pose 做长度、四元数范数、可能格式的静态检查。
4. 给出“矩阵语义线索”和“可疑点”，帮助后续决定如何转换成目标标定格式。

严格只读：不会写回任何文件。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def fmt_shape(value: Any) -> str:
    arr = np.asarray(value)
    return str(arr.shape)


def fmt_dtype(value: Any) -> str:
    arr = np.asarray(value)
    return str(arr.dtype)


def indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def is_numeric_array(value: Any) -> bool:
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return np.issubdtype(arr.dtype, np.number)


def is_square_matrix(arr: np.ndarray, n: int) -> bool:
    return arr.ndim == 2 and arr.shape == (n, n)


def is_homogeneous_matrix4(arr: np.ndarray, atol: float = 1e-4) -> bool:
    if not is_square_matrix(arr, 4):
        return False
    return np.allclose(arr[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=atol)


def summarize_array_stats(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "empty array"
    return (
        f"shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr.min():.6g}, max={arr.max():.6g}, mean={arr.mean():.6g}"
    )


def analyze_rotation_matrix(rot: np.ndarray, atol: float = 1e-3) -> list[str]:
    notes: list[str] = []
    if rot.shape != (3, 3):
        return ["rotation block is not 3x3"]

    should_be_identity = rot.T @ rot
    det = np.linalg.det(rot)
    if np.allclose(should_be_identity, np.eye(3), atol=atol):
        notes.append("rotation block is approximately orthonormal")
    else:
        notes.append("rotation block is NOT orthonormal -> may not be a rigid transform")

    notes.append(f"det(rotation) = {det:.6f}")
    if not np.isclose(det, 1.0, atol=5e-2):
        notes.append("det is not close to 1 -> suspicious for rigid transform")
    return notes


def analyze_intrinsic_matrix(arr: np.ndarray) -> list[str]:
    notes: list[str] = []
    if is_square_matrix(arr, 3):
        fx, fy = float(arr[0, 0]), float(arr[1, 1])
        cx, cy = float(arr[0, 2]), float(arr[1, 2])
        notes.append(f"looks like 3x3 intrinsic: fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}")
        if not np.isclose(arr[2, 2], 1.0, atol=1e-3):
            notes.append("K[2,2] is not close to 1 -> unusual for pinhole intrinsics")
        if not np.isclose(arr[0, 1], 0.0, atol=1e-3) or not np.isclose(arr[1, 0], 0.0, atol=1e-3):
            notes.append("off-diagonal skew/non-zero terms detected")
    elif is_square_matrix(arr, 4):
        notes.append("4x4 matrix; may be intrinsic padded to homogeneous form or not an intrinsic at all")
        top_left = arr[:3, :3]
        notes.extend([f"top-left 3x3 -> {x}" for x in analyze_intrinsic_matrix(top_left)])
        if np.allclose(arr[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-4):
            notes.append("last row matches homogeneous convention [0,0,0,1]")
    else:
        notes.append("shape is neither 3x3 nor 4x4; unlikely to be a standard intrinsic matrix")
    return notes


def analyze_transform_matrix(arr: np.ndarray) -> list[str]:
    notes: list[str] = []
    if not is_square_matrix(arr, 4):
        return ["not a 4x4 matrix"]

    if is_homogeneous_matrix4(arr):
        notes.append("matches 4x4 homogeneous transform layout")
    else:
        notes.append("4x4 matrix but last row is not [0,0,0,1] -> suspicious")

    notes.extend(analyze_rotation_matrix(arr[:3, :3]))
    t = arr[:3, 3]
    notes.append(f"translation = [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")
    notes.append(f"translation norm = {np.linalg.norm(t):.6f}")
    return notes


def maybe_unwrap_singleton_list(value: Any) -> tuple[Any, list[str]]:
    notes: list[str] = []
    cur = value
    depth = 0
    while isinstance(cur, list) and len(cur) == 1:
        cur = cur[0]
        depth += 1
    if depth > 0:
        notes.append(f"value is wrapped by singleton list depth={depth}")
    return cur, notes


def preview_numeric_array(arr: np.ndarray, max_rows: int = 4, max_cols: int = 6) -> str:
    if arr.ndim == 0:
        return repr(arr.item())
    if arr.ndim == 1:
        n = min(max_cols, arr.shape[0])
        return np.array2string(arr[:n], precision=5, suppress_small=False)
    if arr.ndim == 2:
        r = min(max_rows, arr.shape[0])
        c = min(max_cols, arr.shape[1])
        return np.array2string(arr[:r, :c], precision=5, suppress_small=False)
    return f"ndim={arr.ndim}, preview skipped"


def describe_value(name: str, value: Any) -> str:
    lines: list[str] = []
    unwrapped, unwrap_notes = maybe_unwrap_singleton_list(value)

    lines.append(f"name: {name}")
    lines.append(f"python_type: {type(value).__name__}")
    if unwrapped is not value:
        lines.append(f"unwrapped_type: {type(unwrapped).__name__}")
    lines.extend(unwrap_notes)

    if is_numeric_array(unwrapped):
        arr = np.asarray(unwrapped)
        lines.append(summarize_array_stats(arr))
        lines.append(f"preview: {preview_numeric_array(arr)}")

        if is_square_matrix(arr, 4):
            lines.append("heuristic_role: possible_extrinsic_or_transform")
            for note in analyze_transform_matrix(arr):
                lines.append(f"- {note}")
        elif is_square_matrix(arr, 3) or is_square_matrix(arr, 4):
            lines.append("heuristic_role: possible_intrinsic")
            for note in analyze_intrinsic_matrix(arr):
                lines.append(f"- {note}")
        elif arr.ndim == 1 and arr.shape[0] in (6, 7):
            lines.append("heuristic_role: possible_tcp_pose")
            lines.append(f"- vector length = {arr.shape[0]}")
            if arr.shape[0] == 7:
                quat = arr[3:]
                lines.append(f"- tail 4 values may be quaternion = {quat.tolist()}")
                lines.append(f"- quaternion norm = {np.linalg.norm(quat):.6f}")
            else:
                lines.append("- length=6 may be xyz+rpy / xyz+rotvec / other compact pose format")
        else:
            lines.append("heuristic_role: unknown_numeric")
    elif isinstance(unwrapped, dict):
        lines.append(f"dict_keys: {list(unwrapped.keys())}")
    elif isinstance(unwrapped, list):
        lines.append(f"list_len: {len(unwrapped)}")
        if len(unwrapped) > 0:
            lines.append(f"first_item_type: {type(unwrapped[0]).__name__}")
    else:
        lines.append(f"repr: {repr(unwrapped)}")

    return "\n".join(lines)


def load_npy(path: Path) -> Any:
    return np.load(str(path), allow_pickle=True)


def analyze_extrinsics(path: Path) -> str:
    obj = load_npy(path)
    lines: list[str] = [f"=== extrinsics: {path} ==="]
    lines.append(f"np.load type: {type(obj).__name__}")

    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
        lines.append("scalar object array detected; unpacked with .item()")

    if isinstance(obj, dict):
        lines.append(f"top-level dict keys ({len(obj)}): {list(obj.keys())}")
        serials = []
        for key, value in obj.items():
            lines.append(indent(describe_value(str(key), value)))
            serials.append(str(key))
        if serials:
            lines.append("matrix_semantics_clue: keys look like camera serials -> likely per-camera pose relative to the same marker/world target")
            lines.append("suspicion_check: need to determine whether each matrix means camera->marker or marker->camera; this script only checks shape, not direction")
    else:
        lines.append(describe_value("extrinsics", obj))
        lines.append("suspicion_check: extrinsics is not a dict; this differs from current repository assumptions")

    return "\n".join(lines)


def analyze_intrinsics(path: Path) -> str:
    obj = load_npy(path)
    lines: list[str] = [f"=== intrinsics: {path} ==="]
    lines.append(f"np.load type: {type(obj).__name__}")

    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
        lines.append("scalar object array detected; unpacked with .item()")

    if isinstance(obj, dict):
        lines.append(f"top-level dict keys ({len(obj)}): {list(obj.keys())}")
        for key, value in obj.items():
            lines.append(indent(describe_value(str(key), value)))
        lines.append("matrix_semantics_clue: likely per-camera intrinsic matrices")
        lines.append("suspicion_check: some pipelines save intrinsics as 4x4 padded matrix; downstream may only want top-left 3x3")
    else:
        lines.append(describe_value("intrinsics", obj))
        lines.append("suspicion_check: intrinsics is not a dict; this differs from current repository assumptions")

    return "\n".join(lines)


def analyze_tcp(path: Path) -> str:
    obj = load_npy(path)
    lines: list[str] = [f"=== tcp: {path} ==="]
    lines.append(f"np.load type: {type(obj).__name__}")

    if isinstance(obj, np.ndarray):
        arr = obj
        lines.append(summarize_array_stats(arr))
        lines.append(f"preview: {preview_numeric_array(arr)}")

        if arr.ndim == 1 and arr.shape[0] == 7:
            quat = arr[3:]
            lines.append("heuristic_role: likely one TCP pose in [x,y,z,q?,q?,q?,q?] format")
            lines.append(f"quaternion_tail = {quat.tolist()}")
            lines.append(f"quaternion_norm = {np.linalg.norm(quat):.6f}")
            lines.append("suspicion_check: quaternion ordering may be wxyz or xyzw; repository code uses both styles in different places, so direction/order must be verified")
        elif arr.ndim == 2 and arr.shape[1] in (6, 7):
            lines.append("heuristic_role: likely a time series of TCP poses")
            lines.append(f"frame_count = {arr.shape[0]}")
            if arr.shape[1] == 7:
                quat_norms = np.linalg.norm(arr[:, 3:], axis=1)
                lines.append(
                    f"quaternion_norm_range = [{quat_norms.min():.6f}, {quat_norms.max():.6f}]"
                )
        else:
            lines.append("heuristic_role: unknown_tcp_layout")
    else:
        lines.append(describe_value("tcp", obj))
        lines.append("suspicion_check: tcp is not a numpy array; this is unusual")

    return "\n".join(lines)


def collect_missing_files(base_dir: Path, names: Iterable[str]) -> list[str]:
    return [name for name in names if not (base_dir / name).exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description="只读分析标定目录中的 extrinsics/intrinsics/tcp 文件结构")
    parser.add_argument("calib_dir", type=str, help="标定目录路径，里面应包含 extrinsics.npy / intrinsics.npy / tcp.npy")
    args = parser.parse_args()

    base_dir = Path(args.calib_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"目录不存在: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"不是目录: {base_dir}")

    expected = ["extrinsics.npy", "intrinsics.npy", "tcp.npy"]
    missing = collect_missing_files(base_dir, expected)

    print(f"[info] calib_dir = {base_dir}")
    print(f"[info] absolute_path = {base_dir.resolve()}")
    print(f"[info] files = {sorted(os.listdir(base_dir))}")
    print()

    if missing:
        print(f"[warn] missing files: {missing}")
        print("[warn] analysis will continue for files that do exist")
        print()

    if (base_dir / "extrinsics.npy").exists():
        print(analyze_extrinsics(base_dir / "extrinsics.npy"))
        print()

    if (base_dir / "intrinsics.npy").exists():
        print(analyze_intrinsics(base_dir / "intrinsics.npy"))
        print()

    if (base_dir / "tcp.npy").exists():
        print(analyze_tcp(base_dir / "tcp.npy"))
        print()

    print("=== overall_hints ===")
    print("- 如果 extrinsics.npy 是 {serial: 4x4} 或 {serial: [4x4]}，很像标准采集里‘每个相机相对同一标定板/marker 的位姿结果’。")
    print("- 如果 intrinsics.npy 是 {serial: 3x3/4x4}，通常可以直接或切 top-left 3x3 给下游使用。")
    print("- tcp.npy 若是 7 维，最关键的不确定点通常不是数值本身，而是四元数顺序到底是 wxyz 还是 xyzw。")
    print("- 真正转换成目标标定前，还必须确认 extrinsics 的方向语义：camera->marker 还是 marker->camera。")
    print("- 本脚本不会推断唯一正确答案，只输出结构证据和风险点。")


if __name__ == "__main__":
    main()
