#!/usr/bin/env python3
from __future__ import annotations

"""
穷举 joint 轴映射（符号 + 轴顺序），按“末端应垂直向下”排序候选。

用途：
- 输入一帧左右臂 h5 joint（7 维，不含 gripper）；
- 基于 URDF FK 穷举可能的 joint 映射；
- 对每个候选计算末端（tcp_from_flange）z 轴与世界向下方向的一致性；
- 输出 top-k 最优候选，便于快速定位 h5_q -> urdf_q 映射。

说明：
- 这里只使用几何约束“末端朝下”；
- 默认不依赖相机外参，因此可先在 robot base 下筛候选；
- 如需更强约束，可后续叠加 tcp_pose 或渲染 mask 评分。
"""

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
AIREXO_ROOT = WORKSPACE_ROOT / "airexo"
for p in [WORKSPACE_ROOT, AIREXO_ROOT]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from airexo.helpers.constants import ROBOT_TCP_TO_FLANGE  # noqa: E402
from airexo.helpers import urdf_robot as robot_helper  # noqa: E402


@dataclass
class JointCfg:
    num_joints: int = 8
    num_robot_joints: int = 7


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def parse_joint7(s: str) -> np.ndarray:
    arr = np.fromstring(s, sep=",", dtype=np.float64)
    if arr.shape != (7,):
        raise ValueError(f"Expected 7 comma-separated values, got shape {arr.shape}")
    return arr


def tcp_from_fk_map(tf_map: Dict[str, object]) -> np.ndarray:
    if "flange" not in tf_map:
        raise KeyError("FK map has no 'flange' link")
    T_base_flange = np.asarray(tf_map["flange"].matrix(), dtype=np.float64)
    T_tcp_flange = np.asarray(ROBOT_TCP_TO_FLANGE, dtype=np.float64)
    # base->tcp = base->flange @ inv(tcp->flange)
    return T_base_flange @ invert_T(T_tcp_flange)


def fk_tcp(joint7: np.ndarray, urdf_file: str) -> np.ndarray:
    joint8 = np.concatenate([joint7, [0.05]], axis=0).astype(np.float32)
    tf_map = robot_helper.forward_kinematic_single(
        joint=joint8,
        joint_cfgs=JointCfg(),
        is_rad=True,
        urdf_file=urdf_file,
        with_visuals_map=False,
    )
    return tcp_from_fk_map(tf_map)


def axis_down_score(T_base_tcp: np.ndarray, down_axis: np.ndarray) -> Tuple[float, float]:
    """
    返回：
    - angle_deg: tcp z 轴与 down_axis 的夹角（越小越好）
    - dot_val:   tcp z 与 down_axis 点积（越接近 1 越好）
    """
    z_tcp = np.asarray(T_base_tcp[:3, 2], dtype=np.float64)
    z_tcp = z_tcp / (np.linalg.norm(z_tcp) + 1e-12)
    d = down_axis / (np.linalg.norm(down_axis) + 1e-12)
    dot_val = float(np.clip(np.dot(z_tcp, d), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(dot_val)))
    return angle_deg, dot_val


def build_candidate_mappings(
    allow_permutation: bool,
    permutation_scope: str,
    allow_sign_flip: bool,
) -> Iterable[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    产出 (perm, signs)
    - perm: 长度 7，表示 urdf_joint[i] = input_joint[perm[i]]
    - signs: 长度 7，元素为 +1/-1
    """
    base_perm = tuple(range(7))

    if not allow_permutation:
        perms = [base_perm]
    else:
        if permutation_scope == "all":
            perms = itertools.permutations(range(7), 7)
        elif permutation_scope == "first4":
            # 前4轴全排列，后3轴保持
            perms = []
            for p4 in itertools.permutations(range(4), 4):
                perms.append(tuple(list(p4) + [4, 5, 6]))
        elif permutation_scope == "first3":
            perms = []
            for p3 in itertools.permutations(range(3), 3):
                perms.append(tuple(list(p3) + [3, 4, 5, 6]))
        else:
            raise ValueError(f"Unknown permutation_scope: {permutation_scope}")

    if allow_sign_flip:
        sign_patterns = list(itertools.product([1, -1], repeat=7))
    else:
        sign_patterns = [(1, 1, 1, 1, 1, 1, 1)]

    for perm in perms:
        for signs in sign_patterns:
            yield tuple(perm), tuple(signs)


def apply_mapping(joint7: np.ndarray, perm: Sequence[int], signs: Sequence[int]) -> np.ndarray:
    out = np.zeros((7,), dtype=np.float64)
    for i in range(7):
        out[i] = joint7[perm[i]] * float(signs[i])
    return out


def mapping_to_str(perm: Sequence[int], signs: Sequence[int]) -> str:
    parts = []
    for i, (p, s) in enumerate(zip(perm, signs), start=1):
        sign = "-" if s < 0 else "+"
        parts.append(f"u{i}={sign}q{p+1}")
    return ",".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="穷举 joint 映射，按末端朝下约束排序")
    parser.add_argument("--left-joint7", required=True, help="左臂 h5 joint7，逗号分隔")
    parser.add_argument("--right-joint7", required=True, help="右臂 h5 joint7，逗号分隔")
    parser.add_argument(
        "--left-urdf",
        default=str((WORKSPACE_ROOT / "airexo/airexo/urdf_models/robot/left_robot_inhand.urdf").resolve()),
    )
    parser.add_argument(
        "--right-urdf",
        default=str((WORKSPACE_ROOT / "airexo/airexo/urdf_models/robot/right_robot_inhand.urdf").resolve()),
    )
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--allow-permutation", action="store_true")
    parser.add_argument(
        "--permutation-scope",
        choices=["first3", "first4", "all"],
        default="first4",
        help="开启 permutation 时的搜索范围",
    )
    parser.add_argument("--allow-sign-flip", action="store_true")
    parser.add_argument(
        "--down-axis",
        default="0,0,-1",
        help="base 坐标系里的‘向下’方向，默认 [0,0,-1]",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    left_q = parse_joint7(args.left_joint7)
    right_q = parse_joint7(args.right_joint7)
    down_axis = np.fromstring(args.down_axis, sep=",", dtype=np.float64)
    if down_axis.shape != (3,):
        raise ValueError("--down-axis must be 3 values, e.g. 0,0,-1")

    results: List[Dict[str, object]] = []

    for perm, signs in build_candidate_mappings(
        allow_permutation=args.allow_permutation,
        permutation_scope=args.permutation_scope,
        allow_sign_flip=args.allow_sign_flip,
    ):
        left_u = apply_mapping(left_q, perm, signs)
        right_u = apply_mapping(right_q, perm, signs)

        try:
            T_left = fk_tcp(left_u, args.left_urdf)
            T_right = fk_tcp(right_u, args.right_urdf)
        except Exception:
            continue

        left_angle, left_dot = axis_down_score(T_left, down_axis)
        right_angle, right_dot = axis_down_score(T_right, down_axis)

        # 双臂总评分：角度和越小越好
        score = left_angle + right_angle
        results.append(
            {
                "score": score,
                "left_angle_deg": left_angle,
                "right_angle_deg": right_angle,
                "left_dot": left_dot,
                "right_dot": right_dot,
                "perm": list(perm),
                "signs": list(signs),
                "mapping": mapping_to_str(perm, signs),
                "left_joint_mapped": left_u.tolist(),
                "right_joint_mapped": right_u.tolist(),
                "left_tcp_xyz": T_left[:3, 3].tolist(),
                "right_tcp_xyz": T_right[:3, 3].tolist(),
            }
        )

    if len(results) == 0:
        print("No valid mapping candidates found.")
        return

    results.sort(key=lambda x: x["score"])

    topk = min(args.topk, len(results))
    print(f"total_candidates={len(results)}, topk={topk}")
    print("=")
    for i in range(topk):
        r = results[i]
        print(
            f"#{i+1:02d} score={r['score']:.3f}  "
            f"L={r['left_angle_deg']:.3f}deg R={r['right_angle_deg']:.3f}deg  "
            f"map={r['mapping']}"
        )

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results[:topk], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
