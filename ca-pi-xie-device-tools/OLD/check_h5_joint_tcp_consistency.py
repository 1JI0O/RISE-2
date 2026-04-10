#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return R.from_quat(q).as_matrix()


def quat_wxyz_to_xyzw(qwxyz: np.ndarray) -> np.ndarray:
    qwxyz = np.asarray(qwxyz, dtype=np.float64).reshape(4)
    return np.array([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]], dtype=np.float64)


def pose7_xyzw_to_mat(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_mat(pose7[3:7])
    T[:3, 3] = pose7[:3]
    return T


def pose7_wxyz_to_mat(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_mat(quat_wxyz_to_xyzw(pose7[3:7]))
    T[:3, 3] = pose7[:3]
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def trans_err(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3]))


def rot_deg_err(A: np.ndarray, B: np.ndarray) -> float:
    rel = A[:3, :3] @ B[:3, :3].T
    val = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))


def find_h5_index(ts: np.ndarray, camera_ts: int, exact: bool) -> int:
    if exact:
        hit = np.where(ts == camera_ts)[0]
        if len(hit) == 0:
            raise ValueError(f"Exact timestamp {camera_ts} not found in h5")
        return int(hit[0])
    return int(np.argmin(np.abs(ts - camera_ts)))


def compute_fk_map(
    joint8: np.ndarray,
    joint_cfgs: Any,
    urdf_file: str,
) -> Dict[str, np.ndarray]:
    from airexo.helpers import urdf_robot as robot_helper

    tf_map = robot_helper.forward_kinematic_single(
        joint=np.asarray(joint8, dtype=np.float32),
        joint_cfgs=joint_cfgs,
        is_rad=True,
        urdf_file=urdf_file,
        with_visuals_map=False,
    )
    out: Dict[str, np.ndarray] = {}
    for k, v in tf_map.items():
        out[k] = np.asarray(v.matrix(), dtype=np.float64)
    return out


def build_side_alignment_candidates(
    side: str,
    P: np.ndarray,
    LP: np.ndarray,
    RP: np.ndarray,
    LRB: np.ndarray,
    RRB: np.ndarray,
) -> Dict[str, np.ndarray]:
    S = LP if side == "left" else RP
    B = LRB if side == "left" else RRB

    cands = {
        "I": np.eye(4, dtype=np.float64),
        "P": P,
        "invP": invert_T(P),
        "S": S,
        "invS": invert_T(S),
        "P@S": P @ S,
        "S@P": S @ P,
        "inv(P@S)": invert_T(P @ S),
        "B": B,
        "invB": invert_T(B),
        "P@B": P @ B,
        "B@P": B @ P,
        "inv(P@B)": invert_T(P @ B),
        "S@B": S @ B,
        "B@S": B @ S,
        "inv(S@B)": invert_T(S @ B),
    }
    return cands


def build_targets(h5_pose7: np.ndarray, quat_mode: str) -> List[Tuple[str, str, np.ndarray]]:
    # (quat_conv, direction, T)
    targets: List[Tuple[str, str, np.ndarray]] = []

    if quat_mode in ("xyzw", "both"):
        T = pose7_xyzw_to_mat(h5_pose7)
        targets.append(("xyzw", "base->tcp", T))
        targets.append(("xyzw", "tcp->base", invert_T(T)))

    if quat_mode in ("wxyz", "both"):
        T = pose7_wxyz_to_mat(h5_pose7)
        targets.append(("wxyz", "base->tcp", T))
        targets.append(("wxyz", "tcp->base", invert_T(T)))

    return targets


def build_fk_sources(fk_map: Dict[str, np.ndarray], tcp_to_flange: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    rows: List[Tuple[str, np.ndarray]] = []
    if "flange" in fk_map:
        T_base_flange = fk_map["flange"]
        rows.append(("flange @ inv(T_tcp_flange)", T_base_flange @ invert_T(tcp_to_flange)))
        rows.append(("flange @ T_tcp_flange", T_base_flange @ tcp_to_flange))
        rows.append(("flange", T_base_flange))
    if "link7" in fk_map:
        rows.append(("link7", fk_map["link7"]))
    return rows


def rank_candidates(
    fk_map: Dict[str, np.ndarray],
    h5_pose7: np.ndarray,
    align_cands: Dict[str, np.ndarray],
    tcp_to_flange: np.ndarray,
    quat_mode: str,
    topk: int,
) -> List[Tuple[float, float, str, str, str, str]]:
    # (trans_err, rot_err_deg, quat_conv, target_dir, align_name, fk_chain)
    targets = build_targets(h5_pose7, quat_mode)
    fk_sources = build_fk_sources(fk_map, tcp_to_flange)

    rows: List[Tuple[float, float, str, str, str, str]] = []
    for conv, direction, T_h5 in targets:
        for align_name, A in align_cands.items():
            for chain_name, T_fk in fk_sources:
                T_cmp = A @ T_fk
                te = trans_err(T_cmp, T_h5)
                re = rot_deg_err(T_cmp, T_h5)
                rows.append((te, re, conv, direction, align_name, chain_name))

    rows.sort(key=lambda x: (x[0], x[1]))
    return rows[:topk]


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 h5 joint 与 tcp_pose 的自洽性，并自动枚举可能的坐标/链条解释")
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--camera-timestamp", type=int, default=None)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--sample-indices", default="0,100,1000")
    parser.add_argument("--left-suffix", default="062703")
    parser.add_argument("--right-suffix", default="062046")
    parser.add_argument("--workspace-root", default="/home/haoxiang/rise2_mask_aware")
    parser.add_argument("--h5-quat", choices=["wxyz", "xyzw", "both"], default="wxyz")
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    airexo_root = workspace_root / "airexo"

    import sys

    for p in [workspace_root, airexo_root]:
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)

    from airexo.helpers.constants import (
        ROBOT_TCP_TO_FLANGE,
        ROBOT_PREDEFINED_TRANSFORMATION,
        LEFT_ROBOT_PREDEFINED_TRANSFORMATION,
        RIGHT_ROBOT_PREDEFINED_TRANSFORMATION,
        ROBOT_LEFT_REAL_BASE_TO_REAL_BASE,
        ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE,
    )

    left_urdf = str((workspace_root / "airexo/airexo/urdf_models/robot/left_robot_inhand.urdf").resolve())
    right_urdf = str((workspace_root / "airexo/airexo/urdf_models/robot/right_robot_inhand.urdf").resolve())

    class JointCfg:
        def __init__(self):
            self.num_joints = 8
            self.num_robot_joints = 7

    left_cfg = JointCfg()
    right_cfg = JointCfg()

    P = np.asarray(ROBOT_PREDEFINED_TRANSFORMATION, dtype=np.float64)
    LP = np.asarray(LEFT_ROBOT_PREDEFINED_TRANSFORMATION, dtype=np.float64)
    RP = np.asarray(RIGHT_ROBOT_PREDEFINED_TRANSFORMATION, dtype=np.float64)
    LRB = np.asarray(ROBOT_LEFT_REAL_BASE_TO_REAL_BASE, dtype=np.float64)
    RRB = np.asarray(ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE, dtype=np.float64)
    T_tcp_flange = np.asarray(ROBOT_TCP_TO_FLANGE, dtype=np.float64)

    with h5py.File(args.h5_path, "r") as f:
        ts = np.asarray(f["timestamp"][:], dtype=np.int64)

        if args.camera_timestamp is not None:
            indices = [find_h5_index(ts, args.camera_timestamp, args.exact)]
        else:
            indices = [int(x.strip()) for x in args.sample_indices.split(",") if x.strip()]
            indices = [i for i in indices if 0 <= i < len(ts)]

        left_joint_ds = f[f"joint_position_rad_{args.left_suffix}"]
        right_joint_ds = f[f"joint_position_rad_{args.right_suffix}"]
        left_tcp_ds = f[f"tcp_pose_{args.left_suffix}"]
        right_tcp_ds = f[f"tcp_pose_{args.right_suffix}"]
        left_ee_ds = f[f"ee_state_{args.left_suffix}"]
        right_ee_ds = f[f"ee_state_{args.right_suffix}"]

        print(f"h5_path={args.h5_path}")
        print(f"left_suffix={args.left_suffix}, right_suffix={args.right_suffix}")
        print(f"num_frames={len(ts)}")
        print(f"checked_indices={indices}")
        print(f"h5_quat_mode={args.h5_quat}")
        print("-")

        for idx in indices:
            t = int(ts[idx])
            left_joint7 = np.asarray(left_joint_ds[idx], dtype=np.float64)
            right_joint7 = np.asarray(right_joint_ds[idx], dtype=np.float64)
            left_gripper = float(np.asarray(left_ee_ds[idx]).reshape(-1)[0])
            right_gripper = float(np.asarray(right_ee_ds[idx]).reshape(-1)[0])
            left_joint8 = np.concatenate([left_joint7, [left_gripper]], axis=0)
            right_joint8 = np.concatenate([right_joint7, [right_gripper]], axis=0)

            left_tcp_h5 = np.asarray(left_tcp_ds[idx], dtype=np.float64)
            right_tcp_h5 = np.asarray(right_tcp_ds[idx], dtype=np.float64)

            left_fk_map = compute_fk_map(left_joint8, left_cfg, left_urdf)
            right_fk_map = compute_fk_map(right_joint8, right_cfg, right_urdf)

            left_align = build_side_alignment_candidates("left", P, LP, RP, LRB, RRB)
            right_align = build_side_alignment_candidates("right", P, LP, RP, LRB, RRB)

            left_top = rank_candidates(left_fk_map, left_tcp_h5, left_align, T_tcp_flange, args.h5_quat, topk=args.topk)
            right_top = rank_candidates(right_fk_map, right_tcp_h5, right_align, T_tcp_flange, args.h5_quat, topk=args.topk)

            print(f"idx={idx}, timestamp={t}")
            print("  left_joint7 =", np.array2string(left_joint7, precision=6))
            print("  right_joint7=", np.array2string(right_joint7, precision=6))

            print("  [left top candidates]")
            for te, re, conv, direction, align_name, chain_name in left_top:
                print(f"    trans={te:.6f}m rot={re:.3f}deg | h5={conv},{direction} | align={align_name} | fk={chain_name}")

            print("  [right top candidates]")
            for te, re, conv, direction, align_name, chain_name in right_top:
                print(f"    trans={te:.6f}m rot={re:.3f}deg | h5={conv},{direction} | align={align_name} | fk={chain_name}")
            print("-")


if __name__ == "__main__":
    main()
