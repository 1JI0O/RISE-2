import numpy as np
from pathlib import Path

LOWDIM_DIR = Path("/data/haoxiang/data/airexo2/task_0012/train/scene_0001/lowdim")
MOVEMENT_THRESHOLD = 0.01


def load_robot_series(lowdim_dir: Path, key: str) -> tuple[np.ndarray, list[str]]:
    npy_files = sorted([p for p in lowdim_dir.iterdir() if p.suffix == ".npy"])
    timestamps = [p.stem for p in npy_files]
    values = []
    for npy_file in npy_files:
        data = np.load(npy_file, allow_pickle=True).item()
        values.append(np.asarray(data[key], dtype=np.float32)[:7])
    return np.stack(values, axis=0), timestamps


def summarize_series(name: str, data: np.ndarray):
    print(f"\n[{name}]")
    print(f"shape={data.shape}")
    print(f"first_row={data[0].tolist()}")
    print(f"middle_row={data[len(data) // 2].tolist()}")
    print(f"last_row={data[-1].tolist()}")

    diff = data[1:] - data[:-1]
    abs_diff = np.abs(diff)
    step_norm = np.linalg.norm(diff, axis=1)

    print(f"step_norm_mean={float(step_norm.mean()):.6f}")
    print(f"step_norm_max={float(step_norm.max()):.6f}")
    print(f"first_20_step_norm={step_norm[:20].tolist()}")

    print("per_dim_summary=")
    for dim in range(data.shape[1]):
        dim_values = data[:, dim]
        dim_diff = diff[:, dim]
        dim_abs_diff = abs_diff[:, dim]
        print(
            f"dim={dim} "
            f"min={float(dim_values.min()):.6f} "
            f"max={float(dim_values.max()):.6f} "
            f"mean={float(dim_values.mean()):.6f} "
            f"start={float(dim_values[0]):.6f} "
            f"end={float(dim_values[-1]):.6f} "
            f"delta={float(dim_values[-1] - dim_values[0]):.6f} "
            f"mean_step={float(dim_diff.mean()):.6f} "
            f"mean_abs_step={float(dim_abs_diff.mean()):.6f} "
            f"max_abs_step={float(dim_abs_diff.max()):.6f}"
        )

    return step_norm


def first_significant_motion(step_norm: np.ndarray, timestamps: list[str], threshold: float, name: str):
    indices = np.where(step_norm > threshold)[0]
    print(f"\n[{name} first_significant_motion]")
    print(f"threshold={threshold}")
    if len(indices) == 0:
        print("no motion above threshold")
        return None

    idx = int(indices[0])
    print(f"step_index_0_based={idx}")
    print(f"timestamp_pair=({timestamps[idx]}, {timestamps[idx + 1]})")
    print(f"step_norm={float(step_norm[idx]):.6f}")
    return idx


def main():
    left_data, timestamps = load_robot_series(LOWDIM_DIR, "robot_left")
    right_data, _ = load_robot_series(LOWDIM_DIR, "robot_right")

    print(f"lowdim_dir={LOWDIM_DIR}")
    print(f"frame_count={len(timestamps)}")
    print(f"first_timestamp={timestamps[0]}")
    print(f"last_timestamp={timestamps[-1]}")

    left_step_norm = summarize_series("robot_left_tcp_pose", left_data)
    right_step_norm = summarize_series("robot_right_tcp_pose", right_data)

    left_first = first_significant_motion(left_step_norm, timestamps, MOVEMENT_THRESHOLD, "robot_left")
    right_first = first_significant_motion(right_step_norm, timestamps, MOVEMENT_THRESHOLD, "robot_right")

    print("\n[which moves first]")
    if left_first is None and right_first is None:
        print("neither exceeds threshold")
    elif right_first is None or (left_first is not None and left_first < right_first):
        print("robot_left moves first")
    elif left_first is None or right_first < left_first:
        print("robot_right moves first")
    else:
        print("robot_left and robot_right move first at the same step")


if __name__ == "__main__":
    main()
