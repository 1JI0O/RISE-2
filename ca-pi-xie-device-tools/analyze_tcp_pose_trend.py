import h5py
import numpy as np
from pathlib import Path

H5_PATH = Path("/data/haoxiang/data/task0012_260321/task0012_toys_basket/scene_0001/lowdim/lowdim.h5")
DATASET_NAMES = ["tcp_pose_062046", "tcp_pose_062703"]


def analyze_dataset(name: str, data: np.ndarray):
    print(f"\n[{name}]")
    print(f"shape={data.shape}")
    print(f"first_row={data[0].tolist()}")
    print(f"middle_row={data[len(data) // 2].tolist()}")
    print(f"last_row={data[-1].tolist()}")

    diff = data[1:] - data[:-1]
    abs_diff = np.abs(diff)

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


def main():
    with h5py.File(H5_PATH, "r") as f:
        print(f"h5_path={H5_PATH}")
        for dataset_name in DATASET_NAMES:
            data = f[dataset_name][:]
            analyze_dataset(dataset_name, data)


if __name__ == "__main__":
    main()
