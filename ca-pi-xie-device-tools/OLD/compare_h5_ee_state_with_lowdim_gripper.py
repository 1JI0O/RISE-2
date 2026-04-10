import h5py
import numpy as np
from pathlib import Path

H5_PATH = Path("/data/haoxiang/data/task0012_260321/task0012_toys_basket/scene_0001/lowdim/lowdim.h5")
LOWDIM_DIR = Path("/data/haoxiang/data/airexo2/task_0012/train/scene_0001/lowdim")
SAMPLE_COUNT = 200


def load_lowdim_gripper_values(lowdim_dir: Path, key: str, value_index: int, sample_count: int):
    npy_files = sorted([p for p in lowdim_dir.iterdir() if p.suffix == ".npy"])
    if not npy_files:
        return np.array([], dtype=np.float32)

    if len(npy_files) > sample_count:
        indices = np.linspace(0, len(npy_files) - 1, num=sample_count, dtype=int)
        npy_files = [npy_files[i] for i in indices]

    values = []
    for npy_file in npy_files:
        data = np.load(npy_file, allow_pickle=True).item()
        values.append(float(data[key][value_index]))
    return np.array(values, dtype=np.float32)


def summarize(name: str, values: np.ndarray):
    print(f"\n[{name}]")
    print(f"count={len(values)}")
    if len(values) == 0:
        return
    print(f"min={float(values.min()):.6f}")
    print(f"max={float(values.max()):.6f}")
    print(f"mean={float(values.mean()):.6f}")
    print(f"std={float(values.std()):.6f}")
    print(f"first_20={values[:20].tolist()}")
    rounded_unique = sorted(set(np.round(values, 6).tolist()))
    print(f"rounded_unique_first_30={rounded_unique[:30]}")


def compare_style(name_a: str, values_a: np.ndarray, name_b: str, values_b: np.ndarray):
    print(f"\n[compare {name_a} vs {name_b}]")
    if len(values_a) == 0 or len(values_b) == 0:
        print("empty input")
        return

    print(f"range_a=({float(values_a.min()):.6f}, {float(values_a.max()):.6f})")
    print(f"range_b=({float(values_b.min()):.6f}, {float(values_b.max()):.6f})")
    print(f"mean_a={float(values_a.mean()):.6f}")
    print(f"mean_b={float(values_b.mean()):.6f}")
    print(f"std_a={float(values_a.std()):.6f}")
    print(f"std_b={float(values_b.std()):.6f}")

    rounded_a = set(np.round(values_a, 4).tolist())
    rounded_b = set(np.round(values_b, 4).tolist())
    overlap = sorted(rounded_a & rounded_b)
    print(f"rounded_overlap_count_4dp={len(overlap)}")
    print(f"rounded_overlap_first_30={overlap[:30]}")


def main():
    with h5py.File(H5_PATH, "r") as f:
        ee_state_062046 = f["ee_state_062046"][:].reshape(-1)
        ee_state_062703 = f["ee_state_062703"][:].reshape(-1)

    lowdim_left_width = load_lowdim_gripper_values(LOWDIM_DIR, "gripper_left", 0, SAMPLE_COUNT)
    lowdim_left_action = load_lowdim_gripper_values(LOWDIM_DIR, "gripper_left", 1, SAMPLE_COUNT)
    lowdim_right_width = load_lowdim_gripper_values(LOWDIM_DIR, "gripper_right", 0, SAMPLE_COUNT)
    lowdim_right_action = load_lowdim_gripper_values(LOWDIM_DIR, "gripper_right", 1, SAMPLE_COUNT)

    print(f"h5_path={H5_PATH}")
    print(f"lowdim_dir={LOWDIM_DIR}")
    print(f"sample_count={SAMPLE_COUNT}")

    summarize("ee_state_062046", ee_state_062046)
    summarize("ee_state_062703", ee_state_062703)
    summarize("lowdim_gripper_left_width", lowdim_left_width)
    summarize("lowdim_gripper_left_action", lowdim_left_action)
    summarize("lowdim_gripper_right_width", lowdim_right_width)
    summarize("lowdim_gripper_right_action", lowdim_right_action)

    compare_style("ee_state_062046", ee_state_062046, "lowdim_gripper_left_width", lowdim_left_width)
    compare_style("ee_state_062046", ee_state_062046, "lowdim_gripper_left_action", lowdim_left_action)
    compare_style("ee_state_062703", ee_state_062703, "lowdim_gripper_right_width", lowdim_right_width)
    compare_style("ee_state_062703", ee_state_062703, "lowdim_gripper_right_action", lowdim_right_action)


if __name__ == "__main__":
    main()
