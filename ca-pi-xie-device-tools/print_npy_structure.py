from pathlib import Path

import numpy as np

NPY_PATH = Path("/data/haoxiang/data/airexo2/task_0012/train/scene_0001/lowdim/1736245283129.npy")
LOWDIM_DIR = NPY_PATH.parent
SAMPLE_COUNT = 8


def print_structure(obj, prefix="root"):
    if isinstance(obj, np.ndarray):
        print(f"{prefix}: ndarray shape={obj.shape} dtype={obj.dtype}")
    elif isinstance(obj, dict):
        print(f"{prefix}: dict keys={list(obj.keys())}")
        for key, value in obj.items():
            print_structure(value, f"{prefix}.{key}")
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}: {type(obj).__name__} len={len(obj)}")
        for i, value in enumerate(obj):
            print_structure(value, f"{prefix}[{i}]")
    else:
        print(f"{prefix}: {type(obj).__name__} value={obj}")


def load_lowdim_dict(npy_path: Path):
    data = np.load(npy_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()
    return data


def collect_sample_files(lowdim_dir: Path, sample_count: int):
    npy_files = sorted([p for p in lowdim_dir.iterdir() if p.suffix == ".npy"])
    if len(npy_files) <= sample_count:
        return npy_files

    indices = np.linspace(0, len(npy_files) - 1, num=sample_count, dtype=int)
    return [npy_files[i] for i in indices]


def main():
    data = np.load(NPY_PATH, allow_pickle=True)

    print(f"npy_path={NPY_PATH}")
    print(f"loaded_type={type(data).__name__}")

    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()
        print(f"unpacked_scalar_type={type(data).__name__}")

    print_structure(data)

    print("\n[selected field contents]")
    for key in ["robot_left", "gripper_left", "robot_right", "gripper_right"]:
        value = data[key]
        print(f"{key}: shape={value.shape} dtype={value.dtype}")
        print(value.tolist())

    print("\n[training code usage]")
    print("dual action uses robot_left[0:7] + gripper_left[1:2] + robot_right[0:7] + gripper_right[1:2]")
    print("if gripper_info_type='state', then it uses gripper[*][0:1]")
    print("robot means tcp_pose first 7 dims only in training action loading")
    print("gripper default command means second dim, state means first dim")

    print("\n[gripper samples across files]")
    sample_files = collect_sample_files(LOWDIM_DIR, SAMPLE_COUNT)
    for sample_file in sample_files:
        sample_data = load_lowdim_dict(sample_file)
        print(f"timestamp={sample_file.stem}")
        print(f"  gripper_left={sample_data['gripper_left'].tolist()}")
        print(f"  gripper_right={sample_data['gripper_right'].tolist()}")

    print("\n[gripper value distribution from samples]")
    left_first = []
    left_second = []
    right_first = []
    right_second = []
    for sample_file in sample_files:
        sample_data = load_lowdim_dict(sample_file)
        left_first.append(float(sample_data["gripper_left"][0]))
        left_second.append(float(sample_data["gripper_left"][1]))
        right_first.append(float(sample_data["gripper_right"][0]))
        right_second.append(float(sample_data["gripper_right"][1]))

    print(f"sample_count={len(sample_files)}")
    print(f"left_first_unique={sorted(set(left_first))}")
    print(f"left_second_unique={sorted(set(left_second))}")
    print(f"right_first_unique={sorted(set(right_first))}")
    print(f"right_second_unique={sorted(set(right_second))}")


if __name__ == "__main__":
    main()
