import h5py
import numpy as np
from pathlib import Path

H5_PATH = Path("/data/haoxiang/data/task0012_260321/task0012_toys_basket/scene_0001/lowdim/lowdim.h5")
DATASET_NAMES = [
    "ee_command_062046",
    "ee_command_062703",
    "ee_state_062046",
    "ee_state_062703",
]


def summarize_binary_like_signal(name: str, values: np.ndarray):
    values = values.reshape(-1)
    unique_values, counts = np.unique(values, return_counts=True)

    print(f"\n[{name}]")
    print(f"shape={values.shape}")
    print(f"dtype={values.dtype}")
    print(f"first_20={values[:20].tolist()}")
    print(f"last_20={values[-20:].tolist()}")
    print(f"unique_values={unique_values.tolist()}")
    print(f"unique_counts={counts.tolist()}")

    if len(unique_values) <= 10:
        print("value_count_pairs=")
        for value, count in zip(unique_values, counts):
            print(f"value={float(value):.6f} count={int(count)}")

    change_indices = np.where(values[1:] != values[:-1])[0] + 1
    print(f"switch_count={len(change_indices)}")

    if len(change_indices) > 0:
        print(f"first_switch_index_0_based={int(change_indices[0])}")
        print(f"last_switch_index_0_based={int(change_indices[-1])}")

    current_value = values[0]
    current_count = 1
    run_lengths = []
    for value in values[1:]:
        if value == current_value:
            current_count += 1
        else:
            run_lengths.append((float(current_value), current_count))
            current_value = value
            current_count = 1
    run_lengths.append((float(current_value), current_count))

    print(f"run_segment_count={len(run_lengths)}")
    print("first_20_run_segments=")
    for value, count in run_lengths[:20]:
        print(f"value={value:.6f} count={count}")


def compare_command_and_state(command_name: str, state_name: str, command: np.ndarray, state: np.ndarray):
    command = command.reshape(-1)
    state = state.reshape(-1)
    equal_mask = command == state
    diff_indices = np.where(~equal_mask)[0]

    print(f"\n[compare {command_name} vs {state_name}]")
    print(f"same_count={int(equal_mask.sum())}")
    print(f"different_count={int((~equal_mask).sum())}")
    print(f"same_ratio={float(equal_mask.mean()):.6f}")

    if len(diff_indices) > 0:
        print(f"first_diff_index_0_based={int(diff_indices[0])}")
        print(f"last_diff_index_0_based={int(diff_indices[-1])}")
        print("first_20_diffs=")
        for idx in diff_indices[:20]:
            print(
                f"idx={int(idx)} command={float(command[idx]):.6f} state={float(state[idx]):.6f}"
            )


def main():
    with h5py.File(H5_PATH, "r") as f:
        print(f"h5_path={H5_PATH}")
        data = {name: f[name][:] for name in DATASET_NAMES}

    for name in DATASET_NAMES:
        summarize_binary_like_signal(name, data[name])

    compare_command_and_state(
        "ee_command_062046",
        "ee_state_062046",
        data["ee_command_062046"],
        data["ee_state_062046"],
    )
    compare_command_and_state(
        "ee_command_062703",
        "ee_state_062703",
        data["ee_command_062703"],
        data["ee_state_062703"],
    )


if __name__ == "__main__":
    main()
