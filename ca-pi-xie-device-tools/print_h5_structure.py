import h5py

H5_PATH = "/data/haoxiang/data/task0012_260321/task0012_toys_basket/scene_0001/lowdim/lowdim.h5"


def print_h5_structure(name, obj):
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Group):
        print(f"{indent}[Group] {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}[Dataset] {name} shape={obj.shape} dtype={obj.dtype}")


with h5py.File(H5_PATH, "r") as f:
    print("[Group] /")
    f.visititems(print_h5_structure)

    pedal_0 = f["pedal_0"][:].reshape(-1)
    print("\n[pedal_0 run-length summary]")

    current_value = pedal_0[0]
    current_count = 1

    for value in pedal_0[1:]:
        if value == current_value:
            current_count += 1
        else:
            print(f"value={int(current_value)} count={current_count}")
            current_value = value
            current_count = 1

    print(f"value={int(current_value)} count={current_count}")

    timestamp = f["timestamp"][:]
    print("\n[timestamp]")
    print(f"first={timestamp[0]}")
    print(f"last={timestamp[-1]}")

    switch_indices = []
    for i in range(1, len(pedal_0)):
        if pedal_0[i] != pedal_0[i - 1]:
            switch_indices.append(i)

    if switch_indices:
        first_switch = switch_indices[0]
        last_switch = switch_indices[-1]

        print("\n[first pedal switch]")
        print(
            f"idx_pair=({first_switch - 1}, {first_switch}) "
            f"pedal_pair=({int(pedal_0[first_switch - 1])}, {int(pedal_0[first_switch])}) "
            f"timestamp_pair=({timestamp[first_switch - 1]}, {timestamp[first_switch]})"
        )

        print("\n[last pedal switch]")
        print(
            f"idx_pair=({last_switch - 1}, {last_switch}) "
            f"pedal_pair=({int(pedal_0[last_switch - 1])}, {int(pedal_0[last_switch])}) "
            f"timestamp_pair=({timestamp[last_switch - 1]}, {timestamp[last_switch]})"
        )
