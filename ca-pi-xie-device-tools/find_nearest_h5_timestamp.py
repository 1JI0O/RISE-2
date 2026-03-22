import h5py
from pathlib import Path

H5_PATH = Path("/data/haoxiang/data/task0012_260321/task0012_toys_basket/scene_0001/lowdim/lowdim.h5")
TARGET_TIMESTAMP = 1774059832169
IMAGE_DIR = Path("/data/haoxiang/data/task0012_260321/task0012_toys_basket/scene_0001/cam_104122060902/color")


def parse_image_timestamp(path: Path) -> int:
    return int(path.stem)


def find_nearest_h5_timestamp(target_timestamp: int, h5_timestamps):
    nearest_index = min(
        range(len(h5_timestamps)),
        key=lambda i: abs(int(h5_timestamps[i]) - target_timestamp),
    )
    nearest_timestamp = int(h5_timestamps[nearest_index])
    diff = abs(nearest_timestamp - target_timestamp)
    return nearest_index, nearest_timestamp, diff


def main():
    image_paths = sorted(
        [p for p in IMAGE_DIR.iterdir() if p.is_file() and p.stem.isdigit()]
    )
    image_timestamps = [parse_image_timestamp(p) for p in image_paths]

    with h5py.File(H5_PATH, "r") as f:
        timestamps = f["timestamp"][:]

    if len(timestamps) == 0:
        print(f"No timestamp data found in {H5_PATH}")
        return

    if not image_paths:
        print(f"No timestamp image files found in {IMAGE_DIR}")
        return

    timestamp_set = {int(ts) for ts in timestamps}
    matched_timestamps = [ts for ts in image_timestamps if ts in timestamp_set]
    missing_timestamps = [ts for ts in image_timestamps if ts not in timestamp_set]

    nearest_index, nearest_timestamp, abs_diff = find_nearest_h5_timestamp(
        TARGET_TIMESTAMP, timestamps
    )

    print(f"h5_path={H5_PATH}")
    print(f"image_dir={IMAGE_DIR}")
    print(f"h5_timestamp_count={len(timestamps)}")
    print(f"image_file_count={len(image_paths)}")
    print(f"matched_image_timestamp_count={len(matched_timestamps)}")
    print(f"missing_image_timestamp_count={len(missing_timestamps)}")
    print(f"all_image_timestamps_in_h5={len(missing_timestamps) == 0}")

    if missing_timestamps:
        print("missing_image_timestamps=")
        for ts in missing_timestamps:
            missing_nearest_index, missing_nearest_timestamp, missing_diff = find_nearest_h5_timestamp(
                ts, timestamps
            )
            print(
                f"image_timestamp={ts} "
                f"nearest_h5_timestamp={missing_nearest_timestamp} "
                f"diff={missing_diff} "
                f"nearest_h5_index_0_based={missing_nearest_index}"
            )

    print(f"target_timestamp={TARGET_TIMESTAMP}")
    print(f"nearest_timestamp={nearest_timestamp}")
    print(f"abs_diff={abs_diff}")
    print(f"nearest_index_0_based={nearest_index}")
    print(f"nearest_index_1_based={nearest_index + 1}")


if __name__ == "__main__":
    main()
