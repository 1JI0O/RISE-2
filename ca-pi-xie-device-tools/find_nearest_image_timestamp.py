from pathlib import Path

# IMAGE_DIR = Path("/data/haoxiang/data/task0012_260321/task0012_toys_basket/scene_0001/cam_104122060902/color")
IMAGE_DIR = Path("/data/haoxiang/data/task0012_260321/task0012_toys_basket/scene_0001/cam_036422060622/color")
TARGET_TIMESTAMP = 1774059833605


def parse_timestamp(path: Path) -> int:
    return int(path.stem)


def main():
    image_paths = sorted(
        [p for p in IMAGE_DIR.iterdir() if p.is_file() and p.stem.isdigit()]
    )

    if not image_paths:
        print(f"No timestamp image files found in {IMAGE_DIR}")
        return

    print(f"file_count={len(image_paths)}")

    nearest_path = min(
        image_paths,
        key=lambda p: abs(parse_timestamp(p) - TARGET_TIMESTAMP),
    )
    nearest_timestamp = parse_timestamp(nearest_path)
    nearest_index = image_paths.index(nearest_path)

    print(f"target_timestamp={TARGET_TIMESTAMP}")
    print(f"nearest_timestamp={nearest_timestamp}")
    print(f"abs_diff={abs(nearest_timestamp - TARGET_TIMESTAMP)}")
    print(f"nearest_file={nearest_path}")
    print(f"sorted_index_0_based={nearest_index}")
    print(f"sorted_index_1_based={nearest_index + 1}")


if __name__ == "__main__":
    main()
