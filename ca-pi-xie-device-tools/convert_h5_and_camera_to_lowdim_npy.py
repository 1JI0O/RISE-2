import os
from pathlib import Path

import h5py
import numpy as np

# =========================
# 用户需要修改的全局变量
# =========================
# 这里填写 task 根目录。程序会自动遍历其下所有 scene_* 目录。
TASK_DIR = Path("/data/haoxiang/data/task0012_260321/task0012_toys_basket")

# 这里填写要使用的相机目录名（scene 内的子目录名）。
# 程序会在每个 scene 下寻找：scene_xxxx / CAMERA_SUBDIR / color 和 depth
CAMERA_SUBDIR = "cam_104122060902"

# 输出 task 根目录。
# 程序会在这里自动创建 RISE2 训练期望布局：
# OUTPUT_TASK_DIR/
#   train/
#     scene_xxxx/
#       cam_<global_serial>/color
#       cam_<global_serial>/depth
#       lowdim/
#   calib/
OUTPUT_TASK_DIR = Path("/data/haoxiang/data/task0012_260321/task0012_toys_basket_converted")

# 机械臂与 h5 字段的对应关系：
# - 你已经确认 tcp_pose_062703 先开始动，并且它相对于相机视角是 left
# - 因此这里约定：
#   left  <- *_062703
#   right <- *_062046
LEFT_ARM_SUFFIX = "062703"
RIGHT_ARM_SUFFIX = "062046"

# 是否打印每个样本的详细处理日志
VERBOSE = True


def ensure_dir(path: Path):
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink(src: Path, dst: Path):
    """创建软链接；若目标已存在，则跳过。"""
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src, dst)


def find_nearest_h5_index(h5_timestamps: np.ndarray, camera_timestamp: int) -> int:
    """给定相机时间戳，寻找 h5 中最接近它的 timestamp 下标。"""
    return int(np.argmin(np.abs(h5_timestamps - camera_timestamp)))


def build_robot_vector(tcp_pose: np.ndarray) -> np.ndarray:
    """
    构造 robot_* 字段。

    目标长度为 33：
    - 前 7 位放 tcp_pose
    - 剩余位置填 0
    """
    robot = np.zeros(33, dtype=np.float32)
    robot[:7] = tcp_pose.astype(np.float32)
    return robot


def build_gripper_vector(ee_state_value: float) -> np.ndarray:
    """
    构造 gripper_* 字段。

    用户要求：
    - h5 里 ee_state 只有 1 位
    - npy 中对应字段长度为 2
    - 两位都填相同的 ee_state 值
    """
    value = np.float32(ee_state_value)
    return np.array([value, value], dtype=np.float32)


def build_empty_airexo_vector() -> np.ndarray:
    """
    airexo_* 当前不参与训练读取。
    按用户要求可以不管，因此这里统一填 8 维 0。
    """
    return np.zeros(8, dtype=np.float32)


def build_lowdim_dict(
    left_tcp_pose: np.ndarray,
    right_tcp_pose: np.ndarray,
    left_ee_state: float,
    right_ee_state: float,
):
    """按目标 lowdim npy 结构拼装一个 dict。"""
    return {
        "robot_left": build_robot_vector(left_tcp_pose),
        "gripper_left": build_gripper_vector(left_ee_state),
        "airexo_left": build_empty_airexo_vector(),
        "robot_right": build_robot_vector(right_tcp_pose),
        "gripper_right": build_gripper_vector(right_ee_state),
        "airexo_right": build_empty_airexo_vector(),
    }


def process_scene(scene_dir: Path):
    """
    处理单个 scene。

    输入目录约定：
    - scene_dir / CAMERA_SUBDIR / color
    - scene_dir / CAMERA_SUBDIR / depth
    - scene_dir / lowdim / lowdim.h5

    输出目录约定（RISE2 训练布局）：
    - OUTPUT_TASK_DIR / train / scene_xxxx / CAMERA_SUBDIR / color
    - OUTPUT_TASK_DIR / train / scene_xxxx / CAMERA_SUBDIR / depth
    - OUTPUT_TASK_DIR / train / scene_xxxx / lowdim
    """
    camera_dir = scene_dir / CAMERA_SUBDIR
    color_dir = camera_dir / "color"
    depth_dir = camera_dir / "depth"
    h5_path = scene_dir / "lowdim" / "lowdim.h5"

    output_scene_dir = OUTPUT_TASK_DIR / "train" / scene_dir.name
    output_camera_dir = output_scene_dir / CAMERA_SUBDIR
    output_color_dir = output_camera_dir / "color"
    output_depth_dir = output_camera_dir / "depth"
    output_lowdim_dir = output_scene_dir / "lowdim"

    if not color_dir.is_dir():
        print(f"[skip scene] missing color dir: {color_dir}")
        return None
    if not depth_dir.is_dir():
        print(f"[skip scene] missing depth dir: {depth_dir}")
        return None
    if not h5_path.is_file():
        print(f"[skip scene] missing h5 file: {h5_path}")
        return None

    ensure_dir(output_color_dir)
    ensure_dir(output_depth_dir)
    ensure_dir(output_lowdim_dir)

    color_files = sorted([p for p in color_dir.iterdir() if p.is_file()])
    depth_files = sorted([p for p in depth_dir.iterdir() if p.is_file()])

    if len(color_files) != len(depth_files):
        raise RuntimeError(
            f"scene={scene_dir.name} color/depth file count mismatch: "
            f"color={len(color_files)} depth={len(depth_files)}"
        )

    color_names = [p.name for p in color_files]
    depth_names = [p.name for p in depth_files]
    if color_names != depth_names:
        raise RuntimeError(f"scene={scene_dir.name} color/depth filenames are not strictly identical")

    with h5py.File(h5_path, "r") as f:
        h5_timestamps = f["timestamp"][:].astype(np.int64)
        pedal = f["pedal_0"][:].reshape(-1)

        left_tcp_dataset = f[f"tcp_pose_{LEFT_ARM_SUFFIX}"]
        right_tcp_dataset = f[f"tcp_pose_{RIGHT_ARM_SUFFIX}"]
        left_ee_state_dataset = f[f"ee_state_{LEFT_ARM_SUFFIX}"]
        right_ee_state_dataset = f[f"ee_state_{RIGHT_ARM_SUFFIX}"]

        total_images = len(color_files)
        kept_count = 0
        skipped_by_pedal_count = 0

        print(f"\n[scene] {scene_dir.name}")
        print(f"camera_dir={camera_dir}")
        print(f"h5_path={h5_path}")
        print(f"output_scene_dir={output_scene_dir}")
        print(f"output_camera_dir={output_camera_dir}")
        print(f"total_images={total_images}")
        print(f"left_arm_suffix={LEFT_ARM_SUFFIX}")
        print(f"right_arm_suffix={RIGHT_ARM_SUFFIX}")

        for color_file in color_files:
            camera_timestamp = int(color_file.stem)
            nearest_h5_index = find_nearest_h5_index(h5_timestamps, camera_timestamp)
            nearest_h5_timestamp = int(h5_timestamps[nearest_h5_index])
            timestamp_diff = abs(nearest_h5_timestamp - camera_timestamp)
            pedal_value = float(pedal[nearest_h5_index])

            if pedal_value == 1.0:
                skipped_by_pedal_count += 1
                if VERBOSE:
                    print(
                        f"[skip] scene={scene_dir.name} camera_ts={camera_timestamp} "
                        f"nearest_h5_ts={nearest_h5_timestamp} diff={timestamp_diff} pedal_0={pedal_value}"
                    )
                continue

            left_tcp_pose = np.asarray(left_tcp_dataset[nearest_h5_index], dtype=np.float32)
            right_tcp_pose = np.asarray(right_tcp_dataset[nearest_h5_index], dtype=np.float32)
            left_ee_state = float(np.asarray(left_ee_state_dataset[nearest_h5_index]).reshape(-1)[0])
            right_ee_state = float(np.asarray(right_ee_state_dataset[nearest_h5_index]).reshape(-1)[0])

            lowdim_dict = build_lowdim_dict(
                left_tcp_pose=left_tcp_pose,
                right_tcp_pose=right_tcp_pose,
                left_ee_state=left_ee_state,
                right_ee_state=right_ee_state,
            )

            depth_file = depth_dir / color_file.name
            output_color_file = output_color_dir / color_file.name
            output_depth_file = output_depth_dir / depth_file.name
            output_lowdim_file = output_lowdim_dir / f"{camera_timestamp}.npy"

            safe_symlink(color_file, output_color_file)
            safe_symlink(depth_file, output_depth_file)
            np.save(output_lowdim_file, lowdim_dict, allow_pickle=True)

            kept_count += 1

            if VERBOSE:
                print(
                    f"[keep] scene={scene_dir.name} camera_ts={camera_timestamp} "
                    f"nearest_h5_ts={nearest_h5_timestamp} diff={timestamp_diff} pedal_0={pedal_value} "
                    f"saved={output_lowdim_file}"
                )

        print("[scene summary]")
        print(f"scene={scene_dir.name}")
        print(f"total_images={total_images}")
        print(f"kept_count={kept_count}")
        print(f"skipped_by_pedal_count={skipped_by_pedal_count}")
        print(f"output_camera_dir={output_camera_dir}")
        print(f"output_lowdim_dir={output_lowdim_dir}")

        return {
            "scene": scene_dir.name,
            "total_images": total_images,
            "kept_count": kept_count,
            "skipped_by_pedal_count": skipped_by_pedal_count,
        }


def main():
    scene_dirs = sorted([p for p in TASK_DIR.iterdir() if p.is_dir() and p.name.startswith("scene_")])

    ensure_dir(OUTPUT_TASK_DIR / "train")
    ensure_dir(OUTPUT_TASK_DIR / "calib")

    print(f"task_dir={TASK_DIR}")
    print(f"camera_subdir={CAMERA_SUBDIR}")
    print(f"output_task_dir={OUTPUT_TASK_DIR}")
    print(f"output_train_dir={OUTPUT_TASK_DIR / 'train'}")
    print(f"output_calib_dir={OUTPUT_TASK_DIR / 'calib'}")
    print(f"scene_count={len(scene_dirs)}")

    all_results = []
    for scene_dir in scene_dirs:
        result = process_scene(scene_dir)
        if result is not None:
            all_results.append(result)

    print("\n[all scenes summary]")
    print(f"processed_scene_count={len(all_results)}")
    print(f"total_images={sum(x['total_images'] for x in all_results)}")
    print(f"kept_count={sum(x['kept_count'] for x in all_results)}")
    print(f"skipped_by_pedal_count={sum(x['skipped_by_pedal_count'] for x in all_results)}")


if __name__ == "__main__":
    main()
