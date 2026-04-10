#!/usr/bin/env python3
"""
从机械臂 merged mask 中减去抓取物体 mask（先对抓取物体 mask 膨胀），并输出新 mask。

处理规则：
1) 对每个 scene_xxxx：
   - 读取机械臂 mask 目录：scene_xxxx
   - 读取抓取物体 mask 目录：scene_xxxx_toy_text_prompt
2) 两侧按文件名排序后必须严格一一对应（文件名列表完全一致），否则直接抛错。
3) 抓取物体 mask 先膨胀（默认半径 10 像素）后，从机械臂 mask 中扣除。
4) 输出到：merged_masks_without_toys/scene_xxxx

输出为 0/255 的单通道 uint8 mask。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


DEFAULT_ARM_MASK_ROOT = Path(
    "/data/haoxiang/data/task0012_260321/task0012_toys_basket_converted/merged_masks"
)
DEFAULT_TOY_MASK_ROOT = Path(
    "/data/haoxiang/data/task0012_260321_masks_sam3/task0012_toys_basket_converted"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/haoxiang/data/task0012_260321/task0012_toys_basket_converted/merged_masks_without_toys"
)

SCENE_REGEX = re.compile(r"^scene_\d{4}$")
TOY_SCENE_REGEX = re.compile(r"^scene_\d{4}_toy_text_prompt$")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 merged robot masks 中减去膨胀后的 toy masks。"
    )
    parser.add_argument(
        "--arm-mask-root",
        type=Path,
        default=DEFAULT_ARM_MASK_ROOT,
        help="机械臂 mask 根目录（含 scene_xxxx 子目录）",
    )
    parser.add_argument(
        "--toy-mask-root",
        type=Path,
        default=DEFAULT_TOY_MASK_ROOT,
        help=(
            "抓取物体 mask 根目录（含 scene_xxxx_toy_text_prompt 子目录）。"
            "如果误传 scene_0001_toy_text_prompt 这种具体 scene 目录，会自动取其父目录。"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="输出根目录（会创建 scene_xxxx 子目录）",
    )
    parser.add_argument(
        "--dilate-radius",
        type=int,
        default=10,
        help="抓取物体 mask 的膨胀半径（像素），默认 10",
    )
    parser.add_argument(
        "--expected-scenes",
        type=int,
        default=52,
        help="期望 scene 数量，默认 52。<=0 表示不检查。",
    )
    parser.add_argument(
        "--toy-scene-suffix",
        type=str,
        default="_toy_text_prompt",
        help="抓取物体 scene 目录后缀，默认 _toy_text_prompt",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每个 scene 的详细处理日志",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭进度条输出",
    )
    return parser.parse_args()


def list_scene_dirs(root: Path, regex: re.Pattern[str]) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在: {root}")
    scenes = [p for p in root.iterdir() if p.is_dir() and regex.match(p.name)]
    return sorted(scenes, key=lambda p: p.name)


def list_image_files(scene_dir: Path) -> list[Path]:
    files = [
        p
        for p in scene_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(files, key=lambda p: p.name)


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"读取 mask 失败: {path}")
    return mask


def ensure_toy_root(toy_mask_root: Path) -> Path:
    """
    用户可能把 --toy-mask-root 传成某个具体 scene 目录（如 scene_0001_toy_text_prompt）。
    这里自动向上兼容为其父目录。
    """
    if TOY_SCENE_REGEX.match(toy_mask_root.name):
        parent = toy_mask_root.parent
        if parent.is_dir():
            return parent
    return toy_mask_root


def build_kernel(dilate_radius: int) -> np.ndarray:
    if dilate_radius < 0:
        raise ValueError(f"dilate_radius 不能小于 0，当前为 {dilate_radius}")
    if dilate_radius == 0:
        return np.ones((1, 1), dtype=np.uint8)
    k = 2 * dilate_radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def process_scene(
    arm_scene_dir: Path,
    toy_scene_dir: Path,
    output_scene_dir: Path,
    kernel: np.ndarray,
    show_progress: bool,
) -> int:
    arm_files = list_image_files(arm_scene_dir)
    toy_files = list_image_files(toy_scene_dir)

    if len(arm_files) != len(toy_files):
        raise RuntimeError(
            f"scene={arm_scene_dir.name} 文件数量不一致: "
            f"arm={len(arm_files)} toy={len(toy_files)}"
        )

    arm_names = [p.name for p in arm_files]
    toy_names = [p.name for p in toy_files]
    if arm_names != toy_names:
        mismatch_idx = next(i for i, (a, b) in enumerate(zip(arm_names, toy_names)) if a != b)
        raise RuntimeError(
            f"scene={arm_scene_dir.name} 文件名不一致（排序后无法一一对应）: "
            f"idx={mismatch_idx}, arm={arm_names[mismatch_idx]}, toy={toy_names[mismatch_idx]}"
        )

    output_scene_dir.mkdir(parents=True, exist_ok=True)

    name_iter = arm_names
    if show_progress:
        name_iter = tqdm(arm_names, desc=f"{arm_scene_dir.name}", unit="img", leave=False)

    for name in name_iter:
        arm_path = arm_scene_dir / name
        toy_path = toy_scene_dir / name
        out_path = output_scene_dir / name

        arm_mask = read_mask(arm_path)
        toy_mask = read_mask(toy_path)

        if arm_mask.shape != toy_mask.shape:
            raise RuntimeError(
                f"scene={arm_scene_dir.name}, file={name} 尺寸不一致: "
                f"arm={arm_mask.shape}, toy={toy_mask.shape}"
            )

        arm_bin = (arm_mask > 0).astype(np.uint8)
        toy_bin = (toy_mask > 0).astype(np.uint8)

        toy_dilated = cv2.dilate(toy_bin, kernel, iterations=1)
        out_bin = arm_bin & (1 - np.clip(toy_dilated, 0, 1))

        out_mask = (out_bin * 255).astype(np.uint8)
        ok = cv2.imwrite(str(out_path), out_mask)
        if not ok:
            raise RuntimeError(f"写出 mask 失败: {out_path}")

    return len(arm_names)


def main() -> None:
    args = parse_args()

    arm_root = args.arm_mask_root
    toy_root = ensure_toy_root(args.toy_mask_root)
    out_root = args.output_root

    kernel = build_kernel(args.dilate_radius)

    arm_scenes = list_scene_dirs(arm_root, SCENE_REGEX)
    toy_scenes = list_scene_dirs(toy_root, TOY_SCENE_REGEX)

    if args.expected_scenes > 0:
        if len(arm_scenes) != args.expected_scenes:
            raise RuntimeError(
                f"机械臂 scene 数量不符合预期: actual={len(arm_scenes)}, expected={args.expected_scenes}"
            )
        if len(toy_scenes) != args.expected_scenes:
            raise RuntimeError(
                f"抓取物体 scene 数量不符合预期: actual={len(toy_scenes)}, expected={args.expected_scenes}"
            )

    total_files = 0
    scene_iter = arm_scenes
    if not args.no_progress:
        scene_iter = tqdm(arm_scenes, desc="Scenes", unit="scene")

    for arm_scene in scene_iter:
        toy_scene = toy_root / f"{arm_scene.name}{args.toy_scene_suffix}"
        if not toy_scene.is_dir():
            raise FileNotFoundError(
                f"缺少对应抓取物体 scene 目录: arm_scene={arm_scene.name}, toy_scene={toy_scene}"
            )

        out_scene = out_root / arm_scene.name
        count = process_scene(
            arm_scene_dir=arm_scene,
            toy_scene_dir=toy_scene,
            output_scene_dir=out_scene,
            kernel=kernel,
            show_progress=not args.no_progress,
        )
        total_files += count

        if args.verbose:
            print(
                f"[done] scene={arm_scene.name}, files={count}, out_dir={out_scene}"
            )

    print("\n处理完成")
    print(f"arm_root      : {arm_root}")
    print(f"toy_root      : {toy_root}")
    print(f"output_root   : {out_root}")
    print(f"scene_count   : {len(arm_scenes)}")
    print(f"total_images  : {total_files}")
    print(f"dilate_radius : {args.dilate_radius}")


if __name__ == "__main__":
    main()
