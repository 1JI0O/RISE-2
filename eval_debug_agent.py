import os
import h5py
import numpy as np

from copy import deepcopy
from easydict import EasyDict as edict
from PIL import Image

from utils.transformation import xyz_rot_transform


class EvalDebugAgent:
    def __init__(
        self,
        scene_dir,
        camera_serial,
        intrinsics,
        depth_scale=1000.0,
        camera_dir=None,
        frame_id=None,
        stream=False,
        loop=False,
        robot_type="dual",
        h5_path=None,
        left_suffix=None,
        right_suffix=None,
        gripper_key="width",
    ):
        self.scene_dir = os.path.abspath(scene_dir)
        self.camera_serial = str(camera_serial)
        self.robot_type = str(robot_type)
        self.stream = bool(stream)
        self.loop = bool(loop)
        self.intrinsics = np.asarray(intrinsics, dtype=np.float32)
        self.camera = edict({"depth_scale": float(depth_scale)})
        self.gripper_key = str(gripper_key)
        self.h5_path = None if h5_path is None else os.path.abspath(h5_path)
        self.left_suffix = left_suffix
        self.right_suffix = right_suffix
        self._h5_cache = None
        self._h5_ts = None
        self._step_idx = 0
        self._last_frame_id = None
        self._last_action = None

        self.camera_dir = self._resolve_camera_dir(camera_dir)
        self.color_dir = os.path.join(self.camera_dir, "color")
        self.depth_dir = os.path.join(self.camera_dir, "depth")
        self.lowdim_dir = os.path.join(self.scene_dir, "lowdim")
        self.frame_ids = self._discover_frame_ids()
        if len(self.frame_ids) == 0:
            raise RuntimeError(f"no frames found under {self.color_dir}")

        if frame_id is None:
            self._fixed_frame_id = int(self.frame_ids[0])
            self._start_index = 0
        else:
            frame_id = int(frame_id)
            if frame_id not in self.frame_ids:
                raise KeyError(f"frame_id {frame_id} not found in {self.color_dir}")
            self._fixed_frame_id = frame_id
            self._start_index = self.frame_ids.index(frame_id)

        self.max_steps = len(self.frame_ids) - self._start_index if self.stream else 1
        if self.max_steps <= 0:
            self.max_steps = 1

    def _resolve_camera_dir(self, camera_dir):
        if camera_dir is not None:
            return os.path.abspath(camera_dir)
        direct_dir = os.path.join(self.scene_dir, f"cam_{self.camera_serial}")
        if os.path.isdir(direct_dir):
            return direct_dir
        camera_dirs = sorted(
            [
                os.path.join(self.scene_dir, name)
                for name in os.listdir(self.scene_dir)
                if name.startswith("cam_")
                and os.path.isdir(os.path.join(self.scene_dir, name))
            ]
        )
        if len(camera_dirs) == 0:
            raise RuntimeError(f"no camera directory found under {self.scene_dir}")
        return camera_dirs[0]

    def _discover_frame_ids(self):
        frame_ids = []
        for name in sorted(os.listdir(self.color_dir)):
            stem, ext = os.path.splitext(name)
            if ext.lower() != ".png":
                continue
            depth_path = os.path.join(self.depth_dir, name)
            if os.path.isfile(depth_path):
                frame_ids.append(int(stem))
        return frame_ids

    def _select_frame_id(self):
        if not self.stream:
            return self._fixed_frame_id

        idx = self._start_index + self._step_idx
        if idx >= len(self.frame_ids):
            if not self.loop:
                idx = len(self.frame_ids) - 1
            else:
                span = len(self.frame_ids) - self._start_index
                idx = self._start_index + ((idx - self._start_index) % max(1, span))
        self._step_idx += 1
        return int(self.frame_ids[idx])

    def _load_rgbd(self, frame_id):
        color_path = os.path.join(self.color_dir, f"{frame_id}.png")
        depth_path = os.path.join(self.depth_dir, f"{frame_id}.png")
        color = np.asarray(Image.open(color_path).convert("RGB"), dtype=np.uint8)
        depth = np.asarray(Image.open(depth_path), dtype=np.uint16)
        return color, depth

    def _load_lowdim(self, frame_id):
        lowdim_path = os.path.join(self.lowdim_dir, f"{frame_id}.npy")
        if not os.path.isfile(lowdim_path):
            raise FileNotFoundError(f"lowdim not found: {lowdim_path}")
        data = np.load(lowdim_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            data = data.item()
        if not isinstance(data, dict):
            raise TypeError(f"invalid lowdim format: {lowdim_path}")
        return data

    def _ensure_h5_open(self):
        if self.h5_path is None:
            raise RuntimeError("debug joint source h5 is not provided")
        if self._h5_cache is not None:
            return
        self._h5_cache = h5py.File(self.h5_path, "r")
        self._h5_ts = np.asarray(self._h5_cache["timestamp"][:], dtype=np.int64)
        if self.left_suffix is None or self.right_suffix is None:
            suffixes = sorted(
                {
                    key.split("joint_position_rad_")[-1]
                    for key in self._h5_cache.keys()
                    if key.startswith("joint_position_rad_")
                }
            )
            if self.left_suffix is None and len(suffixes) >= 1:
                self.left_suffix = suffixes[0]
            if self.right_suffix is None and len(suffixes) >= 2:
                self.right_suffix = suffixes[1]
            if self.right_suffix is None:
                self.right_suffix = self.left_suffix
        if self.left_suffix is None or self.right_suffix is None:
            raise RuntimeError("failed to determine left/right h5 suffix")

    def _load_joint_from_h5(self, frame_id):
        self._ensure_h5_open()
        idx = int(np.argmin(np.abs(self._h5_ts - int(frame_id))))
        left_joint7 = np.asarray(
            self._h5_cache[f"joint_position_rad_{self.left_suffix}"][idx],
            dtype=np.float32,
        )
        right_joint7 = np.asarray(
            self._h5_cache[f"joint_position_rad_{self.right_suffix}"][idx],
            dtype=np.float32,
        )
        left_gripper = float(
            np.asarray(self._h5_cache[f"ee_state_{self.left_suffix}"][idx]).reshape(-1)[
                0
            ]
        )
        right_gripper = float(
            np.asarray(self._h5_cache[f"ee_state_{self.right_suffix}"][idx]).reshape(
                -1
            )[0]
        )
        left_joint = np.concatenate(
            [left_joint7, np.array([left_gripper], dtype=np.float32)], axis=0
        )
        right_joint = np.concatenate(
            [right_joint7, np.array([right_gripper], dtype=np.float32)], axis=0
        )
        return left_joint, right_joint

    def _get_gripper_value(self, arr):
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return 0.0
        if self.gripper_key == "action" and arr.size >= 2:
            return float(arr[1])
        return float(arr[0])

    def get_global_observation(self):
        frame_id = self._select_frame_id()
        self._last_frame_id = frame_id
        return self._load_rgbd(frame_id)

    def get_proprio(
        self, rotation_rep="rotation_6d", rotation_rep_convention=None, with_joint=False
    ):
        if self._last_frame_id is None:
            self._last_frame_id = self._select_frame_id()

        lowdim = self._load_lowdim(self._last_frame_id)
        if self.robot_type != "dual":
            raise NotImplementedError(
                "EvalDebugAgent currently supports dual-arm debug playback only"
            )

        left_tcp = np.asarray(lowdim["robot_left"], dtype=np.float32).reshape(-1)[:7]
        right_tcp = np.asarray(lowdim["robot_right"], dtype=np.float32).reshape(-1)[:7]
        left_gripper = self._get_gripper_value(lowdim["gripper_left"])
        right_gripper = self._get_gripper_value(lowdim["gripper_right"])

        left_tcp = xyz_rot_transform(
            left_tcp,
            from_rep="quaternion",
            to_rep=rotation_rep,
            to_convention=rotation_rep_convention,
        )
        right_tcp = xyz_rot_transform(
            right_tcp,
            from_rep="quaternion",
            to_rep=rotation_rep,
            to_convention=rotation_rep_convention,
        )

        proprio = np.concatenate(
            [left_tcp, [left_gripper], right_tcp, [right_gripper]], axis=0
        )
        if not with_joint:
            return proprio

        left_joint, right_joint = self._load_joint_from_h5(self._last_frame_id)
        proprio_joint = np.concatenate(
            [left_joint[:7], [left_gripper], right_joint[:7], [right_gripper]], axis=0
        ).astype(np.float32)
        return proprio, proprio_joint

    def action(self, action, rotation_rep="rotation_6d", rotation_rep_convention=None):
        self._last_action = deepcopy(np.asarray(action, dtype=np.float32))

    def stop(self):
        if self._h5_cache is not None:
            try:
                self._h5_cache.close()
            except Exception:
                pass
            self._h5_cache = None
