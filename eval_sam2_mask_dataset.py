"""eval_sam2_mask_dataset.py — Dataset offline test mode for SAM2 mask eval pipeline.

Copy of eval_sam2_mask_final.py with an added --dataset branch that loads frames from disk
instead of connecting to a real robot.  Use this to verify SAM2 tracking quality on existing
training data without any robot hardware.

New CLI args (dataset mode):
    --dataset   PATH   Scene directory, e.g. .../task_0012/train/scene_0001  (enables dataset mode)
    --camera_id NAME   Camera sub-directory to read, e.g. cam_105422061350
    --max_frames N     Stop after N frames (overrides config.deploy.max_steps)
    --save_vis  DIR    Save mask-overlay images to this directory

When --dataset is given, --calib_airexo, --calib_rise2, --type are all optional.
Policy inference runs only if --ckpt is provided; otherwise only SAM2 is tested.
"""

import os
import yaml
import torch
import argparse
import numpy as np
import open3d as o3d
import torchvision.transforms as T

from copy import deepcopy
from easydict import EasyDict as edict

from utils.training import set_seed
from utils.ensemble import EnsembleBuffer
# from remote_eval import WebsocketClientPolicy
from eval_agent import SingleArmAgent, DualArmAgent
from dataset.data_utils import resize_image, ImageProcessor
from dataset.projector import SingleArmProjector, DualArmProjector

from collections import OrderedDict

from sam2.build_sam import build_sam2_video_predictor

import cv2


# ── kept for compatibility with any callers that import this module ───────────
test_color = "/data/haoxiang/data/airexo2/task_0013/train/scene_0001/cam_105422061350/color/1737546126606.png"
test_depth = "/data/haoxiang/data/airexo2/task_0013/train/scene_0001/cam_105422061350/depth/1737546126606.png"

fake_intrinsics = np.array([
    [912.4466,   0.     , 633.4127 ],
    [  0.     , 911.4704, 364.21265],
    [  0.     ,   0.    ,   1.     ]
])

fake_depth_scale = 1000.0

test_low_dim = "/data/haoxiang/data/airexo2/task_0013/train/scene_0001/lowdim/1737546126606.npy"


default_args = edict({
    "type": "local",
    "calib_rise2": "calib_rise2/",
    "calib_airexo": "calib_airexo/",
    "config": "config/dual_teleop_dino.yaml",
    "ckpt": None,
    "host": "127.0.0.1",
    "port": 8000,
    # dataset-mode fields (None = not in dataset mode)
    "dataset":    None,
    "camera_id":  None,
    "max_frames": None,
    "save_vis":   None,
})


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-mode helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_frames(scene_dir, camera_id):
    """Scan scene_dir/camera_id/color/ and return sorted list of (color_path, depth_path)."""
    color_dir = os.path.join(scene_dir, camera_id, "color")
    depth_dir = os.path.join(scene_dir, camera_id, "depth")
    if not os.path.isdir(color_dir):
        raise ValueError(f"Color directory not found: {color_dir}")
    if not os.path.isdir(depth_dir):
        raise ValueError(f"Depth directory not found: {depth_dir}")

    timestamps = sorted(
        f.replace(".png", "")
        for f in os.listdir(color_dir)
        if f.endswith(".png")
    )
    if not timestamps:
        raise ValueError(f"No PNG frames found in {color_dir}")

    frames = []
    for ts in timestamps:
        cp = os.path.join(color_dir, f"{ts}.png")
        dp = os.path.join(depth_dir, f"{ts}.png")
        if os.path.isfile(dp):
            frames.append((cp, dp))
        else:
            print(f"[dataset] depth not found for ts={ts}, skipped")
    print(f"[dataset] loaded {len(frames)} frames from {color_dir}")
    return frames


def load_dataset_frame(color_path, depth_path):
    """Load one color (uint8 RGB HWC) + depth (uint16 HW) pair from disk."""
    color = cv2.imread(color_path)
    if color is None:
        raise ValueError(f"Cannot load color image: {color_path}")
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.uint8)

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Cannot load depth image: {depth_path}")
    depth = depth.astype(np.uint16)
    return color, depth


class DatasetAgent:
    """Minimal agent stub for dataset mode.

    Provides the same interface as DualArmAgent / SingleArmAgent that is used
    in evaluate(): get_global_observation(), action(), stop(), intrinsics,
    camera.depth_scale.  All robot-related operations are no-ops.
    """

    def __init__(self, frames, intrinsics, depth_scale):
        self._frames   = frames          # list of (color_path, depth_path)
        self._idx      = 0
        self.intrinsics = intrinsics     # np.ndarray (3, 3)
        self.camera    = edict({"depth_scale": depth_scale})

    def get_global_observation(self):
        if self._idx >= len(self._frames):
            self._idx = 0               # loop back when exhausted
        color_path, depth_path = self._frames[self._idx]
        self._idx += 1
        return load_dataset_frame(color_path, depth_path)

    def action(self, *args, **kwargs):
        pass                            # no-op: no robot to move

    def stop(self):
        pass                            # no-op: no connection to close


def _save_mask_overlay(color_np, mask01, step, out_dir):
    """Save a red-tinted mask overlay image to out_dir/step_XXXXXX_overlay.png."""
    os.makedirs(out_dir, exist_ok=True)
    mask_np  = np.asarray(mask01)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    mask_bool = mask_np > 0

    overlay = color_np.copy().astype(np.float32)
    if np.any(mask_bool):
        overlay[mask_bool] = (
            0.55 * overlay[mask_bool]
            + 0.45 * np.array([255.0, 0.0, 0.0], dtype=np.float32)
        )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    path = os.path.join(out_dir, f"step_{step:06d}_overlay.png")
    cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers (unchanged from eval_sam2_mask_final.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_mask_aware_cfg(config):
    default_cfg = {
        "enabled": False,
        "enable_3d_filter": True,
        "enable_2d_reweight": True,
        "mask_threshold": 0,
        "mask_white_is_untrusted": True,
        "r_min": 1e-3,
        "interp_eps": 1e-6,
        "interp_tiny": 1e-6,
        "infer_allow_none": True,
        "infer_none_policy": "no_mask_fallback",
        "empty_cloud_policy": "warn_and_skip_filter",
        "urdf": None,
        "sam2": {},
    }
    raw_cfg = getattr(config, "mask_aware", {})
    raw_cfg = dict(raw_cfg) if raw_cfg is not None else {}

    merged_cfg = deepcopy(default_cfg)
    for key in default_cfg:
        if key in raw_cfg and raw_cfg[key] is not None:
            merged_cfg[key] = raw_cfg[key]

    valid_none_policy       = {"no_mask_fallback", "fail_fast"}
    valid_empty_cloud_policy = {"warn_and_skip_filter", "fail_fast"}

    try:
        merged_cfg["enabled"]              = bool(merged_cfg["enabled"])
        merged_cfg["enable_3d_filter"]     = bool(merged_cfg["enable_3d_filter"])
        merged_cfg["enable_2d_reweight"]   = bool(merged_cfg["enable_2d_reweight"])
        merged_cfg["mask_threshold"]       = float(merged_cfg["mask_threshold"])
        merged_cfg["mask_white_is_untrusted"] = bool(merged_cfg["mask_white_is_untrusted"])
        merged_cfg["r_min"]                = float(merged_cfg["r_min"])
        merged_cfg["interp_eps"]           = float(merged_cfg["interp_eps"])
        merged_cfg["interp_tiny"]          = float(merged_cfg["interp_tiny"])
        merged_cfg["infer_allow_none"]     = bool(merged_cfg["infer_allow_none"])
        merged_cfg["infer_none_policy"]    = str(merged_cfg["infer_none_policy"])
        merged_cfg["empty_cloud_policy"]   = str(merged_cfg["empty_cloud_policy"])
        merged_cfg["sam2"]                 = dict(merged_cfg["sam2"]) if merged_cfg["sam2"] is not None else {}
    except Exception as exc:
        print(f"[mask-aware] invalid config, fallback to disabled: {exc}")
        merged_cfg = deepcopy(default_cfg)

    if merged_cfg["infer_none_policy"] not in valid_none_policy:
        print("[mask-aware] invalid infer_none_policy, use no_mask_fallback")
        merged_cfg["infer_none_policy"] = "no_mask_fallback"

    if merged_cfg["empty_cloud_policy"] not in valid_empty_cloud_policy:
        print("[mask-aware] invalid empty_cloud_policy, use warn_and_skip_filter")
        merged_cfg["empty_cloud_policy"] = "warn_and_skip_filter"

    return edict(merged_cfg)


def _log_mask_aware_summary(mask_cfg):
    print(
        "[mask-aware] enabled={} 3d_filter={} 2d_reweight={} threshold={} "
        "none_policy={} empty_cloud_policy={}".format(
            mask_cfg.enabled,
            mask_cfg.enable_3d_filter,
            mask_cfg.enable_2d_reweight,
            mask_cfg.mask_threshold,
            mask_cfg.infer_none_policy,
            mask_cfg.empty_cloud_policy,
        )
    )


def _log_mask_fallback(step, reason, action):
    print(f"[mask-aware] step={step} reason={reason} action={action}")


def _build_sam2_cfg(mask_aware_cfg):
    default_cfg = {
        "enabled": True,
        "config_file": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "ckpt_path": "checkpoints/sam2.1_hiera_base_plus.pt",
        "device": "cuda_if_available",
        "arm_obj_id": 1,
        "gripper_obj_id": 2,
        "dilate_radius": 10,
        "reset_every_n_steps": 100,
        "remote_port": None,
    }

    raw_cfg = getattr(mask_aware_cfg, "sam2", {})
    raw_cfg = dict(raw_cfg) if raw_cfg is not None else {}

    merged_cfg = deepcopy(default_cfg)
    for key in default_cfg:
        if key in raw_cfg and raw_cfg[key] is not None:
            merged_cfg[key] = raw_cfg[key]

    try:
        merged_cfg["enabled"]              = bool(merged_cfg["enabled"])
        merged_cfg["config_file"]          = str(merged_cfg["config_file"])
        merged_cfg["ckpt_path"]            = str(merged_cfg["ckpt_path"])
        merged_cfg["device"]               = str(merged_cfg["device"])
        merged_cfg["arm_obj_id"]           = int(merged_cfg["arm_obj_id"])
        merged_cfg["gripper_obj_id"]       = int(merged_cfg["gripper_obj_id"])
        merged_cfg["dilate_radius"]        = int(merged_cfg["dilate_radius"])
        merged_cfg["reset_every_n_steps"]  = int(merged_cfg["reset_every_n_steps"])
        if merged_cfg["remote_port"] is not None:
            merged_cfg["remote_port"]      = int(merged_cfg["remote_port"])
    except Exception as exc:
        print(f"[mask-aware/sam2] invalid config, fallback to disabled: {exc}")
        merged_cfg = deepcopy(default_cfg)
        merged_cfg["enabled"] = False

    return edict(merged_cfg)


def _log_sam2_summary(sam2_cfg):
    print(
        "[mask-aware/sam2] enabled={} cfg={} ckpt={} device={} "
        "arm_obj_id={} gripper_obj_id={} dilate_radius={} reset_every_n_steps={}".format(
            sam2_cfg.enabled,
            sam2_cfg.config_file,
            sam2_cfg.ckpt_path,
            sam2_cfg.device,
            sam2_cfg.arm_obj_id,
            sam2_cfg.gripper_obj_id,
            sam2_cfg.dilate_radius,
            sam2_cfg.reset_every_n_steps,
        )
    )


# ── Module-level SAM2 runtime reference, initialised by evaluate() ───────────
_sam2_runtime = None


def _resolve_sam2_device(device_str):
    device_str = str(device_str).lower()
    if device_str in ["cuda_if_available", "auto", "cuda"]:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if device_str == "cuda":
            raise RuntimeError("SAM2 device is set to cuda but CUDA is not available")
        return torch.device("cpu")
    if device_str == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("SAM2 device is set to mps but MPS is not available")
    if device_str == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported sam2 device: {device_str}")


def _init_sam2_remote_client(port):
    """Connect to sam2_mask_server running in the sam2 conda environment."""
    import time
    import websockets.sync.client
    from remote_eval import msgpack_numpy as _mnp

    uri = f"ws://127.0.0.1:{port}"
    packer = _mnp.Packer()
    while True:
        try:
            conn = websockets.sync.client.connect(uri, compression=None, max_size=None)
            _mnp.unpackb(conn.recv())   # wait for {"status": "ready"} handshake
            print(f"[mask-aware/sam2] connected to remote server at {uri}")
            return {
                "mode":             "remote",
                "enabled":          True,
                "conn":             conn,
                "packer":           packer,
                "last_fail_reason": None,
            }
        except ConnectionRefusedError:
            print(f"[mask-aware/sam2] waiting for sam2 server on port {port}...")
            time.sleep(2)


def _init_sam2_runtime(sam2_cfg):
    if getattr(sam2_cfg, "remote_port", None) is not None:
        return _init_sam2_remote_client(sam2_cfg.remote_port)

    device = _resolve_sam2_device(sam2_cfg.device)
    predictor = build_sam2_video_predictor(
        config_file=sam2_cfg.config_file,
        ckpt_path=sam2_cfg.ckpt_path,
        device=device,
        mode="eval",
    )
    return {
        "enabled":             True,
        "cfg":                 sam2_cfg,
        "device":              device,
        "predictor":           predictor,
        "inference_state":     None,
        "frame_idx":           0,
        "last_arm_mask_raw":   None,
        "last_gripper_mask_raw": None,
        "last_reset_frame_idx": 0,
        "last_fail_reason":    None,
        "last_frame_np":       None,
    }


def _extract_step_from_meta(meta):
    if not isinstance(meta, dict):
        return None
    step = meta.get("step", None)
    if step is None:
        return None
    try:
        return int(step)
    except Exception:
        return None


def _preprocess_frame_for_sam2(predictor, frame_np, device):
    """Resize and normalize a uint8 HWC numpy frame to a SAM2-ready tensor (1,3,H,W)."""
    img_size = predictor.image_size
    img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    img_std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    frame_resized = cv2.resize(frame_np, (img_size, img_size))
    frame_tensor  = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
    frame_tensor  = (frame_tensor - img_mean) / img_std
    return frame_tensor.to(device)[None]


def _build_video_inference_state_from_np(predictor, frame_np):
    """Manually construct a SAM2VideoPredictor inference_state from a single numpy frame."""
    device       = predictor.device
    frame_tensor = _preprocess_frame_for_sam2(predictor, frame_np, device)
    inference_state = {
        "images":                frame_tensor,
        "num_frames":            1,
        "video_height":          frame_np.shape[0],
        "video_width":           frame_np.shape[1],
        "device":                device,
        "storage_device":        device,
        "offload_video_to_cpu":  False,
        "offload_state_to_cpu":  False,
        "point_inputs_per_obj":  {},
        "mask_inputs_per_obj":   {},
        "cached_features":       {},
        "constants":             {},
        "obj_id_to_idx":         OrderedDict(),
        "obj_idx_to_id":         OrderedDict(),
        "obj_ids":               [],
        "output_dict_per_obj":   {},
        "temp_output_dict_per_obj": {},
        "frames_tracked_per_obj":   {},
    }
    predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    return inference_state


def _append_frame_to_video_state(predictor, inference_state, frame_np):
    """Append one new frame to the growing video buffer in inference_state."""
    device       = predictor.device
    frame_tensor = _preprocess_frame_for_sam2(predictor, frame_np, device)
    inference_state["images"] = torch.cat([inference_state["images"], frame_tensor], dim=0)
    inference_state["num_frames"] += 1


def _cold_start_interactive(predictor, inference_state, cfg, frame_np):
    """Interactive first-frame annotation via cv2.

    Controls
    --------
    [a]          Switch active object to ARM
    [g]          Switch active object to GRIPPER
    Left-click   Add positive point
    Right-click  Add negative point
    [r]          Reset points for the active object
    Enter/Space  Confirm
    ESC          Abort (returns None, None)
    """
    ARM_OBJ_ID   = cfg.arm_obj_id
    GRP_OBJ_ID   = cfg.gripper_obj_id
    ARM_COLOR_BGR = (0, 120, 220)
    GRP_COLOR_BGR = (0, 140, 255)
    WIN = "SAM2 Cold Start  [a]=ARM [g]=GRIPPER [r]=Reset [Enter/Space]=Done [ESC]=Abort"

    state = {
        "active_obj": ARM_OBJ_ID,
        "points":     {ARM_OBJ_ID: [], GRP_OBJ_ID: []},
        "labels":     {ARM_OBJ_ID: [], GRP_OBJ_ID: []},
        "masks":      {ARM_OBJ_ID: None, GRP_OBJ_ID: None},
    }

    def _refresh_masks():
        predictor.reset_state(inference_state)
        for oid in [ARM_OBJ_ID, GRP_OBJ_ID]:
            if not state["points"][oid]:
                continue
            pts_np = np.array(state["points"][oid], dtype=np.float32)
            lbs_np = np.array(state["labels"][oid],  dtype=np.int32)
            with torch.inference_mode():
                _, obj_ids_out, mask_logits = predictor.add_new_points_or_box(
                    inference_state,
                    frame_idx=0,
                    obj_id=oid,
                    points=pts_np,
                    labels=lbs_np,
                    normalize_coords=True,
                )
            if oid in list(obj_ids_out):
                idx = list(obj_ids_out).index(oid)
                state["masks"][oid] = (mask_logits[idx].squeeze().cpu().numpy() > 0.0)

    def _render():
        disp = frame_np.copy().astype(np.float32)
        for oid, bgr in [(ARM_OBJ_ID, ARM_COLOR_BGR), (GRP_OBJ_ID, GRP_COLOR_BGR)]:
            m = state["masks"][oid]
            if m is not None:
                c_rgb = np.array([bgr[2], bgr[1], bgr[0]], dtype=np.float32)
                disp[m] = disp[m] * 0.55 + c_rgb * 0.45
        disp = np.clip(disp, 0, 255).astype(np.uint8)
        for oid, bgr in [(ARM_OBJ_ID, ARM_COLOR_BGR), (GRP_OBJ_ID, GRP_COLOR_BGR)]:
            for (x, y), lbl in zip(state["points"][oid], state["labels"][oid]):
                marker = cv2.MARKER_STAR if lbl == 1 else cv2.MARKER_CROSS
                cv2.drawMarker(disp, (int(x), int(y)), bgr, marker, 18, 2)
        obj_name = "ARM" if state["active_obj"] == ARM_OBJ_ID else "GRIPPER"
        cv2.putText(disp, f"Active: {obj_name}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["points"][state["active_obj"]].append([x, y])
            state["labels"][state["active_obj"]].append(1)
            _refresh_masks()
            cv2.imshow(WIN, _render())
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["points"][state["active_obj"]].append([x, y])
            state["labels"][state["active_obj"]].append(0)
            _refresh_masks()
            cv2.imshow(WIN, _render())

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)
    cv2.imshow(WIN, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
    print("[mask-aware/sam2] Cold start: annotate arm [a] and gripper [g] on frame 0.")
    print("  Left=positive  Right=negative  [r]=reset active obj  Enter/Space=done  ESC=abort")

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord('a'):
            state["active_obj"] = ARM_OBJ_ID
            print(f"[cold-start] active -> ARM (obj={ARM_OBJ_ID})")
            cv2.imshow(WIN, _render())
        elif key == ord('g'):
            state["active_obj"] = GRP_OBJ_ID
            print(f"[cold-start] active -> GRIPPER (obj={GRP_OBJ_ID})")
            cv2.imshow(WIN, _render())
        elif key == ord('r'):
            oid = state["active_obj"]
            state["points"][oid].clear()
            state["labels"][oid].clear()
            state["masks"][oid] = None
            _refresh_masks()
            cv2.imshow(WIN, _render())
            print(f"[cold-start] reset obj={oid}")
        elif key in (13, 32):
            cv2.destroyWindow(WIN)
            break
        elif key == 27:
            cv2.destroyWindow(WIN)
            print("[cold-start] ESC: aborting cold start")
            return None, None

    arm_raw     = state["masks"][ARM_OBJ_ID]
    gripper_raw = state["masks"][GRP_OBJ_ID]
    n_arm = sum(l == 1 for l in state["labels"][ARM_OBJ_ID])
    n_grp = sum(l == 1 for l in state["labels"][GRP_OBJ_ID])
    print(f"[cold-start] confirmed: arm={n_arm} pos pts, gripper={n_grp} pos pts")
    return arm_raw, gripper_raw


def _dilate_mask_bool(mask_bool, radius):
    if radius <= 0:
        return mask_bool
    ks = 2 * int(radius) + 1
    kernel = np.ones((ks, ks), np.uint8)
    return cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1).astype(np.bool_)


def _compute_final_mask(arm_raw, gripper_raw, dilate_radius):
    arm_dilated = _dilate_mask_bool(arm_raw, dilate_radius)
    if gripper_raw is not None:
        grp_dilated = _dilate_mask_bool(gripper_raw, dilate_radius)
        return np.logical_and(arm_dilated, np.logical_not(grp_dilated))
    return arm_dilated


def infer_mask(color, depth, proprio, meta, agent=None):
    """Run SAM2 VideoPredictor online tracking for arm (obj=1) and gripper (obj=2).

    Returns uint8 (H,W) mask (255=arm region) or None on failure.
    """
    del depth, proprio, agent

    global _sam2_runtime
    if _sam2_runtime is None or not bool(_sam2_runtime.get("enabled", False)):
        return None

    # ── remote mode ──────────────────────────────────────────────────────────
    if _sam2_runtime.get("mode") == "remote":
        from remote_eval import msgpack_numpy as _mnp
        color_np = np.asarray(color)
        if color_np.dtype != np.uint8:
            color_np = np.clip(color_np, 0, 255).astype(np.uint8)
        try:
            _sam2_runtime["conn"].send(_sam2_runtime["packer"].pack({"color": color_np}))
            resp = _mnp.unpackb(_sam2_runtime["conn"].recv())
            _sam2_runtime["last_fail_reason"] = resp.get("reason")
            return resp.get("mask")
        except Exception as exc:
            _sam2_runtime["last_fail_reason"] = "sam2_remote_exception"
            print(f"[mask-aware/sam2] remote call failed: {exc}")
            return None

    # ── local mode ───────────────────────────────────────────────────────────
    cfg = _sam2_runtime["cfg"]
    _sam2_runtime["last_fail_reason"] = None

    color_np = np.asarray(color)
    if color_np.ndim != 3 or color_np.shape[2] != 3:
        _sam2_runtime["last_fail_reason"] = "sam2_invalid_color"
        return None
    if color_np.dtype != np.uint8:
        color_np = np.clip(color_np, 0, 255).astype(np.uint8)

    predictor = _sam2_runtime["predictor"]

    # ── cold start ───────────────────────────────────────────────────────────
    if _sam2_runtime["inference_state"] is None:
        try:
            inference_state = _build_video_inference_state_from_np(predictor, color_np)
            arm_raw, gripper_raw = _cold_start_interactive(
                predictor, inference_state, cfg, color_np
            )
        except Exception:
            _sam2_runtime["last_fail_reason"] = "sam2_cold_start_exception"
            return None
        if arm_raw is None:
            _sam2_runtime["last_fail_reason"] = "sam2_cold_start_aborted"
            return None
        _sam2_runtime["inference_state"]       = inference_state
        _sam2_runtime["frame_idx"]             = 0
        _sam2_runtime["last_arm_mask_raw"]     = arm_raw
        _sam2_runtime["last_gripper_mask_raw"] = gripper_raw
        _sam2_runtime["last_reset_frame_idx"]  = 0
        _sam2_runtime["last_frame_np"]         = color_np.copy()
        final = _compute_final_mask(arm_raw, gripper_raw, cfg.dilate_radius)
        return (final.astype(np.uint8) * 255)

    # ── subsequent frames ─────────────────────────────────────────────────────
    inference_state  = _sam2_runtime["inference_state"]
    frame_idx        = _sam2_runtime["frame_idx"]
    last_reset_idx   = _sam2_runtime["last_reset_frame_idx"]
    arm_raw_prev     = _sam2_runtime["last_arm_mask_raw"]
    gripper_raw_prev = _sam2_runtime["last_gripper_mask_raw"]

    need_reset = (
        cfg.reset_every_n_steps > 0
        and (frame_idx - last_reset_idx) >= cfg.reset_every_n_steps
        and arm_raw_prev is not None
    )

    if need_reset:
        last_frame_np = _sam2_runtime.get("last_frame_np")
        try:
            if last_frame_np is not None:
                inference_state = _build_video_inference_state_from_np(predictor, last_frame_np)
            else:
                inference_state = _build_video_inference_state_from_np(predictor, color_np)
            with torch.inference_mode():
                predictor.add_new_mask(
                    inference_state, frame_idx=0,
                    obj_id=cfg.arm_obj_id,
                    mask=torch.tensor(arm_raw_prev, device=predictor.device),
                )
                if gripper_raw_prev is not None:
                    predictor.add_new_mask(
                        inference_state, frame_idx=0,
                        obj_id=cfg.gripper_obj_id,
                        mask=torch.tensor(gripper_raw_prev, device=predictor.device),
                    )
            _append_frame_to_video_state(predictor, inference_state, color_np)
            _sam2_runtime["inference_state"]      = inference_state
            _sam2_runtime["frame_idx"]            = 1
            _sam2_runtime["last_reset_frame_idx"] = 1
            frame_idx = 1
        except Exception:
            _sam2_runtime["last_fail_reason"] = "sam2_reset_exception"
            return None
    else:
        try:
            _append_frame_to_video_state(predictor, inference_state, color_np)
            frame_idx += 1
            _sam2_runtime["frame_idx"] = frame_idx
        except Exception:
            _sam2_runtime["last_fail_reason"] = "sam2_append_exception"
            return None

    arm_raw     = None
    gripper_raw = None
    try:
        with torch.inference_mode():
            for _, obj_ids, mask_logits in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=1,
            ):
                for i, oid in enumerate(obj_ids):
                    m = (mask_logits[i].squeeze().cpu().numpy() > 0.0)
                    if oid == cfg.arm_obj_id:
                        arm_raw = m
                    elif oid == cfg.gripper_obj_id:
                        gripper_raw = m
    except Exception:
        _sam2_runtime["last_fail_reason"] = "sam2_propagate_fail"
        return None

    if arm_raw is None:
        _sam2_runtime["last_fail_reason"] = "sam2_propagate_fail"
        return None

    _sam2_runtime["last_arm_mask_raw"]     = arm_raw
    _sam2_runtime["last_gripper_mask_raw"] = gripper_raw  # always overwrite; None clears stale
    _sam2_runtime["last_frame_np"]         = color_np.copy()

    final = _compute_final_mask(arm_raw, gripper_raw, cfg.dilate_radius)
    return (final.astype(np.uint8) * 255)


def _to_numpy_mask(mask):
    if isinstance(mask, torch.Tensor):
        return mask.detach().cpu().numpy()
    return np.asarray(mask)


def _normalize_mask01(mask, depth_shape, mask_cfg):
    mask_np = _to_numpy_mask(mask)

    if mask_np.ndim == 3:
        if mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        elif mask_np.shape[-1] == 1:
            mask_np = mask_np[..., 0]
        else:
            raise ValueError(f"unsupported mask shape: {mask_np.shape}")
    elif mask_np.ndim != 2:
        raise ValueError(f"unsupported mask shape: {mask_np.shape}")

    target_h, target_w = int(depth_shape[0]), int(depth_shape[1])
    if mask_np.shape[0] != target_h or mask_np.shape[1] != target_w:
        print(
            "[mask-aware/3donly] mask size mismatch, resize with nearest: "
            f"mask={mask_np.shape}, depth=({target_h}, {target_w})"
        )
        mask_np = cv2.resize(
            mask_np.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

    threshold = float(mask_cfg.mask_threshold)
    if bool(mask_cfg.mask_white_is_untrusted):
        mask01 = (mask_np > threshold).astype(np.float32)
    else:
        mask01 = (mask_np <= threshold).astype(np.float32)

    return mask01


def _safe_infer_mask(color, depth, proprio, meta, mask_cfg, agent=None):
    try:
        raw_mask = infer_mask(color, depth, proprio, meta, agent=agent)
    except Exception:
        return None, "infer_exception", None

    if raw_mask is None:
        return None, "infer_none", None

    try:
        mask01 = _normalize_mask01(raw_mask, depth.shape[:2], mask_cfg)
    except Exception:
        return None, "mask_invalid", raw_mask

    return mask01, None, raw_mask


def _save_mask_visualization(colors, mask01, step, config):
    if mask01 is None:
        return

    vis_save_dir = getattr(config.deploy, "vis_save_dir", ".")
    if vis_save_dir is None or len(str(vis_save_dir).strip()) == 0:
        vis_save_dir = "."
    os.makedirs(vis_save_dir, exist_ok=True)

    vis_save_prefix = getattr(config.deploy, "vis_save_prefix", "vis_debug")
    if vis_save_prefix is None or len(str(vis_save_prefix).strip()) == 0:
        vis_save_prefix = "vis_debug"
    vis_save_prefix = str(vis_save_prefix)

    mask_np  = np.asarray(mask01)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    mask_u8 = ((mask_np > 0).astype(np.uint8) * 255)

    overlay = np.asarray(colors, dtype=np.uint8).copy()
    mask_bool = mask_u8 > 0
    if np.any(mask_bool):
        overlay_f32 = overlay.astype(np.float32)
        overlay_f32[mask_bool] = (
            0.6 * overlay_f32[mask_bool]
            + 0.4 * np.array([255.0, 0.0, 0.0], dtype=np.float32)
        )
        overlay = np.clip(overlay_f32, 0, 255).astype(np.uint8)

    mask_path    = os.path.join(vis_save_dir, "{}_step_{:06d}_mask.png".format(vis_save_prefix, step))
    overlay_path = os.path.join(vis_save_dir, "{}_step_{:06d}_mask_overlay.png".format(vis_save_prefix, step))
    cv2.imwrite(mask_path, mask_u8)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("[vis] saved mask: {}".format(mask_path))
    print("[vis] saved mask overlay: {}".format(overlay_path))


def _build_image_mask_weight(mask01, image_processor):
    try:
        mask_tensor = torch.from_numpy(mask01[np.newaxis].astype(np.float32))
        mask_tensor = resize_image(
            mask_tensor,
            image_processor.img_size,
            interpolation=T.InterpolationMode.NEAREST,
        )
        mask_ratio = image_processor.image_coord_pooling(mask_tensor)
        image_mask_weight = (1.0 - mask_ratio).clamp(0.0, 1.0).to(torch.float32)
    except Exception:
        return None

    return image_mask_weight


def load_test_obs(color_path, depth_path):
    color_image = cv2.imread(color_path)
    if color_image is None:
        raise ValueError(f"无法加载图片: {color_path}")
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB).astype(np.uint8)

    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError(f"无法加载深度图: {depth_path}")
    depth_image = depth_image.astype(np.uint16)

    return color_image, depth_image


def create_point_cloud(colors, depths, intrinsics, config, depth_scale=1000.0, rescale_factor=1):
    if rescale_factor != 1:
        H, W = depths.shape
        h, w = int(H * rescale_factor), int(W * rescale_factor)
        colors = colors.transpose([2, 0, 1]).astype(np.float32)
        colors = torch.from_numpy(colors)
        colors = np.ascontiguousarray(resize_image(colors, [h, w]).numpy().transpose([1, 2, 0]))
        depths = depths.astype(np.float32)
        depths = torch.from_numpy(depths[np.newaxis])
        depths = resize_image(depths, [h, w], interpolation=T.InterpolationMode.NEAREST)[0]
        depths = depths.numpy()

    h, w = depths.shape
    fx, fy = intrinsics[0, 0] * rescale_factor, intrinsics[1, 1] * rescale_factor
    cx, cy = intrinsics[0, 2] * rescale_factor, intrinsics[1, 2] * rescale_factor
    colors = o3d.geometry.Image(colors.astype(np.uint8))
    depths = o3d.geometry.Image(depths.astype(np.float32))
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors, depths, depth_scale, convert_rgb_to_intensity=False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
    bbox3d = o3d.geometry.AxisAlignedBoundingBox(
        config.deploy.workspace.min, config.deploy.workspace.max
    )
    cloud = cloud.crop(bbox3d)
    cloud = cloud.voxel_down_sample(config.data.voxel_size)
    return cloud


def create_input(colors, depths, cam_intrinsics, config, depth_scale=1000.0, rescale_factor=1):
    cloud  = create_point_cloud(colors, depths, cam_intrinsics, config,
                                depth_scale=depth_scale, rescale_factor=rescale_factor)
    points = np.asarray(cloud.points)
    coords = np.ascontiguousarray(points / config.data.voxel_size, dtype=np.int32)
    return coords, points, cloud


def create_batch(coords, points):
    import MinkowskiEngine as ME
    coords_batch, feats_batch = ME.utils.sparse_collate([coords], [points.astype(np.float32)])
    return coords_batch, feats_batch


def process_state(state, config, to_control=True):
    if config.robot_type == "single":
        if to_control:
            state[..., 0:3] = (state[..., 0:3] + 1) / 2.0 * (config.data.normalization.trans_max - config.data.normalization.trans_min) + config.data.normalization.trans_min
            state[..., 9]   = (state[..., 9] + 1) / 2.0 * config.data.normalization.max_gripper_width
        else:
            state[..., 0:3] = (state[..., 0:3] - config.data.normalization.trans_min) / (config.data.normalization.trans_max - config.data.normalization.trans_min) * 2.0 - 1
            state[..., 9]   = state[..., 9] / config.data.normalization.max_gripper_width * 2.0 - 1
    else:
        if to_control:
            state[..., 0:3]   = (state[..., 0:3] + 1) / 2.0 * (config.data.normalization.trans_max - config.data.normalization.trans_min) + config.data.normalization.trans_min
            state[..., 10:13] = (state[..., 10:13] + 1) / 2.0 * (config.data.normalization.trans_max - config.data.normalization.trans_min) + config.data.normalization.trans_min
            state[..., 9]     = (state[..., 9] + 1) / 2.0 * config.data.normalization.max_gripper_width
            state[..., 19]    = (state[..., 19] + 1) / 2.0 * config.data.normalization.max_gripper_width
        else:
            state[..., 0:3]   = (state[..., 0:3] - config.data.normalization.trans_min) / (config.data.normalization.trans_max - config.data.normalization.trans_min) * 2.0 - 1
            state[..., 10:13] = (state[..., 10:13] - config.data.normalization.trans_min) / (config.data.normalization.trans_max - config.data.normalization.trans_min) * 2.0 - 1
            state[..., 9]     = state[..., 9] / config.data.normalization.max_gripper_width * 2.0 - 1
            state[..., 19]    = state[..., 19] / config.data.normalization.max_gripper_width * 2.0 - 1
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluate()
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args_override):
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # ── detect dataset mode ───────────────────────────────────────────────────
    dataset_mode = bool(args.get("dataset"))
    if dataset_mode:
        args.type = "local"   # force local; robot and projector are bypassed
        print(f"[dataset] dataset mode: scene={args.dataset}  camera={args.camera_id}")

    # load config
    with open(args.config, "r") as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
    config.data.normalization.trans_min = np.asarray(config.data.normalization.trans_min)
    config.data.normalization.trans_max = np.asarray(config.data.normalization.trans_max)
    config.mask_aware      = _build_mask_aware_cfg(config)
    config.mask_aware.sam2 = _build_sam2_cfg(config.mask_aware)

    set_seed(config.deploy.seed)

    # ── policy (optional in dataset mode) ────────────────────────────────────
    policy = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.type == "local" and args.get("ckpt"):
        from policy import RISE2
        print("Loading policy ...")
        policy = RISE2(
            num_action        = config.data.num_action,
            obs_feature_dim   = config.model.obs_feature_dim,
            cloud_enc_dim     = config.model.cloud_enc_dim,
            image_enc_dim     = config.model.image_enc_dim,
            action_dim        = 10 if config.robot_type == "single" else 20,
            hidden_dim        = config.model.hidden_dim,
            nheads            = config.model.nheads,
            num_attn_layers   = config.model.num_attn_layers,
            dim_feedforward   = config.model.dim_feedforward,
            dropout           = config.model.dropout,
            image_enc         = config.model.image_enc,
            interp_fn_mode    = config.model.interp_fn_mode,
            image_enc_finetune = config.model.image_enc_finetune,
            image_enc_dtype   = config.model.image_enc_dtype,
        ).to(device)
        policy.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
        print(f"Checkpoint {args.ckpt} loaded.")
        policy.eval()
    elif not dataset_mode:
        if args.type == "local":
            # real robot + local policy: ckpt is required
            assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
        else:
            print("Connecting to remote server ...")
            from remote_eval import WebsocketClientPolicy
            policy = WebsocketClientPolicy(host=args.host, port=args.port)
    else:
        print("[dataset] no --ckpt provided; only SAM2 tracking will be tested (no policy inference).")

    # ── projector (skip in dataset mode) ─────────────────────────────────────
    projector = None
    if not dataset_mode:
        Projector = SingleArmProjector if config.robot_type == "single" else DualArmProjector
        projector = Projector(args.calib_rise2, config.deploy.agent.camera_serial)

    # ── image processor ───────────────────────────────────────────────────────
    image_enc = config.model.image_enc
    if image_enc == "resnet18":
        img_size       = config.data.aligner.img_size_resnet
        img_coord_size = config.data.aligner.img_coord_size_resnet
    elif image_enc.startswith("dinov2"):
        img_size       = config.data.aligner.img_size_dinov2
        img_coord_size = config.data.aligner.img_coord_size_dinov2
    elif image_enc.startswith("dinov3"):
        img_size       = config.data.aligner.img_size_dinov3
        img_coord_size = config.data.aligner.img_coord_size_dinov3
    else:
        raise ValueError(f"Unknown image encoder: {image_enc}")

    image_processor = ImageProcessor(
        img_size       = img_size,
        img_coord_size = img_coord_size,
        voxel_size     = config.data.voxel_size,
        img_mean       = config.data.normalization.img_mean,
        img_std        = config.data.normalization.img_std,
    )

    # ── agent ─────────────────────────────────────────────────────────────────
    if dataset_mode:
        frames = load_dataset_frames(args.dataset, args.camera_id)
        agent  = DatasetAgent(frames, fake_intrinsics, fake_depth_scale)
    else:
        Agent = SingleArmAgent if config.robot_type == "single" else DualArmAgent
        agent = Agent(**config.deploy.agent)

    ensemble_buffer = EnsembleBuffer(mode=config.deploy.ensemble_mode)

    _log_mask_aware_summary(config.mask_aware)
    _log_sam2_summary(config.mask_aware.sam2)

    # ── SAM2 runtime ──────────────────────────────────────────────────────────
    global _sam2_runtime
    _sam2_runtime = None
    if config.mask_aware.enabled and config.mask_aware.sam2.enabled:
        try:
            _sam2_runtime = _init_sam2_runtime(config.mask_aware.sam2)
            print("[mask-aware/sam2] runtime initialized")
        except Exception as _e:
            print(f"[mask-aware/sam2] runtime init failed, mask disabled: {_e}")
            _sam2_runtime = None

    mask_stats = {
        "infer_none": 0,
        "infer_exception": 0,
        "mask_invalid": 0,
        "empty_cloud_skip": 0,
        "reweight_fallback": 0,
        "points_nonfinite": 0,
        "weight_nonfinite": 0,
        "sam2_propagate_fail": 0,
        "sam2_invalid_color": 0,
        "sam2_reset_exception": 0,
        "sam2_append_exception": 0,
        "sam2_remote_exception": 0,
        "sam2_cold_start_exception": 0,
        "sam2_cold_start_aborted": 0,
    }

    if not dataset_mode:
        print("Ready for rollout. Press Enter to continue...")
        input()

    # mask_enabled: SAM2 remote mode uses _sam2_runtime["mode"]=="remote"; local mode
    # requires args.type=="local".  In dataset mode we also allow remote SAM2 server.
    mask_enabled = bool(
        config.mask_aware.enabled
        and (args.type == "local" or (
            _sam2_runtime is not None and _sam2_runtime.get("mode") == "remote"
        ))
        and (_sam2_runtime is not None)
    )
    mask01 = None

    # determine total loop count
    if dataset_mode and args.get("max_frames"):
        total_steps = int(args.max_frames)
    elif dataset_mode:
        total_steps = len(agent._frames)
    else:
        total_steps = config.deploy.max_steps

    save_vis_dir = args.get("save_vis")

    with torch.inference_mode():
        for t in range(total_steps):
            colors_raw, depths = agent.get_global_observation()

            # ── SAM2 every step ───────────────────────────────────────────────
            if mask_enabled:
                mask01, mask_reason, raw_mask = _safe_infer_mask(
                    color    = colors_raw,
                    depth    = depths,
                    proprio  = None,
                    meta     = {"step": t, "mode": args.type},
                    mask_cfg = config.mask_aware,
                    agent    = agent,
                )
                # dataset mode: always save overlay if --save_vis given
                if save_vis_dir and mask01 is not None:
                    _save_mask_overlay(colors_raw, mask01, t, save_vis_dir)
                if getattr(config.deploy, "vis", False) and mask01 is not None:
                    _save_mask_visualization(colors_raw, mask01, t, config)
                if mask01 is None:
                    reason = mask_reason or "unknown_infer_failure"
                    if reason in mask_stats:
                        mask_stats[reason] += 1
                    if _sam2_runtime is not None:
                        runtime_reason = _sam2_runtime.get("last_fail_reason", None)
                        if runtime_reason in mask_stats:
                            mask_stats[runtime_reason] += 1
                    if t % config.deploy.num_inference_steps == 0:
                        if config.mask_aware.infer_none_policy == "fail_fast":
                            _log_mask_fallback(t, reason, "fail_fast")
                            raise RuntimeError(f"mask unavailable with fail_fast, reason={reason}")
                        _log_mask_fallback(t, reason, "no_mask_fallback")

            # ── inference step: point cloud + policy ──────────────────────────
            if t % config.deploy.num_inference_steps == 0 and policy is not None:
                depths_for_cloud = depths
                if mask_enabled and config.mask_aware.enable_3d_filter and mask01 is not None:
                    depths_for_cloud = depths.copy()
                    depths_for_cloud[mask01 > 0.5] = 0

                create_input_kwargs = dict(
                    cam_intrinsics = agent.intrinsics,
                    config         = config,
                    depth_scale    = agent.camera.depth_scale,
                    rescale_factor = 1.0,
                )
                coords, points, cloud = create_input(
                    colors_raw, depths_for_cloud, **create_input_kwargs
                )

                if points.size > 0 and (not np.isfinite(points).all()):
                    if config.mask_aware.empty_cloud_policy == "fail_fast":
                        _log_mask_fallback(t, "points_nonfinite", "fail_fast")
                        raise RuntimeError("non-finite points after cloud build")
                    mask_stats["points_nonfinite"] += 1
                    _log_mask_fallback(t, "points_nonfinite", "rebuild_from_original_depth")
                    coords, points, cloud = create_input(
                        colors_raw, depths, **create_input_kwargs
                    )

                if (
                    mask_enabled
                    and config.mask_aware.enable_3d_filter
                    and mask01 is not None
                    and points.shape[0] == 0
                ):
                    if config.mask_aware.empty_cloud_policy == "fail_fast":
                        _log_mask_fallback(t, "empty_cloud_after_3d_filter", "fail_fast")
                        raise RuntimeError("empty cloud after 3d filter")
                    mask_stats["empty_cloud_skip"] += 1
                    _log_mask_fallback(t, "empty_cloud_after_3d_filter", "skip_filter_rebuild")
                    coords, points, cloud = create_input(
                        colors_raw, depths, **create_input_kwargs
                    )

                image_coords = image_processor.get_image_coordinates(
                    depths, agent.intrinsics, agent.camera.depth_scale
                )
                colors_proc, image_coords = image_processor.preprocess_images(
                    colors_raw, image_coords
                )

                image_mask_weight = None
                if mask_enabled and config.mask_aware.enable_2d_reweight and mask01 is not None:
                    image_mask_weight = _build_image_mask_weight(mask01, image_processor)
                    if image_mask_weight is None:
                        mask_stats["reweight_fallback"] += 1
                        _log_mask_fallback(t, "reweight_build_failed", "disable_2d_reweight")
                    elif not torch.isfinite(image_mask_weight).all():
                        mask_stats["weight_nonfinite"] += 1
                        _log_mask_fallback(t, "reweight_nonfinite", "disable_2d_reweight")
                        image_mask_weight = None

                if args.type == "local":
                    import MinkowskiEngine as ME
                    coords_batch, feats_batch = create_batch(coords, points)
                    coords_batch = coords_batch.to(device)
                    feats_batch  = feats_batch.to(device)
                    cloud_data   = ME.SparseTensor(feats_batch, coords_batch)

                    colors_proc_dev    = colors_proc.unsqueeze(0).to(device)
                    image_coords_dev   = image_coords.unsqueeze(0).to(device)
                    if image_mask_weight is not None:
                        image_mask_weight = image_mask_weight.unsqueeze(0).to(device)

                    pred_raw_action = policy(
                        cloud_data,
                        colors_proc_dev,
                        image_coords_dev,
                        image_mask_weight = image_mask_weight,
                        actions           = None,
                    )
                else:
                    obs_dict = {
                        "coords":       coords,
                        "points":       points,
                        "colors":       colors_proc.numpy(),
                        "image_coords": image_coords.numpy(),
                    }
                    pred_raw_action = policy.infer(obs_dict)["actions"]

                action = process_state(pred_raw_action, config, to_control=True)

                # projector: skipped in dataset mode
                if projector is not None:
                    if config.robot_type == "single":
                        action_tcp = projector.project_tcp_to_base_coord(
                            action[..., :9], rotation_rep="rotation_6d"
                        )
                        action = np.concatenate([action_tcp, action[..., 9:10]], axis=-1)
                    else:
                        action_left_tcp  = projector.project_tcp_to_base_coord(
                            action[..., :9], "left", rotation_rep="rotation_6d"
                        )
                        action_right_tcp = projector.project_tcp_to_base_coord(
                            action[..., 10:19], "right", rotation_rep="rotation_6d"
                        )
                        action = np.concatenate(
                            [action_left_tcp, action[..., 9:10],
                             action_right_tcp, action[..., 19:20]], axis=-1
                        )

                ensemble_buffer.add_action(action, t)

            # ── execute step action ───────────────────────────────────────────
            step_action = ensemble_buffer.get_action()
            if step_action is None:
                continue
            agent.action(step_action, rotation_rep="rotation_6d")
            if not dataset_mode:
                print(f"execute {step_action}")

    print(
        "[mask-aware] summary infer_none={} infer_exception={} mask_invalid={} "
        "empty_cloud_skip={} reweight_fallback={} points_nonfinite={} weight_nonfinite={} "
        "sam2_propagate_fail={} sam2_invalid_color={} "
        "sam2_reset_exception={} sam2_append_exception={} sam2_remote_exception={} "
        "sam2_cold_start_exception={} sam2_cold_start_aborted={}".format(
            mask_stats["infer_none"],
            mask_stats["infer_exception"],
            mask_stats["mask_invalid"],
            mask_stats["empty_cloud_skip"],
            mask_stats["reweight_fallback"],
            mask_stats["points_nonfinite"],
            mask_stats["weight_nonfinite"],
            mask_stats["sam2_propagate_fail"],
            mask_stats["sam2_invalid_color"],
            mask_stats["sam2_reset_exception"],
            mask_stats["sam2_append_exception"],
            mask_stats["sam2_remote_exception"],
            mask_stats["sam2_cold_start_exception"],
            mask_stats["sam2_cold_start_aborted"],
        )
    )

    if dataset_mode and save_vis_dir:
        print(f"[dataset] visualizations saved to: {save_vis_dir}")

    agent.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM2 mask eval — supports real robot and dataset offline test modes."
    )

    # ── dataset mode args ─────────────────────────────────────────────────────
    parser.add_argument("--dataset",    type=str, default=None,
                        help="Enable dataset mode: path to scene dir, e.g. .../scene_0001")
    parser.add_argument("--camera_id", type=str, default=None,
                        help="Camera sub-directory to read, e.g. cam_105422061350")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Stop after N frames (dataset mode only)")
    parser.add_argument("--save_vis",   type=str, default=None,
                        help="Save mask-overlay PNGs to this directory (dataset mode)")

    # ── standard args (required for real robot, optional in dataset mode) ─────
    parser.add_argument("--type",          type=str, default="local",
                        choices=["local", "remote"],
                        help="Evaluation type (default: local)")
    parser.add_argument("--calib_airexo",  type=str, default=None,
                        help="AirExo calibration path (not needed in dataset mode)")
    parser.add_argument("--calib_rise2",   type=str, default=None,
                        help="Rise2 calibration path (not needed in dataset mode)")
    parser.add_argument("--config",        type=str, required=True,
                        help="YAML config file path")
    parser.add_argument("--ckpt",          type=str, default=None,
                        help="Checkpoint path (optional in dataset mode)")
    parser.add_argument("--host",          type=str, default="127.0.0.1",
                        help="Remote server host")
    parser.add_argument("--port",          type=int, default=8000,
                        help="Remote server port")

    parsed = parser.parse_args()

    # validate: if dataset mode, camera_id is required
    if parsed.dataset and not parsed.camera_id:
        parser.error("--camera_id is required when --dataset is specified")

    evaluate(vars(parsed))
