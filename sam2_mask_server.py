#!/usr/bin/env python3
"""
SAM2 mask WebSocket server — run in the `sam2` conda environment.

Usage:
    conda run -n sam2 python sam2/sam2_mask_server.py \\
        --config configs/dual_teleop_dino.yaml --port 8765

Protocol (msgpack-numpy, same as WebsocketPolicyServer):
    Client  →  Server : {"color": np.ndarray uint8 HWC}
    Server  →  Client : {"mask": np.ndarray uint8 HW | None, "reason": str | None}

Cold start (first frame) opens a cv2 annotation window in this process.
The client blocks on recv() until annotation is complete.
"""

import argparse
import asyncio
import sys
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import yaml
import websockets.asyncio.server
import websockets.frames

# ── path setup ──────────────────────────────────────────────────────────────
# Run from project root: python sam2_mask_server.py
# sys.path needs project root so that `remote_eval` and `sam2` packages resolve
import os as _os
_project_root = _os.path.dirname(_os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from remote_eval import msgpack_numpy           # existing serialization layer
from sam2.build_sam import build_sam2_video_predictor


# ════════════════════════════════════════════════════════════════════════════
# SAM2 helper functions (self-contained, no rise2-env imports)
# ════════════════════════════════════════════════════════════════════════════

def _resolve_device(device_str):
    device_str = str(device_str).lower()
    if device_str in ("cuda_if_available", "auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if device_str == "cuda":
            raise RuntimeError("device=cuda but CUDA not available")
        return torch.device("cpu")
    if device_str == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("device=mps but MPS not available")
    if device_str == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device: {device_str}")


def _preprocess_frame(predictor, frame_np, device):
    """uint8 HWC numpy → SAM2 tensor (1,3,H,W)."""
    img_size = predictor.image_size
    img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    img_std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    resized  = cv2.resize(frame_np, (img_size, img_size))
    t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    t = (t - img_mean) / img_std
    return t.to(device)[None]


def _build_state_from_frame(predictor, frame_np):
    """Build a fresh inference_state from a single numpy frame."""
    device = predictor.device
    img_t  = _preprocess_frame(predictor, frame_np, device)
    state  = {
        "images":                  img_t,
        "num_frames":              1,
        "video_height":            frame_np.shape[0],
        "video_width":             frame_np.shape[1],
        "device":                  device,
        "storage_device":          device,
        "offload_video_to_cpu":    False,
        "offload_state_to_cpu":    False,
        "point_inputs_per_obj":    {},
        "mask_inputs_per_obj":     {},
        "cached_features":         {},
        "constants":               {},
        "obj_id_to_idx":           OrderedDict(),
        "obj_idx_to_id":           OrderedDict(),
        "obj_ids":                 [],
        "output_dict_per_obj":     {},
        "temp_output_dict_per_obj": {},
        "frames_tracked_per_obj":  {},
    }
    # 兼容层：复现 SAM2VideoPredictor.init_state() 第 58-98 行
    # 升级 SAM2 时需核对字段；_get_image_feature 为私有方法
    predictor._get_image_feature(state, frame_idx=0, batch_size=1)
    return state


def _append_frame(predictor, state, frame_np):
    """Append one new frame to the growing image buffer."""
    device = predictor.device
    img_t  = _preprocess_frame(predictor, frame_np, device)
    state["images"]    = torch.cat([state["images"], img_t], dim=0)
    state["num_frames"] += 1


def _cold_start_interactive(predictor, state, cfg, frame_np):
    """
    Open cv2 window for first-frame annotation.

    Controls: [a]=ARM  [g]=GRIPPER  left-click=pos  right-click=neg
              [r]=reset active obj  Enter/Space=confirm  ESC=abort

    Returns (arm_raw_bool, gripper_raw_bool) or (None, None) on abort.
    """
    ARM_ID  = cfg["arm_obj_id"]
    GRP_ID  = cfg["gripper_obj_id"]
    ARM_BGR = (0, 120, 220)
    GRP_BGR = (0, 140, 255)
    WIN = "SAM2 Cold Start  [a]=ARM [g]=GRIPPER [r]=Reset [Enter/Space]=Done [ESC]=Abort"

    ui = {
        "active": ARM_ID,
        "pts":    {ARM_ID: [], GRP_ID: []},
        "lbs":    {ARM_ID: [], GRP_ID: []},
        "masks":  {ARM_ID: None, GRP_ID: None},
    }

    def _refresh():
        predictor.reset_state(state)
        for oid in [ARM_ID, GRP_ID]:
            if not ui["pts"][oid]:
                continue
            pts_np = np.array(ui["pts"][oid], dtype=np.float32)
            lbs_np = np.array(ui["lbs"][oid],  dtype=np.int32)
            with torch.inference_mode():
                _, obj_ids_out, logits = predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=oid,
                    points=pts_np, labels=lbs_np, normalize_coords=True,
                )
            if oid in list(obj_ids_out):
                idx = list(obj_ids_out).index(oid)
                ui["masks"][oid] = (logits[idx].squeeze().cpu().numpy() > 0.0)

    def _render():
        disp = frame_np.copy().astype(np.float32)
        for oid, bgr in [(ARM_ID, ARM_BGR), (GRP_ID, GRP_BGR)]:
            m = ui["masks"][oid]
            if m is not None:
                c = np.array([bgr[2], bgr[1], bgr[0]], dtype=np.float32)
                disp[m] = disp[m] * 0.55 + c * 0.45
        disp = np.clip(disp, 0, 255).astype(np.uint8)
        for oid, bgr in [(ARM_ID, ARM_BGR), (GRP_ID, GRP_BGR)]:
            for (x, y), lbl in zip(ui["pts"][oid], ui["lbs"][oid]):
                marker = cv2.MARKER_STAR if lbl == 1 else cv2.MARKER_CROSS
                cv2.drawMarker(disp, (int(x), int(y)), bgr, marker, 18, 2)
        name = "ARM" if ui["active"] == ARM_ID else "GRIPPER"
        cv2.putText(disp, f"Active: {name}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ui["pts"][ui["active"]].append([x, y])
            ui["lbs"][ui["active"]].append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            ui["pts"][ui["active"]].append([x, y])
            ui["lbs"][ui["active"]].append(0)
        else:
            return
        _refresh()
        cv2.imshow(WIN, _render())

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)
    cv2.imshow(WIN, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
    print("[sam2-server] Cold start: annotate arm [a] / gripper [g] on frame 0.")
    print("  Left=positive  Right=negative  [r]=reset  Enter/Space=done  ESC=abort")

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord('a'):
            ui["active"] = ARM_ID
            cv2.imshow(WIN, _render())
        elif key == ord('g'):
            ui["active"] = GRP_ID
            cv2.imshow(WIN, _render())
        elif key == ord('r'):
            oid = ui["active"]
            ui["pts"][oid].clear(); ui["lbs"][oid].clear(); ui["masks"][oid] = None
            _refresh(); cv2.imshow(WIN, _render())
        elif key in (13, 32):           # Enter or Space
            cv2.destroyWindow(WIN)
            break
        elif key == 27:                 # ESC
            cv2.destroyWindow(WIN)
            print("[sam2-server] Cold start aborted.")
            return None, None

    arm_raw     = ui["masks"][ARM_ID]
    gripper_raw = ui["masks"][GRP_ID]
    n_arm = sum(l == 1 for l in ui["lbs"][ARM_ID])
    n_grp = sum(l == 1 for l in ui["lbs"][GRP_ID])
    print(f"[sam2-server] Cold start done: arm={n_arm} pts, gripper={n_grp} pts")
    return arm_raw, gripper_raw


def _dilate_mask(mask_bool, radius):
    if radius <= 0:
        return mask_bool
    ks = 2 * int(radius) + 1
    kernel = np.ones((ks, ks), np.uint8)
    return cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1).astype(np.bool_)


def _compute_final_mask(arm_raw, gripper_raw, dilate_radius):
    """final = dilate(arm) AND NOT dilate(gripper)."""
    arm_d = _dilate_mask(arm_raw, dilate_radius)
    if gripper_raw is not None:
        grp_d = _dilate_mask(gripper_raw, dilate_radius)
        return np.logical_and(arm_d, np.logical_not(grp_d))
    return arm_d


# ════════════════════════════════════════════════════════════════════════════
# SAM2MaskServer
# ════════════════════════════════════════════════════════════════════════════

class SAM2MaskServer:
    """WebSocket server that wraps SAM2 VideoPredictor online tracking."""

    def __init__(self, cfg: dict, port: int):
        self._cfg  = cfg
        self._port = port
        self._rt   = None   # runtime dict, initialised in _setup()
        self._executor = ThreadPoolExecutor(max_workers=1)  # serial SAM2 calls
        self._setup()

    def _setup(self):
        device    = _resolve_device(self._cfg["device"])
        predictor = build_sam2_video_predictor(
            config_file = self._cfg["config_file"],
            ckpt_path   = self._cfg["ckpt_path"],
            device      = device,
            mode        = "eval",
        )
        self._rt = {
            "cfg":                  self._cfg,
            "device":               device,
            "predictor":            predictor,
            "inference_state":      None,
            "frame_idx":            0,
            "last_arm_mask_raw":    None,
            "last_gripper_mask_raw": None,
            "last_reset_frame_idx": 0,
            "last_fail_reason":     None,
            "last_frame_np":        None,
        }
        print(f"[sam2-server] model loaded: {self._cfg['ckpt_path']}")

    # ── core inference (blocking, runs in executor thread) ──────────────────

    def _infer(self, color_np: np.ndarray):
        """
        Identical logic to eval_mask_final.infer_mask, using self._rt.
        Returns uint8 (H,W) mask (0/255) or None.
        """
        rt  = self._rt
        cfg = rt["cfg"]
        rt["last_fail_reason"] = None

        if color_np.ndim != 3 or color_np.shape[2] != 3:
            rt["last_fail_reason"] = "sam2_invalid_color"
            return None
        if color_np.dtype != np.uint8:
            color_np = np.clip(color_np, 0, 255).astype(np.uint8)

        predictor = rt["predictor"]

        # ── cold start ──────────────────────────────────────────────────────
        if rt["inference_state"] is None:
            try:
                state = _build_state_from_frame(predictor, color_np)
                arm_raw, gripper_raw = _cold_start_interactive(
                    predictor, state, cfg, color_np
                )
            except Exception:
                rt["last_fail_reason"] = "sam2_cold_start_exception"
                return None
            if arm_raw is None:
                rt["last_fail_reason"] = "sam2_cold_start_aborted"
                return None
            rt["inference_state"]       = state
            rt["frame_idx"]             = 0
            rt["last_arm_mask_raw"]     = arm_raw
            rt["last_gripper_mask_raw"] = gripper_raw
            rt["last_reset_frame_idx"]  = 0
            rt["last_frame_np"]         = color_np.copy()
            final = _compute_final_mask(arm_raw, gripper_raw, cfg["dilate_radius"])
            return (final.astype(np.uint8) * 255)

        # ── subsequent frames ────────────────────────────────────────────────
        state          = rt["inference_state"]
        frame_idx      = rt["frame_idx"]
        last_reset_idx = rt["last_reset_frame_idx"]
        arm_raw_prev   = rt["last_arm_mask_raw"]
        grp_raw_prev   = rt["last_gripper_mask_raw"]

        need_reset = (
            cfg["reset_every_n_steps"] > 0
            and (frame_idx - last_reset_idx) >= cfg["reset_every_n_steps"]
            and arm_raw_prev is not None
        )

        if need_reset:
            last_frame_np = rt.get("last_frame_np")
            try:
                src = last_frame_np if last_frame_np is not None else color_np
                state = _build_state_from_frame(predictor, src)
                with torch.inference_mode():
                    predictor.add_new_mask(
                        state, frame_idx=0,
                        obj_id=cfg["arm_obj_id"],
                        mask=torch.tensor(arm_raw_prev, device=predictor.device),
                    )
                    if grp_raw_prev is not None:
                        predictor.add_new_mask(
                            state, frame_idx=0,
                            obj_id=cfg["gripper_obj_id"],
                            mask=torch.tensor(grp_raw_prev, device=predictor.device),
                        )
                _append_frame(predictor, state, color_np)
                rt["inference_state"]      = state
                rt["frame_idx"]            = 1
                rt["last_reset_frame_idx"] = 1
                frame_idx = 1
            except Exception:
                rt["last_fail_reason"] = "sam2_reset_exception"
                return None
        else:
            try:
                _append_frame(predictor, state, color_np)
                frame_idx += 1
                rt["frame_idx"] = frame_idx
            except Exception:
                rt["last_fail_reason"] = "sam2_append_exception"
                return None

        # ── propagate ───────────────────────────────────────────────────────
        arm_raw = gripper_raw = None
        try:
            with torch.inference_mode():
                for _, obj_ids, mask_logits in predictor.propagate_in_video(
                    state, start_frame_idx=frame_idx, max_frame_num_to_track=1
                ):
                    for i, oid in enumerate(obj_ids):
                        m = (mask_logits[i].squeeze().cpu().numpy() > 0.0)
                        if oid == cfg["arm_obj_id"]:
                            arm_raw = m
                        elif oid == cfg["gripper_obj_id"]:
                            gripper_raw = m
        except Exception:
            rt["last_fail_reason"] = "sam2_propagate_fail"
            return None

        if arm_raw is None:
            rt["last_fail_reason"] = "sam2_propagate_fail"
            return None

        rt["last_arm_mask_raw"] = arm_raw
        if gripper_raw is not None:
            rt["last_gripper_mask_raw"] = gripper_raw
        rt["last_frame_np"] = color_np.copy()

        final = _compute_final_mask(arm_raw, gripper_raw, cfg["dilate_radius"])
        return (final.astype(np.uint8) * 255)

    # ── WebSocket handler ────────────────────────────────────────────────────

    async def _handler(self, websocket):
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack({"status": "ready"}))
        print(f"[sam2-server] client connected: {websocket.remote_address}")

        loop = asyncio.get_event_loop()
        while True:
            try:
                raw = await websocket.recv()
                req = msgpack_numpy.unpackb(raw)

                color_np = np.asarray(req["color"])
                mask = await loop.run_in_executor(self._executor, self._infer, color_np)

                reason = self._rt.get("last_fail_reason") if self._rt else None
                await websocket.send(packer.pack({"mask": mask, "reason": reason}))

            except websockets.ConnectionClosed:
                print(f"[sam2-server] client disconnected: {websocket.remote_address}")
                break
            except Exception:
                tb = traceback.format_exc()
                print(f"[sam2-server] error:\n{tb}")
                try:
                    await websocket.send(tb)
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal error; traceback in previous frame.",
                    )
                except Exception:
                    pass
                break

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            "127.0.0.1",
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            print(f"[sam2-server] listening on 127.0.0.1:{self._port}")
            await server.serve_forever()


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def _load_sam2_cfg_from_yaml(config_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ma   = config.get("mask_aware", {}) or {}
    raw  = ma.get("sam2", {}) or {}

    defaults = {
        "config_file":        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "ckpt_path":          "checkpoints/sam2.1_hiera_base_plus.pt",
        "device":             "cuda_if_available",
        "arm_obj_id":         1,
        "gripper_obj_id":     2,
        "dilate_radius":      10,
        "reset_every_n_steps": 100,
    }
    merged = {**defaults, **{k: v for k, v in raw.items() if v is not None}}
    merged["arm_obj_id"]          = int(merged["arm_obj_id"])
    merged["gripper_obj_id"]      = int(merged["gripper_obj_id"])
    merged["dilate_radius"]       = int(merged["dilate_radius"])
    merged["reset_every_n_steps"] = int(merged["reset_every_n_steps"])
    return merged


def main():
    parser = argparse.ArgumentParser(description="SAM2 mask WebSocket server")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config (e.g. configs/dual_teleop_dino.yaml)")
    parser.add_argument("--port", type=int, default=8765,
                        help="WebSocket port (default: 8765)")
    args = parser.parse_args()

    cfg = _load_sam2_cfg_from_yaml(args.config)
    print(f"[sam2-server] config: {cfg}")

    server = SAM2MaskServer(cfg=cfg, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
