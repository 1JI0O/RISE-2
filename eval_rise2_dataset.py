"""eval_rise2_dataset.py — Offline dataset evaluation for vanilla RISE2 policy.

This script reuses the same policy inference path as deployment, but loads color/depth
frames from an existing dataset scene directory instead of connecting to real robots.

Key points:
- No SAM2 / mask-aware filtering is used.
- Supports local policy checkpoint inference and remote websocket policy inference.
- Uses DatasetAgent (no-op action/stop) to mimic rollout loop behavior.
"""

import yaml
import torch
import argparse
import numpy as np
import open3d as o3d

from copy import deepcopy
from easydict import EasyDict as edict

from utils.training import set_seed
from utils.ensemble import EnsembleBuffer
from remote_eval import WebsocketClientPolicy
from dataset.data_utils import ImageProcessor

from eval_sam2_mask_dataset import (
    load_dataset_frames,
    DatasetAgent,
    fake_intrinsics,
    fake_depth_scale,
    create_input,
    create_batch,
    process_state,
)


default_args = edict({
    "type": "local",
    "config": "configs/dual_teleop_dino_rise2_dataset.yaml",
    "ckpt": None,
    "host": "127.0.0.1",
    "port": 8000,
    "dataset": None,
    "camera_id": None,
    "max_frames": None,
})


def _build_image_processor(config):
    image_enc = config.model.image_enc
    if image_enc == "resnet18":
        img_size = config.data.aligner.img_size_resnet
        img_coord_size = config.data.aligner.img_coord_size_resnet
    elif image_enc.startswith("dinov2"):
        img_size = config.data.aligner.img_size_dinov2
        img_coord_size = config.data.aligner.img_coord_size_dinov2
    elif image_enc.startswith("dinov3"):
        img_size = config.data.aligner.img_size_dinov3
        img_coord_size = config.data.aligner.img_coord_size_dinov3
    else:
        raise ValueError(f"Unknown image encoder: {image_enc}")

    return ImageProcessor(
        img_size=img_size,
        img_coord_size=img_coord_size,
        voxel_size=config.data.voxel_size,
        img_mean=config.data.normalization.img_mean,
        img_std=config.data.normalization.img_std,
    )


def _load_local_policy(args, config, device):
    from policy import RISE2

    assert args.ckpt is not None, "Please provide --ckpt for local offline evaluation."

    print("Loading policy ...")
    policy = RISE2(
        num_action=config.data.num_action,
        obs_feature_dim=config.model.obs_feature_dim,
        cloud_enc_dim=config.model.cloud_enc_dim,
        image_enc_dim=config.model.image_enc_dim,
        action_dim=10 if config.robot_type == "single" else 20,
        hidden_dim=config.model.hidden_dim,
        nheads=config.model.nheads,
        num_attn_layers=config.model.num_attn_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        image_enc=config.model.image_enc,
        interp_fn_mode=config.model.interp_fn_mode,
        image_enc_finetune=config.model.image_enc_finetune,
        image_enc_dtype=config.model.image_enc_dtype,
    ).to(device)

    policy.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    print(f"Checkpoint {args.ckpt} loaded.")
    policy.eval()
    return policy


def _visualize_policy_step(config, cloud, action):
    if not getattr(config.deploy, "vis", False):
        return

    tcp_vis_list = []
    for raw_tcp in action:
        tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(raw_tcp[:3])
        tcp_vis_list.append(tcp_vis)
        if config.robot_type == "dual":
            tcp_vis_r = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(raw_tcp[10:13])
            tcp_vis_list.append(tcp_vis_r)

    o3d.visualization.draw_geometries([cloud, *tcp_vis_list])
    # input("press enter")


def evaluate(args_override):
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # dataset mode required for this script
    if not args.get("dataset"):
        raise ValueError("--dataset is required for eval_rise2_dataset.py")
    if not args.get("camera_id"):
        raise ValueError("--camera_id is required for eval_rise2_dataset.py")

    print(f"[dataset] vanilla rise2 mode: scene={args.dataset}  camera={args.camera_id}")

    with open(args.config, "r") as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))

    config.data.normalization.trans_min = np.asarray(config.data.normalization.trans_min)
    config.data.normalization.trans_max = np.asarray(config.data.normalization.trans_max)

    # Explicitly disable mask_aware branch for vanilla RISE2 offline eval.
    if hasattr(config, "mask_aware") and bool(getattr(config.mask_aware, "enabled", False)):
        print("[dataset] mask_aware.enabled=True in config, force disabled for vanilla rise2 eval.")
    config.mask_aware = edict({"enabled": False})

    set_seed(config.deploy.seed)
    image_processor = _build_image_processor(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.type == "local":
        policy = _load_local_policy(args, config, device)
    else:
        print("Connecting to remote server ...")
        policy = WebsocketClientPolicy(host=args.host, port=args.port)

    frames = load_dataset_frames(args.dataset, args.camera_id)
    agent = DatasetAgent(frames, fake_intrinsics, fake_depth_scale)
    ensemble_buffer = EnsembleBuffer(mode=config.deploy.ensemble_mode)

    if args.get("max_frames") is not None:
        total_steps = min(int(args.max_frames), len(frames))
    else:
        total_steps = len(frames)

    with torch.inference_mode():
        for t in range(total_steps):
            colors_raw, depths = agent.get_global_observation()

            if t % config.deploy.num_inference_steps == 0:
                coords, points, cloud = create_input(
                    colors_raw,
                    depths,
                    cam_intrinsics=agent.intrinsics,
                    config=config,
                    depth_scale=agent.camera.depth_scale,
                    rescale_factor=1.0,
                )

                image_coords = image_processor.get_image_coordinates(
                    depths,
                    agent.intrinsics,
                    agent.camera.depth_scale,
                )
                colors_proc, image_coords = image_processor.preprocess_images(colors_raw, image_coords)

                if args.type == "local":
                    import MinkowskiEngine as ME

                    coords_batch, feats_batch = create_batch(coords, points)
                    coords_batch = coords_batch.to(device)
                    feats_batch = feats_batch.to(device)
                    cloud_data = ME.SparseTensor(feats_batch, coords_batch)

                    colors_proc_dev = colors_proc.unsqueeze(0).to(device)
                    image_coords_dev = image_coords.unsqueeze(0).to(device)

                    pred_raw_action = policy(
                        cloud_data,
                        colors_proc_dev,
                        image_coords_dev,
                        image_mask_weight=None,
                        actions=None,
                    ).squeeze(0).cpu().numpy()
                else:
                    obs_dict = {
                        "coords": coords,
                        "points": points,
                        "colors": colors_proc.numpy(),
                        "image_coords": image_coords.numpy(),
                    }
                    pred_raw_action = deepcopy(policy.infer(obs_dict)["actions"])

                action = process_state(pred_raw_action, config, to_control=True)

                _visualize_policy_step(config, cloud, action)
                print(f"[policy] infer step={t} action={action}")
                ensemble_buffer.add_action(action, t)

            step_action = ensemble_buffer.get_action()
            if step_action is None:
                continue

            agent.action(step_action, rotation_rep="rotation_6d")
            print(f"execute {step_action}")

    print("[dataset] vanilla rise2 evaluation finished")
    agent.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vanilla RISE2 offline dataset evaluation (no SAM2 / no mask-aware)."
    )

    parser.add_argument("--dataset", type=str, required=True,
                        help="Scene dir, e.g. .../task_xxxx/train/scene_0001")
    parser.add_argument("--camera_id", type=str, required=True,
                        help="Camera sub-directory, e.g. cam_105422061350")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Stop after N frames (default: full scene)")

    parser.add_argument("--type", type=str, default="local", choices=["local", "remote"],
                        help="Policy inference mode")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config path")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Policy checkpoint path (required in local mode)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Remote server host (remote mode)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Remote server port (remote mode)")

    parsed = parser.parse_args()
    evaluate(vars(parsed))
