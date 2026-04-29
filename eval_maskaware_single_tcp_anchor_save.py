import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from copy import deepcopy

import eval_maskaware_single_tcp_anchor_deploy as core


default_args = deepcopy(core.default_args)
default_args["save_root"] = "deploy_capture_single_tcp_anchor"


def _format_step_action_for_save(step_action, action_dim):
    action_dim = int(action_dim)
    save_action = np.full((action_dim,), np.nan, dtype=np.float32)
    if step_action is None:
        return save_action
    try:
        action_np = np.asarray(step_action, dtype=np.float32).reshape(-1)
    except Exception:
        return save_action
    if action_np.size <= 0:
        return save_action
    copy_n = min(action_dim, int(action_np.size))
    save_action[:copy_n] = action_np[:copy_n]
    return save_action


def _format_mask_u8_for_save(mask01, depth_shape):
    h, w = int(depth_shape[0]), int(depth_shape[1])
    if mask01 is None:
        return np.zeros((h, w), dtype=np.uint8)
    mask_np = np.asarray(mask01)
    if mask_np.ndim == 3:
        if mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        elif mask_np.shape[-1] == 1:
            mask_np = mask_np[..., 0]
        else:
            mask_np = mask_np[..., 0]
    if mask_np.shape[:2] != (h, w):
        mask_np = cv2.resize(
            mask_np.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
        )
    return (mask_np > 0.5).astype(np.uint8) * 255


def _build_capture_dirs(save_root):
    session_name = time.strftime("capture_%Y%m%d_%H%M%S")
    session_root = os.path.abspath(os.path.join(save_root, session_name))
    subdirs = {
        "rgb": "rgb",
        "depth": "depth",
        "mask": "mask",
        "actions": "actions",
        "proprio": "proprio",
        "joint": "joint",
        "tcp_camera": "tcp_camera",
        "anchor_points": "anchor_points",
    }
    out = {"root": session_root}
    for key, name in subdirs.items():
        path = os.path.join(session_root, name)
        os.makedirs(path, exist_ok=True)
        out[key] = path
    return out


def _save_session_meta(session_root, args, config):
    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "calib_rise2": str(args.calib_rise2),
        "camera_serial": str(config.deploy.agent.camera_serial),
        "robot_type": str(config.robot_type),
        "num_inference_steps": int(config.deploy.num_inference_steps),
        "max_steps": int(config.deploy.max_steps),
        "tcp_anchor_enabled": bool(config.tcp_anchor.enabled),
        "anchor_num_points": int(config.tcp_anchor.anchor_num_points),
        "anchor_radius_scale": float(config.tcp_anchor.anchor_radius_scale),
    }
    with open(os.path.join(session_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _save_step_bundle(
    capture_dirs,
    step,
    colors,
    depths,
    mask01,
    step_action,
    proprio,
    proprio_joint,
    tcp_camera,
    anchor_points,
    action_dim,
):
    stem = f"step_{int(step):06d}"
    colors_u8 = np.asarray(colors, dtype=np.uint8)
    depth_np = np.asarray(depths)
    if np.issubdtype(depth_np.dtype, np.integer):
        depth_u16 = depth_np.astype(np.uint16)
    else:
        depth_u16 = np.clip(np.rint(depth_np), 0, np.iinfo(np.uint16).max).astype(
            np.uint16
        )
    mask_u8 = _format_mask_u8_for_save(mask01, depth_u16.shape[:2])
    action_np = _format_step_action_for_save(step_action, action_dim)

    cv2.imwrite(
        os.path.join(capture_dirs["rgb"], f"{stem}.png"),
        cv2.cvtColor(colors_u8, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(os.path.join(capture_dirs["depth"], f"{stem}.png"), depth_u16)
    cv2.imwrite(os.path.join(capture_dirs["mask"], f"{stem}.png"), mask_u8)
    np.save(
        os.path.join(capture_dirs["actions"], f"{stem}.npy"),
        action_np,
        allow_pickle=False,
    )

    if proprio is not None:
        np.save(
            os.path.join(capture_dirs["proprio"], f"{stem}.npy"),
            np.asarray(proprio, dtype=np.float32),
            allow_pickle=False,
        )
    if proprio_joint is not None:
        np.save(
            os.path.join(capture_dirs["joint"], f"{stem}.npy"),
            np.asarray(proprio_joint, dtype=np.float32),
            allow_pickle=False,
        )
    if tcp_camera is not None:
        np.save(
            os.path.join(capture_dirs["tcp_camera"], f"{stem}.npy"),
            np.asarray(tcp_camera, dtype=np.float32),
            allow_pickle=False,
        )
    if anchor_points is not None:
        np.save(
            os.path.join(capture_dirs["anchor_points"], f"{stem}.npy"),
            np.asarray(anchor_points, dtype=np.float32),
            allow_pickle=False,
        )


def _read_save_proprio(agent, step):
    try:
        tcp_pose = np.asarray(agent.robot.get_tcp_pose(), dtype=np.float32).reshape(-1)
        tcp_pose = core.xyz_rot_transform(
            tcp_pose, from_rep="quaternion", to_rep="rotation_6d", to_convention=None
        )
        joint_pos = np.asarray(agent.robot.get_joint_pos(), dtype=np.float32).reshape(
            -1
        )
        gripper_width = float(
            np.asarray(agent.gripper.get_states()["width"], dtype=np.float32).reshape(
                -1
            )[0]
        )
        proprio = np.concatenate(
            [tcp_pose, np.array([gripper_width], dtype=np.float32)], axis=0
        )
        proprio_joint = np.concatenate(
            [joint_pos[:7], np.array([gripper_width], dtype=np.float32)], axis=0
        )
        return proprio.astype(np.float32), proprio_joint.astype(np.float32)
    except Exception as exc:
        core._warn(
            f"step={step} failed to read proprio/joint for save via raw API: {exc}"
        )
        return None, None


def _read_save_joint_only(agent, step):
    try:
        joint_pos = np.asarray(agent.robot.get_joint_pos(), dtype=np.float32).reshape(
            -1
        )
        gripper_width = float(
            np.asarray(agent.gripper.get_states()["width"], dtype=np.float32).reshape(
                -1
            )[0]
        )
        return np.concatenate(
            [joint_pos[:7], np.array([gripper_width], dtype=np.float32)], axis=0
        ).astype(np.float32)
    except Exception as exc:
        core._warn(f"step={step} failed to read joint-only save payload: {exc}")
        return None


def evaluate(args_override):
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    with open(args.config, "r", encoding="utf-8") as f:
        config = core.edict(core.yaml.load(f, Loader=core.yaml.FullLoader))
    config.data.normalization.trans_min = np.asarray(
        config.data.normalization.trans_min
    )
    config.data.normalization.trans_max = np.asarray(
        config.data.normalization.trans_max
    )
    config.mask_aware = core._build_mask_aware_cfg(config)
    config.tcp_anchor = core._build_tcp_anchor_cfg(config)

    if str(config.robot_type) != "single":
        raise ValueError(
            f"eval_maskaware_single_tcp_anchor_save.py only supports robot_type=single, got {config.robot_type}"
        )

    core.set_seed(config.deploy.seed)
    capture_dirs = _build_capture_dirs(args.save_root)
    _save_session_meta(capture_dirs["root"], args, config)
    print(f"[capture] saving deployment observations to {capture_dirs['root']}")

    if args.type == "local":
        from policy import RISE2

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading policy ...")
        policy = RISE2(
            num_action=config.data.num_action,
            obs_feature_dim=config.model.obs_feature_dim,
            cloud_enc_dim=config.model.cloud_enc_dim,
            image_enc_dim=config.model.image_enc_dim,
            action_dim=10,
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
        assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
        policy.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
        print(f"Checkpoint {args.ckpt} loaded.")
        policy.eval()
    else:
        raise NotImplementedError(
            "remote inference is not wired in this tcp-anchor capture eval script"
        )

    projector = core.SingleArmProjector(
        args.calib_rise2, config.deploy.agent.camera_serial
    )

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

    image_processor = core.ImageProcessor(
        img_size=img_size,
        img_coord_size=img_coord_size,
        voxel_size=config.data.voxel_size,
        img_mean=config.data.normalization.img_mean,
        img_std=config.data.normalization.img_std,
    )

    agent = None
    try:
        agent = core._build_agent(config)
        core._log_mask_aware_summary(config.mask_aware)
        core._log_tcp_anchor_summary(config.tcp_anchor)

        core._arm_renderer = core._init_mask_renderer(config.mask_aware, agent)
        ensemble_buffer = core.EnsembleBuffer(mode=config.deploy.ensemble_mode)

        print("Ready for rollout. Press Enter to continue...")
        input()

        with torch.inference_mode():
            for t in range(int(config.deploy.max_steps)):
                colors, depths = agent.get_global_observation()
                latest_colors = np.asarray(colors, dtype=np.uint8).copy()
                latest_depths = np.asarray(depths).copy()
                latest_proprio, latest_joint = _read_save_proprio(agent, t)
                if latest_joint is None:
                    latest_joint = _read_save_joint_only(agent, t)

                mask01 = None
                if config.mask_aware.enabled:
                    mask01, mask_reason = core._infer_mask(
                        agent, depths.shape[:2], config.mask_aware, t
                    )
                    if mask01 is None:
                        core._warn(
                            f"step={t} mask unavailable, continue without mask: {mask_reason}"
                        )
                    elif config.mask_aware.debug_save_mask and (
                        t % config.mask_aware.debug_save_every == 0
                    ):
                        try:
                            core._save_mask_debug(colors, mask01, t, config.mask_aware)
                        except Exception as exc:
                            core._warn(
                                f"step={t} failed to save mask debug images: {exc}"
                            )
                latest_mask01 = None if mask01 is None else np.asarray(mask01).copy()

                if config.tcp_anchor.enabled and mask01 is not None:
                    mask01 = core._dilate_mask01(
                        mask01, config.tcp_anchor.mask_dilate_kernel
                    )

                tcp_camera = None
                if config.tcp_anchor.enabled:
                    try:
                        tcp_camera = core._get_tcp_camera(agent, projector)
                    except Exception as exc:
                        core._warn(
                            f"step={t} failed to get tcp camera pose, continue without tcp anchor: {exc}"
                        )
                latest_tcp_camera = (
                    None
                    if tcp_camera is None
                    else np.asarray(tcp_camera, dtype=np.float32).copy()
                )

                if tcp_camera is not None and config.tcp_anchor.enabled:
                    latest_anchor_points = core._build_tcp_anchor_points(
                        tcp_camera,
                        voxel_size=config.data.voxel_size,
                        radius_scale=config.tcp_anchor.anchor_radius_scale,
                    ).astype(np.float32)
                else:
                    latest_anchor_points = None

                if t % config.deploy.num_inference_steps == 0:
                    depths_for_cloud = depths
                    if (
                        config.mask_aware.enabled
                        and config.mask_aware.enable_3d_filter
                        and mask01 is not None
                    ):
                        depths_for_cloud = depths.copy()
                        depths_for_cloud[mask01 > 0.5] = 0

                    coords, points, cloud = core.create_input_with_anchor(
                        colors,
                        depths_for_cloud,
                        cam_intrinsics=agent.intrinsics,
                        config=config,
                        tcp_camera=tcp_camera,
                        depth_scale=agent.camera.depth_scale,
                        rescale_factor=1.0,
                    )

                    if points.shape[0] == 0:
                        core._warn(f"step={t} cloud is empty; skip inference")
                        _save_step_bundle(
                            capture_dirs,
                            t,
                            latest_colors,
                            latest_depths,
                            latest_mask01,
                            None,
                            latest_proprio,
                            latest_joint,
                            latest_tcp_camera,
                            latest_anchor_points,
                            10,
                        )
                        continue

                    image_coords = image_processor.get_image_coordinates(
                        depths, agent.intrinsics, agent.camera.depth_scale
                    )
                    colors_tensor, image_coords = image_processor.preprocess_images(
                        colors, image_coords
                    )

                    image_mask_weight = None
                    if (
                        config.mask_aware.enabled
                        and config.mask_aware.enable_2d_reweight
                        and mask01 is not None
                    ):
                        try:
                            image_mask_weight = core._build_image_mask_weight(
                                mask01, image_processor
                            )
                            if tcp_camera is not None and config.tcp_anchor.enabled:
                                image_mask_weight = core._restore_tcp_patch_weight(
                                    image_mask_weight,
                                    tcp_camera,
                                    agent.intrinsics,
                                    depths.shape[:2],
                                    image_processor,
                                    config.tcp_anchor,
                                )
                        except Exception as exc:
                            core._warn(
                                f"step={t} failed to build image_mask_weight, continue without it: {exc}"
                            )
                            image_mask_weight = None

                    import MinkowskiEngine as ME

                    coords_batch, feats_batch = core.create_batch(coords, points)
                    coords_batch = coords_batch.to(device)
                    feats_batch = feats_batch.to(device)
                    cloud_data = ME.SparseTensor(feats_batch, coords_batch)

                    colors_tensor = colors_tensor.unsqueeze(0).to(device)
                    image_coords = image_coords.unsqueeze(0).to(device)
                    if image_mask_weight is not None:
                        image_mask_weight = image_mask_weight.unsqueeze(0).to(device)

                    pred_raw_action = (
                        policy(
                            cloud_data,
                            colors_tensor,
                            image_coords,
                            image_mask_weight=image_mask_weight,
                            actions=None,
                        )
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )

                    action = core.process_state(
                        pred_raw_action, config, to_control=True
                    )

                    if getattr(config.deploy, "vis", False):
                        tcp_vis_list = []
                        for raw_tcp in action:
                            tcp_vis_list.append(
                                core.o3d.geometry.TriangleMesh.create_sphere(0.01).translate(
                                    raw_tcp[:3]
                                )
                            )
                        anchor_vis_list = []
                        if latest_anchor_points is not None:
                            for anchor_pt in np.asarray(
                                latest_anchor_points, dtype=np.float32
                            ):
                                anchor_vis = core.o3d.geometry.TriangleMesh.create_sphere(
                                    0.006
                                )
                                anchor_vis.paint_uniform_color([0.0, 0.2, 1.0])
                                anchor_vis.translate(anchor_pt[:3])
                                anchor_vis_list.append(anchor_vis)
                        core.o3d.visualization.draw_geometries(
                            [cloud, *tcp_vis_list, *anchor_vis_list]
                        )

                    action_tcp = projector.project_tcp_to_base_coord(
                        action[..., :9], rotation_rep="rotation_6d"
                    )
                    action = np.concatenate([action_tcp, action[..., 9:10]], axis=-1)
                    ensemble_buffer.add_action(action, t)

                step_action = ensemble_buffer.get_action()
                _save_step_bundle(
                    capture_dirs,
                    t,
                    latest_colors,
                    latest_depths,
                    latest_mask01,
                    step_action,
                    latest_proprio,
                    latest_joint,
                    latest_tcp_camera,
                    latest_anchor_points,
                    10,
                )

                if step_action is None:
                    continue

                print(f"Step {t}: executing action {step_action} ...")
                agent.action(step_action, rotation_rep="rotation_6d")
    finally:
        if agent is not None:
            try:
                agent.stop()
            except Exception as exc:
                core._warn(f"agent stop failed during save capture: {exc}")
        core._arm_renderer = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type", action="store", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument("--calib_rise2", action="store", type=str, required=True)
    parser.add_argument("--config", action="store", type=str, required=True)
    parser.add_argument(
        "--ckpt", action="store", type=str, required=False, default=None
    )
    parser.add_argument(
        "--host", action="store", type=str, required=False, default="127.0.0.1"
    )
    parser.add_argument(
        "--port", action="store", type=int, required=False, default=8000
    )
    parser.add_argument(
        "--save-root",
        action="store",
        type=str,
        required=False,
        default="deploy_capture_single_tcp_anchor",
    )
    evaluate(vars(parser.parse_args()))
