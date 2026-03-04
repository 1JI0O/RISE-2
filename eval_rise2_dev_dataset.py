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
from remote_eval import WebsocketClientPolicy
# from eval_agent import SingleArmAgent, DualArmAgent
from dataset.data_utils import resize_image, ImageProcessor
from dataset.projector import SingleArmProjector, DualArmProjector

import cv2

# test_color = "/data/haoxiang/realdata_rise2_ready/train/task_0014_user_0020_scene_0001_cfg_0001/cam_104122063550/color/1765000020748.png"
# test_depth = "/data/haoxiang/realdata_rise2_ready/train/task_0014_user_0020_scene_0001_cfg_0001/cam_104122063550/depth/1765000020748.png"


test_color = "/data/haoxiang/data/airexo2/task_0013/train/scene_0001/cam_105422061350/color/1737546126606.png"
test_depth = "/data/haoxiang/data/airexo2/task_0013/train/scene_0001/cam_105422061350/depth/1737546126606.png"

fake_intrinsics = np.array([
    [912.4466 ,   0.     , 633.4127 ],
    [  0.     , 911.4704 , 364.21265],
    [  0.     ,   0.     ,   1.     ]
])

fake_depth_scale = 1000.0

test_low_dim = "/data/haoxiang/data/airexo2/task_0013/train/scene_0001/lowdim/1737546126606.npy"
# lowdim npy 期望是 dict，包含 robot_left/right 和 gripper_left/right。


default_args = edict({
    "type": "local",
    "calib_rise2": "calib_rise2/",
    "calib_airexo": "calib_airexo/",
    "config": "config/dual_teleop_dino.yaml",
    "ckpt": "logs/collect_toys",
    "host": "127.0.0.1",
    "port": 8000
})


def _build_mask_aware_cfg(config):
    # 统一推理期 mask-aware 配置并补齐默认值
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
    }
    raw_cfg = getattr(config, "mask_aware", {})
    raw_cfg = dict(raw_cfg) if raw_cfg is not None else {}

    merged_cfg = deepcopy(default_cfg)
    for key in default_cfg:
        if key in raw_cfg and raw_cfg[key] is not None:
            merged_cfg[key] = raw_cfg[key]

    valid_none_policy = {"no_mask_fallback", "fail_fast"}
    valid_empty_cloud_policy = {"warn_and_skip_filter", "fail_fast"}

    try:
        merged_cfg["enabled"] = bool(merged_cfg["enabled"])
        merged_cfg["enable_3d_filter"] = bool(merged_cfg["enable_3d_filter"])
        merged_cfg["enable_2d_reweight"] = bool(merged_cfg["enable_2d_reweight"])
        merged_cfg["mask_threshold"] = float(merged_cfg["mask_threshold"])
        merged_cfg["mask_white_is_untrusted"] = bool(merged_cfg["mask_white_is_untrusted"])
        merged_cfg["r_min"] = float(merged_cfg["r_min"])
        merged_cfg["interp_eps"] = float(merged_cfg["interp_eps"])
        merged_cfg["interp_tiny"] = float(merged_cfg["interp_tiny"])
        merged_cfg["infer_allow_none"] = bool(merged_cfg["infer_allow_none"])
        merged_cfg["infer_none_policy"] = str(merged_cfg["infer_none_policy"])
        merged_cfg["empty_cloud_policy"] = str(merged_cfg["empty_cloud_policy"])
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
    # 启动时打印一次配置摘要用于排查
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
    # 异常回退统一日志输出
    print(f"[mask-aware] step={step} reason={reason} action={action}")


# ── 模块级 renderer 引用，由 evaluate() 在启动时初始化 ──
_arm_renderer = None


def get_arm_joints(meta=None):
    """
    预留接口：返回当前机械臂关节角。

    Returns
    -------
    (left_joint, right_joint) : tuple of np.ndarray, shape (num_robot_joints+1,)
        前 num_robot_joints 个元素为关节角（弧度），最后一个元素为夹爪宽度（米）。
    None
        无法获取时返回 None，上层会触发 no_mask_fallback。

    实际部署时在此对接机械臂 SDK（例如读取 Flexiv 的 joint_pos + gripper_width）。
    """
    lowdim_path = test_low_dim
    if meta is not None and hasattr(meta, "get"):
        lowdim_path = meta.get("lowdim_path", lowdim_path)

    try:
        lowdim_raw = np.load(lowdim_path, allow_pickle=True)
        lowdim = lowdim_raw.item() if isinstance(lowdim_raw, np.ndarray) else lowdim_raw

        left_robot = np.asarray(lowdim["robot_left"], dtype=np.float32).reshape(-1)
        right_robot = np.asarray(lowdim["robot_right"], dtype=np.float32).reshape(-1)
        left_gripper = np.asarray(lowdim["gripper_left"], dtype=np.float32).reshape(-1)
        right_gripper = np.asarray(lowdim["gripper_right"], dtype=np.float32).reshape(-1)

        if left_gripper.size == 0 or right_gripper.size == 0:
            return None

        left_joint = np.concatenate([left_robot, np.array([left_gripper[0]], dtype=np.float32)])
        right_joint = np.concatenate([right_robot, np.array([right_gripper[0]], dtype=np.float32)])
        return left_joint, right_joint
    except Exception as exc:
        print(f"[mask-aware] failed to load arm joints from lowdim npy: {exc}")
        return None


def infer_mask(color, depth, proprio, meta):
    """
    通过 URDF 渲染当前机械臂姿态，返回像素级 mask。

    Returns
    -------
    mask : np.ndarray (H, W) uint8  —— 255 = 机械臂像素，0 = 背景
    None  —— renderer 未初始化或关节角不可用时
    """
    if _arm_renderer is None:
        return None
    joints = get_arm_joints(meta)
    if joints is None:
        return None
    left_joint, right_joint = joints
    _arm_renderer.update_joints(left_joint, right_joint)
    return _arm_renderer.render_mask()


def _to_numpy_mask(mask):
    # 将输入 mask 统一转换为 numpy
    if isinstance(mask, torch.Tensor):
        return mask.detach().cpu().numpy()
    return np.asarray(mask)


def _normalize_mask01(mask, depth_shape, mask_cfg):
    # 将 mask 规范到 depth 尺寸并转换为 float32 的 0/1
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
            "[mask-aware] mask size mismatch, resize with nearest: "
            f"mask={mask_np.shape}, depth=({target_h}, {target_w})"
        )
        mask_np = cv2.resize(
            mask_np.astype(np.float32),
            (target_w, target_h),
            interpolation = cv2.INTER_NEAREST,
        )

    threshold = float(mask_cfg.mask_threshold)
    if bool(mask_cfg.mask_white_is_untrusted):
        mask01 = (mask_np > threshold).astype(np.float32)
    else:
        mask01 = (mask_np <= threshold).astype(np.float32)

    return mask01


def _safe_infer_mask(color, depth, proprio, meta, mask_cfg):
    # 执行 infer_mask 并把异常统一转换为回退信号
    try:
        raw_mask = infer_mask(color, depth, proprio, meta)
    except Exception:
        return None, "infer_exception"

    if raw_mask is None:
        return None, "infer_none"

    try:
        mask01 = _normalize_mask01(raw_mask, depth.shape[:2], mask_cfg)
    except Exception:
        return None, "mask_invalid"

    return mask01, None


def _build_image_mask_weight(mask01, image_processor):
    # 根据二维 mask 构建图像 patch 级可信度权重
    try:
        mask_tensor = torch.from_numpy(mask01[np.newaxis].astype(np.float32))
        mask_tensor = resize_image(
            mask_tensor,
            image_processor.img_size,
            interpolation = T.InterpolationMode.NEAREST,
        )
        mask_ratio = image_processor.image_coord_pooling(mask_tensor)
        image_mask_weight = (1.0 - mask_ratio).clamp(0.0, 1.0).to(torch.float32)
    except Exception:
        return None

    return image_mask_weight


def load_test_obs(color_path, depth_path):
    # 1. 加载彩色图并转为 RGB (OpenCV 默认读入是 BGR)
    color_image = cv2.imread(color_path)
    if color_image is None:
        raise ValueError(f"无法加载图片: {color_path}")
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB).astype(np.uint8)

    # 2. 加载深度图
    # 注意：必须使用 cv2.IMREAD_UNCHANGED 才能保留 16bit 深度信息
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError(f"无法加载深度图: {depth_path}")
    
    # 确保是 uint16。如果你的 PNG 是 8bit 的，需要根据量化比例转回 uint16 (通常单位是毫米)
    depth_image = depth_image.astype(np.uint16)

    return color_image, depth_image


def create_point_cloud(colors, depths, intrinsics, config, depth_scale = 1000.0, rescale_factor = 1):
    """
    color, depth => point cloud
    """
    if rescale_factor != 1:
        H, W = depths.shape
        h, w = int(H * rescale_factor), int(W * rescale_factor)
        colors = colors.transpose([2, 0, 1]).astype(np.float32)
        colors = torch.from_numpy(colors)
        colors = np.ascontiguousarray(resize_image(colors, [h, w]).numpy().transpose([1, 2, 0]))
        depths = depths.astype(np.float32)
        depths = torch.from_numpy(depths[np.newaxis])
        depths = resize_image(depths, [h,w], interpolation = T.InterpolationMode.NEAREST)[0]
        depths = depths.numpy()

    # generate point cloud
    h, w = depths.shape
    fx, fy = intrinsics[0, 0] * rescale_factor, intrinsics[1, 1] * rescale_factor
    cx, cy = intrinsics[0, 2] * rescale_factor, intrinsics[1, 2] * rescale_factor
    colors = o3d.geometry.Image(colors.astype(np.uint8))
    depths = o3d.geometry.Image(depths.astype(np.float32))
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors, depths, depth_scale, convert_rgb_to_intensity = False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
    # crop point cloud
    bbox3d = o3d.geometry.AxisAlignedBoundingBox(config.deploy.workspace.min, config.deploy.workspace.max)
    cloud = cloud.crop(bbox3d)
    # downsample
    cloud = cloud.voxel_down_sample(config.data.voxel_size)
    return cloud


def create_input(colors, depths, cam_intrinsics, config, depth_scale = 1000.0, rescale_factor = 1):
    """
    colors, depths => coords, points
    """
    # create point cloud
    cloud = create_point_cloud(
        colors, 
        depths, 
        cam_intrinsics, 
        config,
        depth_scale = depth_scale,
        rescale_factor = rescale_factor,
    )

    # convert to sparse tensor
    points = np.asarray(cloud.points)
    coords = np.ascontiguousarray(points / config.data.voxel_size, dtype = np.int32)

    return coords, points, cloud
    

def create_batch(coords, points):
    """
    coords, points => batch coords, batch feats
    """
    import MinkowskiEngine as ME
    input_coords = [coords]
    input_feats = [points.astype(np.float32)]
    coords_batch, feats_batch = ME.utils.sparse_collate(input_coords, input_feats)
    return coords_batch, feats_batch


def process_state(state, config, to_control = True):
    if config.robot_type == "single":
        if to_control:
            state[..., 0: 3] = (state[..., 0: 3] + 1) / 2.0 * (config.data.normalization.trans_max - config.data.normalization.trans_min) + config.data.normalization.trans_min
            state[..., 9] = (state[..., 9] + 1) / 2.0 * config.data.normalization.max_gripper_width
        else:
            state[..., 0: 3] = (state[..., 0: 3] - config.data.normalization.trans_min) / (config.data.normalization.trans_max - config.data.normalization.trans_min) * 2.0 - 1
            state[..., 9] = state[..., 9] / config.data.normalization.max_gripper_width * 2.0 - 1
    else:
        if to_control:
            state[..., 0: 3] = (state[..., 0: 3] + 1) / 2.0 * (config.data.normalization.trans_max - config.data.normalization.trans_min) + config.data.normalization.trans_min
            state[..., 10: 13] = (state[..., 10: 13] + 1) / 2.0 * (config.data.normalization.trans_max - config.data.normalization.trans_min) + config.data.normalization.trans_min
            state[..., 9] = (state[..., 9] + 1) / 2.0 * config.data.normalization.max_gripper_width
            state[..., 19] = (state[..., 19] + 1) / 2.0 * config.data.normalization.max_gripper_width
        else:
            state[..., 0: 3] = (state[..., 0: 3] - config.data.normalization.trans_min) / (config.data.normalization.trans_max - config.data.normalization.trans_min) * 2.0 - 1
            state[..., 10: 13] = (state[..., 10: 13] - config.data.normalization.trans_min) / (config.data.normalization.trans_max - config.data.normalization.trans_min) * 2.0 - 1
            state[..., 9] = state[..., 9] / config.data.normalization.max_gripper_width * 2.0 - 1
            state[..., 19] = state[..., 19] / config.data.normalization.max_gripper_width * 2.0 - 1

    return state



def evaluate(args_override):
    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # load config
    with open(args.config, "r") as f:
        config = edict(yaml.load(f, Loader = yaml.FullLoader))
    config.data.normalization.trans_min = np.asarray(config.data.normalization.trans_min)
    config.data.normalization.trans_max = np.asarray(config.data.normalization.trans_max)
    config.mask_aware = _build_mask_aware_cfg(config)

    # set seed
    set_seed(config.deploy.seed)

    # load policy for local inference
    if args.type == "local":
        from policy import RISE2
        # set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load policy
        print("Loading policy ...")
        policy = RISE2(
            num_action = config.data.num_action,
            obs_feature_dim = config.model.obs_feature_dim,
            cloud_enc_dim = config.model.cloud_enc_dim,
            image_enc_dim = config.model.image_enc_dim,
            action_dim = 10 if config.robot_type == "single" else 20,
            hidden_dim = config.model.hidden_dim,
            nheads = config.model.nheads,
            num_attn_layers = config.model.num_attn_layers,
            dim_feedforward = config.model.dim_feedforward,
            dropout = config.model.dropout,
            image_enc = config.model.image_enc,
            interp_fn_mode = config.model.interp_fn_mode,
            image_enc_finetune = config.model.image_enc_finetune,
            image_enc_dtype = config.model.image_enc_dtype
        ).to(device)

        # load checkpoint
        assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
        policy.load_state_dict(torch.load(args.ckpt, map_location = device), strict = False)
        print("Checkpoint {} loaded.".format(args.ckpt))

        # set evaluation
        policy.eval()

    else:
        # connect to remote inference service
        print("Connecting to remote server ...")
        policy = WebsocketClientPolicy(host = args.host, port = args.port)

    # projector
    Projector = SingleArmProjector if config.robot_type == "single" else DualArmProjector
    projector = Projector(args.calib_rise2, config.deploy.agent.camera_serial)

    # image processor
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
    
    image_processor = ImageProcessor(
        img_size = img_size,
        img_coord_size = img_coord_size,
        voxel_size = config.data.voxel_size,
        img_mean = config.data.normalization.img_mean,
        img_std = config.data.normalization.img_std
    )

    # evaluation
    # Agent = SingleArmAgent if config.robot_type == "single" else DualArmAgent
    # agent = Agent(**config.deploy.agent)

    # ensemble buffer
    ensemble_buffer = EnsembleBuffer(mode = config.deploy.ensemble_mode)

    # 输出 mask-aware 配置摘要
    _log_mask_aware_summary(config.mask_aware)

    # 初始化 URDF renderer（仅本地 mask-aware 模式）
    global _arm_renderer
    _arm_renderer = None
    if config.mask_aware.enabled and args.type == "local":
        try:
            from mask.renderer import ArmOnlyRobotRenderer
            from airexo.airexo.calibration.calib_info import CalibrationInfo
            # 标定与 notebook "直接用renderer" cell 完全一致
            cam_serial  = config.deploy.agent.camera_serial
            calib_ts    = int(os.path.splitext(os.path.basename(args.calib_airexo))[0])
            calib_info  = CalibrationInfo(os.path.dirname(args.calib_airexo), calib_ts)
            _urdf = config.mask_aware.urdf or os.path.join(
                "airexo", "airexo", "urdf_models", "robot", "robot_inhand.urdf"
            )
            _left_cfg  = edict(yaml.safe_load(open(os.path.join("airexo", "airexo", "configs", "joint", "left",  "robot.yaml"))))
            _right_cfg = edict(yaml.safe_load(open(os.path.join("airexo", "airexo", "configs", "joint", "right", "robot.yaml"))))
            _arm_renderer = ArmOnlyRobotRenderer(  # gripper link 已在子类中过滤
                left_joint_cfgs  = _left_cfg,
                right_joint_cfgs = _right_cfg,
                cam_to_base      = calib_info.get_camera_to_base(cam_serial),
                intrinsic        = calib_info.get_intrinsic(cam_serial),
                urdf_file        = _urdf,
                width=1280, height=720, near_plane=0.01, far_plane=100.0,
            )
            print("[mask-aware] renderer initialized ({})".format(_urdf))
        except Exception as _e:
            print(f"[mask-aware] renderer init failed, mask disabled: {_e}")
            _arm_renderer = None

    # 记录异常回退统计
    mask_stats = {
        "infer_none": 0,
        "infer_exception": 0,
        "mask_invalid": 0,
        "empty_cloud_skip": 0,
        "reweight_fallback": 0,
        "points_nonfinite": 0,
        "weight_nonfinite": 0,
    }

    # evaluation rollout
    print("Ready for rollout. Press Enter to continue...")
    input()
    
    with torch.inference_mode():
        for t in range(config.deploy.max_steps):
            if t % config.deploy.num_inference_steps == 0:
                # pre-process inputs
                # colors, depths = agent.get_global_observation()

                colors, depths = load_test_obs(test_color, test_depth)

                # 本地推理启用 mask-aware 分支
                mask_enabled = bool(config.mask_aware.enabled and args.type == "local")
                mask01, mask_reason = None, None
                if mask_enabled:
                    mask01, mask_reason = _safe_infer_mask(
                        color = colors,
                        depth = depths,
                        proprio = None,
                        meta = {"step": t, "mode": args.type},
                        mask_cfg = config.mask_aware,
                    )
                    if mask01 is None:
                        reason = mask_reason or "unknown_infer_failure"
                        if reason in mask_stats:
                            mask_stats[reason] += 1
                        if config.mask_aware.infer_none_policy == "fail_fast":
                            _log_mask_fallback(t, reason, "fail_fast")
                            raise RuntimeError(f"mask unavailable with fail_fast, reason={reason}")
                        _log_mask_fallback(t, reason, "no_mask_fallback")

                # 根据 mask 生成点云深度输入
                depths_for_cloud = depths
                if mask_enabled and config.mask_aware.enable_3d_filter and mask01 is not None:
                    depths_for_cloud = depths.copy()
                    depths_for_cloud[mask01 > 0.5] = 0

                # create cloud inputs
                create_input_kwargs = dict(
                    # cam_intrinsics = agent.intrinsics,
                    cam_intrinsics = fake_intrinsics,
                    config = config,
                    # depth_scale = agent.camera.depth_scale,
                    depth_scale = fake_depth_scale,
                    rescale_factor = 1.0,
                )
                coords, points, cloud = create_input(
                    colors,
                    depths_for_cloud,
                    **create_input_kwargs,
                )

                # 点云出现非法数值时优先回退到原始深度重建
                if points.size > 0 and (not np.isfinite(points).all()):
                    if config.mask_aware.empty_cloud_policy == "fail_fast":
                        _log_mask_fallback(t, "points_nonfinite", "fail_fast")
                        raise RuntimeError("non-finite points after cloud build")
                    mask_stats["points_nonfinite"] += 1
                    _log_mask_fallback(t, "points_nonfinite", "rebuild_from_original_depth")
                    coords, points, cloud = create_input(
                        colors,
                        depths,
                        **create_input_kwargs,
                    )

                # 过滤后空点云回退
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
                        colors,
                        depths,
                        **create_input_kwargs,
                    )

                # create image inputs
                # image_coords = image_processor.get_image_coordinates(depths, agent.intrinsics, agent.camera.depth_scale)
                image_coords = image_processor.get_image_coordinates(depths, fake_intrinsics, fake_depth_scale)
                colors, image_coords = image_processor.preprocess_images(colors, image_coords)

                # 根据 mask 构建 patch 级可信度权重
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

                # predict action
                if args.type == "local":
                    import MinkowskiEngine as ME
                    coords_batch, feats_batch = create_batch(coords, points)
                    coords_batch, feats_batch = coords_batch.to(device), feats_batch.to(device)
                    cloud_data = ME.SparseTensor(feats_batch, coords_batch)

                    colors = colors.unsqueeze(0).to(device)
                    image_coords = image_coords.unsqueeze(0).to(device)
                    if image_mask_weight is not None:
                        image_mask_weight = image_mask_weight.unsqueeze(0).to(device)

                    # predict
                    pred_raw_action = policy(
                        cloud_data,
                        colors,
                        image_coords,
                        image_mask_weight = image_mask_weight,
                        actions = None,
                    ).squeeze(0).cpu().numpy()

                else:
                    obs_dict = {
                        "coords": coords,
                        "points": points,
                        "colors": colors.numpy(),
                        "image_coords": image_coords.numpy()
                    }

                    pred_raw_action = deepcopy(policy.infer(obs_dict)["actions"])

                # unnormalize predicted actions
                action = process_state(pred_raw_action, config, to_control = True)

                # visualization
                if config.deploy.vis:
                    vis_save_dir = getattr(config.deploy, "vis_save_dir", ".")
                    if vis_save_dir is None or len(str(vis_save_dir).strip()) == 0:
                        vis_save_dir = "."
                    os.makedirs(vis_save_dir, exist_ok = True)

                    vis_save_prefix = getattr(config.deploy, "vis_save_prefix", "vis_debug")
                    if vis_save_prefix is None or len(str(vis_save_prefix).strip()) == 0:
                        vis_save_prefix = "vis_debug"
                    vis_save_prefix = str(vis_save_prefix)

                    combined_cloud = o3d.geometry.PointCloud()
                    combined_cloud += cloud

                    tcp_points = []
                    for raw_tcp in action:
                        tcp_points.append(raw_tcp[:3])
                        if config.robot_type == "dual":
                            tcp_points.append(raw_tcp[10:13])

                    if len(tcp_points) > 0:
                        tcp_points = np.asarray(tcp_points, dtype = np.float64).reshape(-1, 3)
                        tcp_cloud = o3d.geometry.PointCloud()
                        tcp_cloud.points = o3d.utility.Vector3dVector(tcp_points)
                        tcp_cloud.paint_uniform_color([1.0, 1.0, 0.0])
                        combined_cloud += tcp_cloud

                    ply_path = os.path.join(vis_save_dir, "{}_step_{:06d}.ply".format(vis_save_prefix, t))
                    o3d.io.write_point_cloud(ply_path, combined_cloud)
                    print("[vis] saved ply: {}".format(ply_path))
                    input("press enter")
                
                # project action to base coordinate
                if config.robot_type == "single":
                    action_tcp = projector.project_tcp_to_base_coord(action[..., :9], rotation_rep = "rotation_6d")
                    action = np.concatenate([action_tcp, action[..., 9:10]], axis = -1)
                else:
                    action_left_tcp = projector.project_tcp_to_base_coord(action[..., :9], "left", rotation_rep = "rotation_6d")
                    action_right_tcp = projector.project_tcp_to_base_coord(action[..., 10:19], "right", rotation_rep = "rotation_6d")
                    action = np.concatenate([action_left_tcp, action[..., 9:10], action_right_tcp, action[..., 19:20]], axis = -1)
                
                # add to ensemble buffer
                ensemble_buffer.add_action(action, t)
            
            # get step action from ensemble buffer
            step_action = ensemble_buffer.get_action()
            # 这个是 config.deploy.num_inference_steps 这么多次循环完成后
            # 根据 ensemble_buffer 存的一串动作加权平均得到的
            
            if step_action is None:   # no action in the buffer => no movement.
                continue
            
            # agent.action(step_action, rotation_rep = "rotation_6d")
            print(f"execute {step_action}")
            input("enter")

    print(
        "[mask-aware] summary infer_none={} infer_exception={} mask_invalid={} "
        "empty_cloud_skip={} reweight_fallback={} points_nonfinite={} weight_nonfinite={}".format(
            mask_stats["infer_none"],
            mask_stats["infer_exception"],
            mask_stats["mask_invalid"],
            mask_stats["empty_cloud_skip"],
            mask_stats["reweight_fallback"],
            mask_stats["points_nonfinite"],
            mask_stats["weight_nonfinite"],
        )
    )

    agent.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', action = 'store', type = str, help = 'evaluation type, choices: ["local", "remote"].', required = True, choices = ["local", "remote"])
    parser.add_argument('--calib_airexo', action = 'store', type = str, help = 'airexo calibration path', required = True)
    parser.add_argument('--calib_rise2', action = 'store', type = str, help = 'rise2 calibration path', required = True)
    parser.add_argument('--config', action = 'store', type = str, help = 'data and model config during training and deployment', required = True)
    parser.add_argument('--ckpt', action = 'store', type = str, help = 'checkpoint path', required = False, default = None)
    parser.add_argument('--host', action = 'store', type = str, help = 'server host address', required = False, default = "127.0.0.1")
    parser.add_argument('--port', action = 'store', type = int, help = 'server port', required = False, default = 8000)

    evaluate(vars(parser.parse_args()))