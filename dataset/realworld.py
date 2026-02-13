import os
import json
import torch
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import torchvision.transforms as T
import collections.abc as container_abcs

from PIL import Image
from torch.utils.data import Dataset

from dataset.projector import SingleArmProjector, DualArmProjector
from dataset.data_utils import ImageProcessor, TrajLoader, load_action, resize_image, vis_data
from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform


TO_TENSOR_KEYS = ['cloud_coords', 'cloud_feats', 'image_coords', 'image_feats', 'action', 'action_normalized', 'mask01', 'image_mask_weight']

# need to be modified

class RealWorldDataset(Dataset):
    """
    Real-world dataset for single/dual-arm.
    """
    def __init__(self, path, config, split = "train"):
        self.path = path
        self.split = split
        self.data_path = os.path.join(path, split)
        self.calib_path = os.path.join(path, "calib")
        self._parse_config(config)
        self._validate_mask_aware_config()
        assert not (self.robot_type == "single" and self.data_type == "wild"),\
            "Single arm does not support in-the-wild data currently."

        self.all_demos = sorted(os.listdir(self.data_path))
        self.all_demos = [x for x in self.all_demos if "scene_" in x]
        self.num_demos = len(self.all_demos)

        self.data_paths = []
        self.cam_ids = []
        self.calib_timestamps = []
        self.obs_frame_ids = []
        self.action_frame_ids = []
        self.mask_paths = []
        self.projectors = {}
        # 缓存每个场景相机的 color 到 mask 映射
        self._mask_lookup_cache = {}

        # create image processor
        self.image_processor = ImageProcessor(
            self.img_size,
            self.img_coord_size,
            self.voxel_size,
            self.img_mean,
            self.img_std
        )

        # create color jitter
        if self.aug_color:
            jitter = T.ColorJitter(
                brightness = self.aug_color_params[0],
                contrast = self.aug_color_params[1],
                saturation = self.aug_color_params[2],
                hue = self.aug_color_params[3]
            )
            self.jitter = T.RandomApply([jitter], p = self.aug_color_prob)

        # load trajectories
        self.traj_loader = TrajLoader(
            robot_type = config.robot_type,
            sample_traj = config.data.sample_traj.enable,
            trans_delta = config.data.sample_traj.trans_delta,
            rot_delta = config.data.sample_traj.rot_delta,
            width_delta = config.data.sample_traj.width_delta
        )

        for i in range(self.num_demos):
            demo_path = os.path.join(self.data_path, self.all_demos[i])
            # get frame ids
            frame_ids, _ = self.traj_loader.load_traj(demo_path)
            # get projectors
            calib_timestamp, cam_ids = self.register_projector(demo_path, self.calib_path)
            for cam_id in cam_ids:
                # path
                cam_path = os.path.join(demo_path, "cam_{}".format(cam_id))
                if not os.path.exists(cam_path):
                    continue
                # 统一解析当前相机的图像目录
                color_dir, depth_dir = self._resolve_image_dirs(demo_path, cam_id)
                # 目录不完整时跳过该相机
                if not os.path.exists(color_dir) or not os.path.exists(depth_dir):
                    continue

                # get samples according to num_action
                obs_frame_ids_list = frame_ids[:-1]
                action_frame_ids_list = []

                for cur_idx in range(len(frame_ids) - 1):
                    action_pad_after = max(0, self.num_action - (len(frame_ids) - 1 - cur_idx))
                    frame_end = min(len(frame_ids), cur_idx + self.num_action + 1)
                    action_frame_ids = frame_ids[cur_idx + 1: frame_end] + frame_ids[-1:] * action_pad_after
                    action_frame_ids_list.append(action_frame_ids)

                self.data_paths += [demo_path] * len(obs_frame_ids_list)
                self.cam_ids += [cam_id] * len(obs_frame_ids_list)
                self.calib_timestamps += [calib_timestamp] * len(obs_frame_ids_list)
                self.obs_frame_ids += obs_frame_ids_list
                self.action_frame_ids += action_frame_ids_list

                if self.mask_aware_enabled:
                    # 每个 scene 和 cam 只构建一次对齐映射
                    cache_key = (demo_path, cam_id)
                    if cache_key not in self._mask_lookup_cache:
                        self._mask_lookup_cache[cache_key] = self._build_mask_lookup(demo_path, cam_id, color_dir)
                    mask_lookup = self._mask_lookup_cache[cache_key]
                    # 为每个 observation 绑定对应 mask 路径
                    for frame_id in obs_frame_ids_list:
                        mask_path = mask_lookup.get(frame_id, mask_lookup.get(str(frame_id), None))
                        if mask_path is None:
                            raise KeyError(
                                "Missing aligned mask for scene={} cam={} frame={}".format(
                                    os.path.basename(demo_path), cam_id, frame_id
                                )
                            )
                        self.mask_paths.append(mask_path)

        if self.mask_aware_enabled and len(self.mask_paths) != len(self.data_paths):
            raise RuntimeError(
                "Mask path count mismatch: {} vs {}".format(len(self.mask_paths), len(self.data_paths))
            )

        self.data_paths = self.data_paths * self.repeat_dataset
        self.cam_ids = self.cam_ids * self.repeat_dataset
        self.calib_timestamps = self.calib_timestamps * self.repeat_dataset
        self.obs_frame_ids = self.obs_frame_ids * self.repeat_dataset
        self.action_frame_ids = self.action_frame_ids * self.repeat_dataset
        if self.mask_aware_enabled:
            self.mask_paths = self.mask_paths * self.repeat_dataset

    def __len__(self):
        return len(self.obs_frame_ids)

    def _frame_sort_key(self, file_name):
        # 帧名优先按数值排序，无法转数值时按字符串排序
        stem = os.path.splitext(file_name)[0]
        try:
            return (0, int(stem))
        except ValueError:
            try:
                return (0, float(stem))
            except ValueError:
                return (1, stem)

    def _list_sorted_pngs(self, folder):
        # 统一收集并排序 png 帧
        pngs = [x for x in os.listdir(folder) if x.lower().endswith(".png")]
        pngs = sorted(pngs, key = self._frame_sort_key)
        return pngs

    def _resolve_image_dirs(self, demo_path, cam_id):
        # 根据数据类型返回 color 和 depth 目录
        cam_dir = os.path.join(demo_path, "cam_{}".format(cam_id))
        if self.data_type == "teleop":
            color_dir = os.path.join(cam_dir, "color")
            depth_dir = os.path.join(cam_dir, "depth")
        else:
            color_dir = os.path.join(cam_dir, "color_controlnet")
            depth_dir = os.path.join(cam_dir, "depth_inpainting")
        return color_dir, depth_dir

    def _resolve_mask_dir(self, demo_path, cam_id):
        # 按候选路径查找存在且非空的 mask 目录
        scene_name = os.path.basename(demo_path)
        cam_name = "cam_{}".format(cam_id)
        candidate_dirs = [
            os.path.join(self.mask_root, scene_name, cam_name, "mask"),
            os.path.join(self.mask_root, scene_name, cam_name),
            os.path.join(self.mask_root, scene_name, "mask", cam_name),
            os.path.join(self.mask_root, scene_name, "mask"),
            os.path.join(self.mask_root, scene_name),
            os.path.join(demo_path, cam_name, "mask"),
        ]

        for mask_dir in candidate_dirs:
            if not os.path.isdir(mask_dir):
                continue
            mask_files = self._list_sorted_pngs(mask_dir)
            if len(mask_files) > 0:
                return mask_dir, mask_files

        raise FileNotFoundError(
            "Unable to resolve mask directory for scene={} cam={} under mask_root={}".format(
                scene_name, cam_id, self.mask_root
            )
        )

    def _build_mask_lookup(self, demo_path, cam_id, color_dir):
        # 按 sorted_index 规则构建 color 到 mask 的映射
        color_files = self._list_sorted_pngs(color_dir)
        if len(color_files) == 0:
            raise RuntimeError("No color frames found in {}".format(color_dir))

        mask_dir, mask_files = self._resolve_mask_dir(demo_path, cam_id)
        if len(color_files) != len(mask_files):
            raise ValueError(
                "Color/mask length mismatch for scene={} cam={}: {} vs {}".format(
                    os.path.basename(demo_path), cam_id, len(color_files), len(mask_files)
                )
            )

        mask_lookup = {}
        for color_name, mask_name in zip(color_files, mask_files):
            color_stem = os.path.splitext(color_name)[0]
            mask_path = os.path.join(mask_dir, mask_name)
            mask_lookup[color_stem] = mask_path
            try:
                mask_lookup[int(color_stem)] = mask_path
            except ValueError:
                pass

        if self.mask_log_alignment_preview and len(color_files) > 0:
            print(
                "[mask-align] scene={} cam={} n={} first={}=>{} last={}=>{}".format(
                    os.path.basename(demo_path),
                    cam_id,
                    len(color_files),
                    color_files[0],
                    mask_files[0],
                    color_files[-1],
                    mask_files[-1],
                )
            )

        return mask_lookup

    def _validate_mask_aware_config(self):
        # mask-aware 关闭时跳过额外校验
        if not self.mask_aware_enabled:
            return

        # 当前仅支持排序后一一对应
        if self.mask_align_mode != "sorted_index":
            raise ValueError(f"Unsupported mask_align_mode: {self.mask_align_mode}")

        # 方案A和方案B都必须可用
        if not self.mask_enable_2d_reweight or not self.mask_enable_3d_filter:
            raise ValueError("mask_aware requires both enable_2d_reweight and enable_3d_filter")

        # 开启后必须提供mask根目录
        if self.mask_root is None:
            raise ValueError("mask_root is required when mask_aware.enabled is true")
        if not os.path.isdir(self.mask_root):
            raise FileNotFoundError("mask_root does not exist: {}".format(self.mask_root))

        # 数值参数需要合法
        if self.mask_r_min < 0:
            raise ValueError(f"mask r_min must be >= 0, got {self.mask_r_min}")
        if self.mask_interp_eps <= 0:
            raise ValueError(f"mask interp_eps must be > 0, got {self.mask_interp_eps}")
        if self.mask_interp_tiny <= 0:
            raise ValueError(f"mask interp_tiny must be > 0, got {self.mask_interp_tiny}")

        # 空点云处理策略必须在白名单内
        valid_empty_policies = {"warn_and_skip", "fail_fast", "fallback_disable_3d_filter"}
        if self.mask_empty_cloud_policy not in valid_empty_policies:
            raise ValueError(f"Unsupported empty_cloud_policy: {self.mask_empty_cloud_policy}")

    def _parse_config(self, config):
        self.robot_type = config.robot_type
        self.data_type = config.data.type 
        self.num_action = config.data.num_action 
        self.vis = config.data.vis 
        self.voxel_size = config.data.voxel_size
        self.translation_min = np.asarray(config.data.normalization.trans_min)
        self.translation_max = np.asarray(config.data.normalization.trans_max)
        self.max_gripper_width = config.data.normalization.max_gripper_width
        self.img_mean = config.data.normalization.img_mean
        self.img_std = config.data.normalization.img_std
        self.aug_color_params = np.array(config.train.augmentation.aug_color_params)
        self.aug_color_prob = config.train.augmentation.aug_color_prob
        self.aug_point = config.train.augmentation.point 
        self.aug_color = config.train.augmentation.color 
        self.aug_trans_min = np.asarray(config.train.augmentation.aug_trans_min)
        self.aug_trans_max = np.asarray(config.train.augmentation.aug_trans_max)
        self.aug_rot_min = np.asarray(config.train.augmentation.aug_rot_min)
        self.aug_rot_max = np.asarray(config.train.augmentation.aug_rot_max)
        self.workspace_min = np.asarray(config.train.workspace.min)
        self.workspace_max = np.asarray(config.train.workspace.max)
        image_enc = config.model.image_enc
        if image_enc == "resnet18":
            self.img_size = config.data.aligner.img_size_resnet
            self.img_coord_size = config.data.aligner.img_coord_size_resnet
        elif image_enc.startswith("dinov2"):
            self.img_size = config.data.aligner.img_size_dinov2
            self.img_coord_size = config.data.aligner.img_coord_size_dinov2
        elif image_enc.startswith("dinov3"):
            self.img_size = config.data.aligner.img_size_dinov3
            self.img_coord_size = config.data.aligner.img_coord_size_dinov3
        else:
            raise ValueError(f"Unknown image encoder: {image_enc}")
        self.repeat_dataset = config.data.repeat_dataset

        # 统一读取mask-aware配置
        mask_cfg = getattr(config, "mask_aware", None)
        self.mask_aware_enabled = bool(getattr(mask_cfg, "enabled", False))
        mask_root = getattr(mask_cfg, "mask_root", None)
        if mask_root in [None, "", "null"]:
            self.mask_root = None
        else:
            self.mask_root = str(mask_root)
        self.mask_align_mode = str(getattr(mask_cfg, "align_mode", "sorted_index"))
        self.mask_enable_2d_reweight = bool(getattr(mask_cfg, "enable_2d_reweight", True))
        self.mask_enable_3d_filter = bool(getattr(mask_cfg, "enable_3d_filter", True))
        self.mask_threshold = int(getattr(mask_cfg, "mask_threshold", 0))
        self.mask_white_is_untrusted = bool(getattr(mask_cfg, "mask_white_is_untrusted", True))
        self.mask_r_min = float(getattr(mask_cfg, "r_min", 1e-3))
        self.mask_interp_eps = float(getattr(mask_cfg, "interp_eps", 1e-6))
        self.mask_interp_tiny = float(getattr(mask_cfg, "interp_tiny", 1e-6))
        self.mask_empty_cloud_policy = str(getattr(mask_cfg, "empty_cloud_policy", "warn_and_skip"))
        self.mask_log_alignment_preview = bool(getattr(mask_cfg, "log_alignment_preview", True))
    
    def register_projector(self, demo_path, calib_path):
        # get calib_timestamp
        with open(os.path.join(demo_path, "meta.json"), "r") as f:
            meta = json.load(f)
        calib_timestamp = meta["calib_timestamp"]
        calib_path = os.path.join(calib_path, "{}.npy".format(calib_timestamp))

        if self.robot_type == "single":
            Projector = SingleArmProjector
        else:
            Projector = DualArmProjector
        if calib_timestamp not in self.projectors:
            calib_file = np.load(calib_path, allow_pickle = True).item()
            cam_ids = calib_file["camera_serials_global"]
            # create projector cache
            self.projectors[calib_timestamp] = {}
            for cam_id in cam_ids:
                self.projectors[calib_timestamp][cam_id] = Projector(calib_path, cam_id)
        cam_ids = list(self.projectors[calib_timestamp].keys())

        return calib_timestamp, cam_ids

    def _augmentation(self, points, image_coords, tcps):
        translation_offsets = np.random.rand(3) * (self.aug_trans_max - self.aug_trans_min) + self.aug_trans_min
        rotation_angles = np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        rotation_angles = rotation_angles / 180 * np.pi  # tranform from degree to radius
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)
        center = points.mean(axis=0)

        points -= center
        points = apply_mat_to_pcd(points, aug_mat)
        points += center

        tcps[...,:3] -= center
        tcps = apply_mat_to_pose(tcps, aug_mat, rotation_rep = "quaternion")
        tcps[...,:3] += center

        c, h, w = image_coords.shape
        image_coords = image_coords.reshape(3, -1)
        image_coords -= center[:, np.newaxis]
        image_coords = (np.matmul(aug_mat[:3, :3], image_coords).T + aug_mat[:3, 3]).T
        image_coords += center[:, np.newaxis]
        image_coords = image_coords.reshape(3, h, w)

        return points, image_coords, tcps

    def _normalize_tcp(self, tcp_list):
        ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width) + 3(trans) + 6(rot) + 1(width)]'''
        tcp_list[:, :3] = (tcp_list[:, :3] - self.translation_min) / (self.translation_max - self.translation_min) * 2 - 1
        tcp_list[:, 9] = tcp_list[:, 9] / self.max_gripper_width * 2 - 1
        return tcp_list

    def load_point_cloud(self, colors, depths, intrinsics, depth_scale=1000., rescale_factor=1):
        if rescale_factor != 1:
            H, W = depths.shape
            h, w = int(H * rescale_factor), int(W * rescale_factor)
            colors = colors.transpose([2, 0, 1]).astype(np.float32)
            colors = torch.from_numpy(colors)
            colors = np.ascontiguousarray(resize_image(colors, [h,w]).numpy().transpose([1, 2, 0]))
            depths = depths.astype(np.float32)
            depths = torch.from_numpy(depths[np.newaxis])
            depths = resize_image(depths, [h,w], interpolation=T.InterpolationMode.NEAREST)[0]
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
        bbox3d = o3d.geometry.AxisAlignedBoundingBox(self.workspace_min, self.workspace_max)
        cloud = cloud.crop(bbox3d)
        # downsample
        cloud = cloud.voxel_down_sample(self.voxel_size)
        return cloud

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        cam_id = self.cam_ids[index]
        calib_timestamp = self.calib_timestamps[index]
        obs_frame_id = self.obs_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]

        # directories
        color_dir, depth_dir = self._resolve_image_dirs(data_path, cam_id)
        lowdim_dir = os.path.join(data_path, "lowdim")

        # load camera projector by calib timestamp and cam_id
        projector = self.projectors[calib_timestamp][cam_id]

        # load colors and depths
        colors = Image.open(os.path.join(color_dir, "{}.png".format(obs_frame_id)))
        if self.aug_color:
            colors = self.jitter(colors)
        colors = np.asarray(colors)
        depths = np.asarray(Image.open(os.path.join(depth_dir, "{}.png".format(obs_frame_id))), dtype = np.float32)

        mask_path = None
        mask01 = None
        image_mask_weight = None
        if self.mask_aware_enabled:
            # 按样本索引读取已对齐的 mask
            mask_path = self.mask_paths[index]
            mask_img = np.asarray(Image.open(mask_path).convert("L"), dtype = np.uint8)
            # mask 与 depth 分辨率不一致时做最近邻对齐
            if mask_img.shape != depths.shape:
                mask_img = torch.from_numpy(mask_img[np.newaxis].astype(np.float32))
                mask_img = resize_image(mask_img, list(depths.shape), interpolation = T.InterpolationMode.NEAREST)[0]
                mask_img = mask_img.numpy().astype(np.uint8)
            # 统一输出 1 为不可信区域的二值 mask
            if self.mask_white_is_untrusted:
                mask01 = (mask_img > self.mask_threshold).astype(np.float32)
            else:
                mask01 = (mask_img <= self.mask_threshold).astype(np.float32)

            # 生成 patch 级可信度用于 2D 降权
            mask01_t = torch.from_numpy(mask01).float().unsqueeze(0)
            mask01_t = resize_image(mask01_t, self.img_size, interpolation = T.InterpolationMode.NEAREST)
            mask_ratio = self.image_processor.image_coord_pooling(mask01_t)
            image_mask_weight = 1.0 - mask_ratio
        
        # 在深度图阶段应用 mask 做精确 3D 过滤
        depths_for_cloud = depths
        if self.mask_aware_enabled and self.mask_enable_3d_filter and mask01 is not None:
            depths_for_cloud = depths.copy()
            depths_for_cloud[mask01 > 0.5] = 0.0

        # point cloud
        intrinsics, depth_scale = projector.intrinsics[cam_id], projector.depth_scales[cam_id]
        cloud = self.load_point_cloud(colors, depths_for_cloud, intrinsics, depth_scale, rescale_factor=0.5)
        points = np.asarray(cloud.points)
        if len(points) == 0:
            if self.mask_aware_enabled and self.mask_enable_3d_filter and mask01 is not None:
                if self.mask_empty_cloud_policy == "fail_fast":
                    raise RuntimeError(
                        "Empty cloud after 3d filter at scene={} cam={} frame={}".format(
                            os.path.basename(data_path), cam_id, obs_frame_id
                        )
                    )
                print(
                    "[mask-3d-filter] empty cloud after mask, skip 3d filter for this sample: scene={} cam={} frame={}".format(
                        os.path.basename(data_path), cam_id, obs_frame_id
                    )
                )
                cloud = self.load_point_cloud(colors, depths, intrinsics, depth_scale, rescale_factor=0.5)
                points = np.asarray(cloud.points)
            if len(points) == 0:
                raise RuntimeError(
                    "Empty cloud after processing at scene={} cam={} frame={}".format(
                        os.path.basename(data_path), cam_id, obs_frame_id
                    )
                )

        # get organized point cloud
        image_coords = self.image_processor.get_image_coordinates(depths, intrinsics, depth_scale)

        # actions
        actions = []
        for frame_id in action_frame_ids:
            action = load_action(lowdim_dir, frame_id, self.robot_type, gripper_info_type = "command")
            if self.robot_type == "single":
                action[0: 7] = projector.project_tcp_to_camera_coord(
                    action[0: 7],
                    rotation_rep = "quaternion"
                )
            else:
                action[0: 7] = projector.project_tcp_to_camera_coord(
                    action[0: 7], 
                    robot = "left",
                    rotation_rep = "quaternion"
                )
                action[8: 15] = projector.project_tcp_to_camera_coord(
                    action[8: 15], 
                    robot = "right",
                    rotation_rep = "quaternion"
                )
            actions.append(action)
        actions = np.stack(actions)
        if self.robot_type == "dual":
            actions = actions.reshape(-1, 8) # for data aug and rot transform

        # point augmentations
        if self.aug_point:
            action_tcps = actions[:, :7]
            points, image_coords, action_tcps = self._augmentation(points, image_coords, action_tcps)
            actions[:, :7] = action_tcps

        # visualization
        if self.vis:
            vis_data(
                points,
                np.asarray(cloud.colors),
                actions[:, :7],
                self.workspace_min,
                self.workspace_max,
                self.translation_min,
                self.translation_max
            )
        
        # rotation transformation (to 6d)
        action_tcps = actions[:, :7]
        action_tcps = xyz_rot_transform(action_tcps, from_rep = "quaternion", to_rep = "rotation_6d")
        actions = np.concatenate([action_tcps, actions[:, -1:]], axis = -1)

        # normalization
        actions_normalized = self._normalize_tcp(actions.copy())
        if self.robot_type == "dual":
            actions = actions.reshape(-1, 20)
            actions_normalized = actions_normalized.reshape(-1, 20)

        # make voxel input
        # Upd Note. Make coords contiguous.
        coords = np.ascontiguousarray(points / self.voxel_size, dtype = np.int32)
        # Upd Note. API change.
        input_coords = [coords]
        input_feats = [points.astype(np.float32)]

        # convert to torch
        actions = torch.from_numpy(actions).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()
        colors, image_coords = self.image_processor.preprocess_images(colors, image_coords)

        ret_dict = {
            'cloud_coords': input_coords,
            'cloud_feats': input_feats,
            'image_coords': image_coords,
            'image_feats': colors,
            'action': actions,
            'action_normalized': actions_normalized,
        }
        if self.mask_aware_enabled:
            # 仅在 mask-aware 模式下返回 mask 相关字段
            ret_dict['mask01'] = torch.from_numpy(mask01).float()
            ret_dict['mask_path'] = mask_path
            ret_dict['image_mask_weight'] = image_mask_weight

        return ret_dict
        

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                ret_dict[key] = collate_fn([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        coords_batch = ret_dict['cloud_coords']
        feats_batch = ret_dict['cloud_feats']
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
        ret_dict['cloud_coords'] = coords_batch
        ret_dict['cloud_feats'] = feats_batch
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))
