import os

import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image, ImageDraw

from dataset.data_utils import load_action, resize_image, vis_data
from dataset.realworld import RealWorldDataset as BaseRealWorldDataset, collate_fn
from utils.transformation import xyz_rot_transform


class RealWorldDataset(BaseRealWorldDataset):
    """
    Experimental dataset that removes arm points from the cloud, injects a TCP
    anchor cluster, and restores local patch weights around the projected TCP.
    """

    def __init__(self, path, config, split="train"):
        self._arm_mask_lookup_cache = {}
        super().__init__(path, config, split=split)

        self.arm_mask_paths = []
        if self.tcp_anchor_enabled:
            self._build_arm_mask_paths()

    def _parse_config(self, config):
        super()._parse_config(config)

        tcp_cfg = getattr(config, "tcp_anchor", None)
        self.tcp_anchor_enabled = bool(getattr(tcp_cfg, "enabled", False))
        self.remove_arm_from_cloud = bool(
            getattr(tcp_cfg, "remove_arm_from_cloud", True)
        )

        arm_mask_root = getattr(tcp_cfg, "arm_mask_root", None)
        if arm_mask_root in [None, "", "null", "None"]:
            self.arm_mask_root = None
        else:
            self.arm_mask_root = str(arm_mask_root)

        self.arm_mask_subdir = str(getattr(tcp_cfg, "arm_mask_subdir", "arm"))
        self.anchor_num_points = int(getattr(tcp_cfg, "anchor_num_points", 7))
        self.anchor_radius_scale = float(getattr(tcp_cfg, "anchor_radius_scale", 0.75))
        self.anchor_color = np.asarray(
            getattr(tcp_cfg, "anchor_color", [1.0, 0.0, 0.0]), dtype=np.float32
        )
        self.anchor_color = np.clip(self.anchor_color, 0.0, 1.0)
        self.fail_if_missing_arm_mask = bool(
            getattr(tcp_cfg, "fail_if_missing_arm_mask", True)
        )
        self.patch_weight_mode = str(
            getattr(tcp_cfg, "patch_weight_mode", "anchor_only")
        )
        self.tcp_patch_radius = int(getattr(tcp_cfg, "tcp_patch_radius", 1))
        self.tcp_patch_weight_floor = float(
            getattr(tcp_cfg, "tcp_patch_weight_floor", 1.0)
        )

    def _validate_mask_aware_config(self):
        super()._validate_mask_aware_config()
        self._validate_tcp_anchor_config()

    def _validate_tcp_anchor_config(self):
        if not self.tcp_anchor_enabled:
            return

        if self.robot_type != "single":
            raise ValueError("tcp_anchor currently supports robot_type=single only")
        if self.anchor_num_points <= 0:
            raise ValueError("tcp_anchor.anchor_num_points must be > 0")
        if self.anchor_num_points > 27:
            raise ValueError("tcp_anchor.anchor_num_points must be <= 27")
        if self.anchor_radius_scale < 0:
            raise ValueError("tcp_anchor.anchor_radius_scale must be >= 0")
        if self.arm_mask_root is None:
            raise ValueError(
                "tcp_anchor.arm_mask_root is required when tcp_anchor.enabled is true"
            )
        if not os.path.isdir(self.arm_mask_root):
            raise FileNotFoundError(
                "tcp_anchor.arm_mask_root does not exist: {}".format(self.arm_mask_root)
            )
        if self.arm_mask_subdir in [None, "", "null", "None"]:
            raise ValueError(
                "tcp_anchor.arm_mask_subdir is required when tcp_anchor.enabled is true"
            )
        if self.patch_weight_mode not in {
            "anchor_only",
            "full_downweight",
            "no_2d_downweight_for_arm",
        }:
            raise ValueError(
                "Unsupported tcp_anchor.patch_weight_mode: {}".format(
                    self.patch_weight_mode
                )
            )
        if self.tcp_patch_radius < 0:
            raise ValueError("tcp_anchor.tcp_patch_radius must be >= 0")
        if not (0.0 <= self.tcp_patch_weight_floor <= 1.0):
            raise ValueError("tcp_anchor.tcp_patch_weight_floor must be in [0, 1]")

    def _resolve_lookup_dir(
        self, demo_path, cam_id, root_dir, leaf_subdir, fail_if_missing
    ):
        scene_name = os.path.basename(demo_path)
        cam_name = "cam_{}".format(cam_id)

        if root_dir in [None, "", "null", "None"]:
            if fail_if_missing:
                raise FileNotFoundError(
                    "Unable to resolve lookup directory for scene={} cam={} because root_dir is empty".format(
                        scene_name, cam_id
                    )
                )
            return None, []

        candidate_dirs = [
            os.path.join(root_dir, scene_name, cam_name, leaf_subdir),
            os.path.join(root_dir, scene_name, leaf_subdir, cam_name),
            os.path.join(root_dir, scene_name, leaf_subdir),
            os.path.join(root_dir, scene_name, cam_name),
            os.path.join(root_dir, scene_name),
            os.path.join(demo_path, cam_name, leaf_subdir),
        ]

        for lookup_dir in candidate_dirs:
            if not os.path.isdir(lookup_dir):
                continue
            mask_files = self._list_sorted_pngs(lookup_dir)
            if len(mask_files) > 0:
                return lookup_dir, mask_files

        if fail_if_missing:
            raise FileNotFoundError(
                "Unable to resolve lookup directory for scene={} cam={} under root_dir={} subdir={}".format(
                    scene_name, cam_id, root_dir, leaf_subdir
                )
            )
        return None, []

    def _build_aligned_lookup(
        self,
        demo_path,
        cam_id,
        color_dir,
        root_dir,
        leaf_subdir,
        log_prefix,
        fail_if_missing,
    ):
        color_files = self._list_sorted_pngs(color_dir)
        if len(color_files) == 0:
            raise RuntimeError("No color frames found in {}".format(color_dir))

        lookup_dir, mask_files = self._resolve_lookup_dir(
            demo_path,
            cam_id,
            root_dir=root_dir,
            leaf_subdir=leaf_subdir,
            fail_if_missing=fail_if_missing,
        )
        if lookup_dir is None:
            return {}

        if len(color_files) != len(mask_files):
            raise ValueError(
                "Color/mask length mismatch for scene={} cam={}: {} vs {}".format(
                    os.path.basename(demo_path),
                    cam_id,
                    len(color_files),
                    len(mask_files),
                )
            )

        lookup = {}
        for color_name, mask_name in zip(color_files, mask_files):
            color_stem = os.path.splitext(color_name)[0]
            mask_path = os.path.join(lookup_dir, mask_name)
            lookup[color_stem] = mask_path
            try:
                lookup[int(color_stem)] = mask_path
            except ValueError:
                pass

        if self.mask_log_alignment_preview and len(color_files) > 0:
            print(
                "[{}] scene={} cam={} n={} first={}=>{} last={}=>{}".format(
                    log_prefix,
                    os.path.basename(demo_path),
                    cam_id,
                    len(color_files),
                    color_files[0],
                    mask_files[0],
                    color_files[-1],
                    mask_files[-1],
                )
            )

        return lookup

    def _build_arm_mask_paths(self):
        for data_path, cam_id, obs_frame_id in zip(
            self.data_paths, self.cam_ids, self.obs_frame_ids
        ):
            cache_key = (data_path, cam_id)
            color_dir, _ = self._resolve_image_dirs(data_path, cam_id)
            if cache_key not in self._arm_mask_lookup_cache:
                self._arm_mask_lookup_cache[cache_key] = self._build_aligned_lookup(
                    data_path,
                    cam_id,
                    color_dir,
                    root_dir=self.arm_mask_root,
                    leaf_subdir=self.arm_mask_subdir,
                    log_prefix="arm-mask-align",
                    fail_if_missing=self.fail_if_missing_arm_mask,
                )
            lookup = self._arm_mask_lookup_cache[cache_key]
            arm_mask_path = lookup.get(
                obs_frame_id, lookup.get(str(obs_frame_id), None)
            )
            if arm_mask_path is None and self.fail_if_missing_arm_mask:
                raise KeyError(
                    "Missing aligned arm mask for scene={} cam={} frame={}".format(
                        os.path.basename(data_path), cam_id, obs_frame_id
                    )
                )
            self.arm_mask_paths.append(arm_mask_path)

        if len(self.arm_mask_paths) != len(self.data_paths):
            raise RuntimeError(
                "Arm mask path count mismatch: {} vs {}".format(
                    len(self.arm_mask_paths), len(self.data_paths)
                )
            )

    def _load_mask01(self, mask_path, target_shape):
        if mask_path is None:
            return None

        mask_img = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if mask_img.shape != target_shape:
            mask_t = torch.from_numpy(mask_img[np.newaxis].astype(np.float32))
            mask_t = resize_image(
                mask_t, list(target_shape), interpolation=T.InterpolationMode.NEAREST
            )[0]
            mask_img = mask_t.numpy().astype(np.uint8)

        if self.mask_white_is_untrusted:
            return (mask_img > self.mask_threshold).astype(np.float32)
        return (mask_img <= self.mask_threshold).astype(np.float32)

    def _combine_masks(self, *mask_list):
        combined = None
        for mask in mask_list:
            if mask is None:
                continue
            combined = mask.copy() if combined is None else np.maximum(combined, mask)
        return combined

    def _load_observation_tcp_camera(self, lowdim_dir, obs_frame_id, projector):
        obs_action = load_action(
            lowdim_dir, obs_frame_id, self.robot_type, gripper_info_type="command"
        )
        obs_tcp = projector.project_tcp_to_camera_coord(
            obs_action[0:7], rotation_rep="quaternion"
        )
        return obs_tcp.astype(np.float32)

    def _project_camera_xyz_to_pixel(self, xyz, intrinsics, image_shape):
        xyz = np.asarray(xyz, dtype=np.float32)
        if xyz.shape[0] < 3 or not np.isfinite(xyz[:3]).all() or xyz[2] <= 1e-6:
            return None

        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        pixel_x = fx * float(xyz[0]) / float(xyz[2]) + cx
        pixel_y = fy * float(xyz[1]) / float(xyz[2]) + cy
        height, width = image_shape

        if not np.isfinite(pixel_x) or not np.isfinite(pixel_y):
            return None
        if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
            return None
        return np.asarray([pixel_x, pixel_y], dtype=np.float32)

    def _pixel_to_patch_coord(self, pixel_xy, image_shape):
        if pixel_xy is None:
            return None

        image_h, image_w = image_shape
        patch_h, patch_w = self.img_coord_size
        patch_x = int(np.floor(float(pixel_xy[0]) * patch_w / image_w))
        patch_y = int(np.floor(float(pixel_xy[1]) * patch_h / image_h))
        patch_x = int(np.clip(patch_x, 0, patch_w - 1))
        patch_y = int(np.clip(patch_y, 0, patch_h - 1))
        return np.asarray([patch_x, patch_y], dtype=np.int32)

    def _make_anchor_offsets(self):
        radius = self.anchor_radius_scale * self.voxel_size
        offsets = [
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            np.asarray([0.0, -1.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        ]

        if self.anchor_num_points > len(offsets):
            for dx in [-1.0, 0.0, 1.0]:
                for dy in [-1.0, 0.0, 1.0]:
                    for dz in [-1.0, 0.0, 1.0]:
                        if dx == 0.0 and dy == 0.0 and dz == 0.0:
                            continue
                        candidate = np.asarray([dx, dy, dz], dtype=np.float32)
                        if any(np.array_equal(candidate, offset) for offset in offsets):
                            continue
                        offsets.append(candidate)
                        if len(offsets) >= self.anchor_num_points:
                            break
                    if len(offsets) >= self.anchor_num_points:
                        break
                if len(offsets) >= self.anchor_num_points:
                    break

        return np.stack(offsets[: self.anchor_num_points], axis=0) * radius

    def _inject_tcp_anchor(self, points, colors, tcp_center_xyz):
        tcp_center_xyz = np.asarray(tcp_center_xyz[:3], dtype=np.float32)
        anchor_points = tcp_center_xyz[None, :] + self._make_anchor_offsets()
        anchor_colors = np.tile(
            self.anchor_color.reshape(1, 3), (anchor_points.shape[0], 1)
        ).astype(np.float32)
        points = np.concatenate([points, anchor_points], axis=0)
        colors = np.concatenate([colors, anchor_colors], axis=0)
        return points, colors

    def _restore_tcp_patch_weights(self, patch_weight, tcp_patch_xy):
        if (
            patch_weight is None
            or tcp_patch_xy is None
            or self.patch_weight_mode != "anchor_only"
        ):
            return patch_weight

        patch_x, patch_y = int(tcp_patch_xy[0]), int(tcp_patch_xy[1])
        patch_h, patch_w = patch_weight.shape[-2:]
        x0 = max(0, patch_x - self.tcp_patch_radius)
        x1 = min(patch_w, patch_x + self.tcp_patch_radius + 1)
        y0 = max(0, patch_y - self.tcp_patch_radius)
        y1 = min(patch_h, patch_y + self.tcp_patch_radius + 1)
        patch_weight[:, y0:y1, x0:x1] = torch.clamp(
            patch_weight[:, y0:y1, x0:x1],
            min=self.tcp_patch_weight_floor,
            max=1.0,
        )
        return patch_weight

    def _save_arm_mask_overlay(
        self, colors, arm_mask01, tcp_pixel_xy, save_dir, save_prefix
    ):
        if save_dir is None or arm_mask01 is None or not self.vis_save_png:
            return

        os.makedirs(save_dir, exist_ok=True)
        color_img = colors.astype(np.float32)
        overlay = color_img.copy()
        mask = arm_mask01 > 0.5
        overlay[mask, 0] = 255.0
        overlay[mask, 1] *= 0.35
        overlay[mask, 2] *= 0.35
        overlay = np.clip(0.45 * color_img + 0.55 * overlay, 0.0, 255.0).astype(
            np.uint8
        )
        overlay_img = Image.fromarray(overlay)

        if tcp_pixel_xy is not None:
            draw = ImageDraw.Draw(overlay_img)
            px = int(round(float(tcp_pixel_xy[0])))
            py = int(round(float(tcp_pixel_xy[1])))
            radius = 8
            draw.line((px - radius, py, px + radius, py), fill=(0, 255, 0), width=2)
            draw.line((px, py - radius, px, py + radius), fill=(0, 255, 0), width=2)

        overlay_path = os.path.join(save_dir, "{}_arm_overlay.png".format(save_prefix))
        overlay_img.save(overlay_path)
        print("[vis] saved arm overlay: {}".format(overlay_path))

    def _save_patch_weight_preview(
        self, patch_weight, tcp_patch_xy, save_dir, save_prefix
    ):
        if save_dir is None or patch_weight is None or not self.vis_save_png:
            return

        os.makedirs(save_dir, exist_ok=True)
        grid = patch_weight.detach().cpu().numpy()
        if grid.ndim == 3:
            grid = grid[0]
        grid = np.clip(grid, 0.0, 1.0)
        preview = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        preview[..., 0] = np.round((1.0 - grid) * 255.0).astype(np.uint8)
        preview[..., 1] = np.round(grid * 255.0).astype(np.uint8)
        preview[..., 2] = 32
        preview_img = Image.fromarray(preview).resize(
            (grid.shape[1] * 24, grid.shape[0] * 24), Image.NEAREST
        )

        if tcp_patch_xy is not None:
            draw = ImageDraw.Draw(preview_img)
            patch_x = int(tcp_patch_xy[0])
            patch_y = int(tcp_patch_xy[1])
            cell = 24
            draw.rectangle(
                (
                    patch_x * cell,
                    patch_y * cell,
                    (patch_x + 1) * cell - 1,
                    (patch_y + 1) * cell - 1,
                ),
                outline=(255, 255, 255),
                width=2,
            )

        preview_path = os.path.join(
            save_dir, "{}_tcp_patch_weight.png".format(save_prefix)
        )
        preview_img.save(preview_path)
        print("[vis] saved patch-weight preview: {}".format(preview_path))

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        cam_id = self.cam_ids[index]
        calib_timestamp = self.calib_timestamps[index]
        obs_frame_id = self.obs_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]

        color_dir, depth_dir = self._resolve_image_dirs(data_path, cam_id)
        lowdim_dir = os.path.join(data_path, "lowdim")
        projector = self.projectors[calib_timestamp][cam_id]

        colors = Image.open(os.path.join(color_dir, "{}.png".format(obs_frame_id)))
        if self.aug_color:
            colors = self.jitter(colors)
        colors = np.asarray(colors)
        depths = np.asarray(
            Image.open(os.path.join(depth_dir, "{}.png".format(obs_frame_id))),
            dtype=np.float32,
        )

        intrinsics = projector.intrinsics[cam_id]
        depth_scale = projector.depth_scales[cam_id]

        mask_path = self.mask_paths[index] if self.mask_aware_enabled else None
        mask01 = (
            self._load_mask01(mask_path, depths.shape)
            if self.mask_aware_enabled
            else None
        )

        arm_mask_path = self.arm_mask_paths[index] if self.tcp_anchor_enabled else None
        arm_mask01 = (
            self._load_mask01(arm_mask_path, depths.shape)
            if self.tcp_anchor_enabled
            else None
        )

        tcp_obs = (
            self._load_observation_tcp_camera(lowdim_dir, obs_frame_id, projector)
            if self.tcp_anchor_enabled
            else None
        )
        tcp_pixel_xy = (
            self._project_camera_xyz_to_pixel(tcp_obs[:3], intrinsics, depths.shape)
            if tcp_obs is not None
            else None
        )
        tcp_patch_xy = (
            self._pixel_to_patch_coord(tcp_pixel_xy, depths.shape)
            if tcp_pixel_xy is not None
            else None
        )

        patch_mask01 = None
        if (
            self.mask_aware_enabled
            and self.mask_enable_2d_reweight
            and mask01 is not None
        ):
            patch_mask01 = mask01.copy()
        if (
            self.tcp_anchor_enabled
            and arm_mask01 is not None
            and self.patch_weight_mode in {"anchor_only", "full_downweight"}
        ):
            patch_mask01 = self._combine_masks(patch_mask01, arm_mask01)

        image_mask_weight = None
        patch_weight_preview = None
        if patch_mask01 is not None:
            patch_mask_t = torch.from_numpy(patch_mask01).float().unsqueeze(0)
            patch_mask_t = resize_image(
                patch_mask_t, self.img_size, interpolation=T.InterpolationMode.NEAREST
            )
            mask_ratio = self.image_processor.image_coord_pooling(patch_mask_t)
            patch_weight_preview = 1.0 - mask_ratio
            patch_weight_preview = self._restore_tcp_patch_weights(
                patch_weight_preview, tcp_patch_xy
            )

            flat_weight = patch_weight_preview.reshape(1, -1)
            interp_tail = torch.tensor(
                [[self.mask_r_min, self.mask_interp_eps, self.mask_interp_tiny]],
                dtype=flat_weight.dtype,
            )
            image_mask_weight = torch.cat([flat_weight, interp_tail], dim=1)

        depths_for_cloud = depths.copy()
        applied_3d_filter = False
        if (
            self.mask_aware_enabled
            and self.mask_enable_3d_filter
            and mask01 is not None
        ):
            depths_for_cloud[mask01 > 0.5] = 0.0
            applied_3d_filter = True
        if (
            self.tcp_anchor_enabled
            and self.remove_arm_from_cloud
            and arm_mask01 is not None
        ):
            depths_for_cloud[arm_mask01 > 0.5] = 0.0
            applied_3d_filter = True

        cloud = self.load_point_cloud(
            colors, depths_for_cloud, intrinsics, depth_scale, rescale_factor=0.5
        )
        points = np.asarray(cloud.points)
        vis_colors = np.asarray(cloud.colors)

        if len(points) == 0 and applied_3d_filter:
            if self.mask_empty_cloud_policy == "fail_fast":
                raise RuntimeError(
                    "Empty cloud after 3d filter at scene={} cam={} frame={}".format(
                        os.path.basename(data_path), cam_id, obs_frame_id
                    )
                )
            print(
                "[tcp-anchor] empty cloud after filtering, fallback to unfiltered cloud: scene={} cam={} frame={}".format(
                    os.path.basename(data_path), cam_id, obs_frame_id
                )
            )
            cloud = self.load_point_cloud(
                colors, depths, intrinsics, depth_scale, rescale_factor=0.5
            )
            points = np.asarray(cloud.points)
            vis_colors = np.asarray(cloud.colors)

        if len(points) == 0:
            raise RuntimeError(
                "Empty cloud after processing at scene={} cam={} frame={}".format(
                    os.path.basename(data_path), cam_id, obs_frame_id
                )
            )

        if self.tcp_anchor_enabled and tcp_obs is not None:
            points, vis_colors = self._inject_tcp_anchor(
                points, vis_colors, tcp_obs[:3]
            )

        image_coords = self.image_processor.get_image_coordinates(
            depths, intrinsics, depth_scale
        )

        actions = []
        for frame_id in action_frame_ids:
            action = load_action(
                lowdim_dir, frame_id, self.robot_type, gripper_info_type="command"
            )
            action[0:7] = projector.project_tcp_to_camera_coord(
                action[0:7], rotation_rep="quaternion"
            )
            actions.append(action)
        actions = np.stack(actions)

        if self.aug_point:
            action_tcps = actions[:, :7]
            points, image_coords, action_tcps = self._augmentation(
                points, image_coords, action_tcps
            )
            actions[:, :7] = action_tcps

        if self.vis:
            scene_name = os.path.basename(data_path)
            vis_prefix = "{}_cam{}_obs{}_idx{}".format(
                scene_name, cam_id, obs_frame_id, index
            )
            self._save_arm_mask_overlay(
                colors, arm_mask01, tcp_pixel_xy, self.vis_save_dir, vis_prefix
            )
            self._save_patch_weight_preview(
                patch_weight_preview, tcp_patch_xy, self.vis_save_dir, vis_prefix
            )
            vis_data(
                points,
                vis_colors,
                actions[:, :7],
                self.workspace_min,
                self.workspace_max,
                self.translation_min,
                self.translation_max,
                save_dir=self.vis_save_dir,
                save_prefix=vis_prefix,
                save_png=self.vis_save_png,
                save_ply=self.vis_save_ply,
                force_offscreen=self.vis_force_offscreen,
            )

        action_tcps = actions[:, :7]
        action_tcps = xyz_rot_transform(
            action_tcps, from_rep="quaternion", to_rep="rotation_6d"
        )
        actions = np.concatenate([action_tcps, actions[:, -1:]], axis=-1)

        actions_normalized = self._normalize_tcp(actions.copy())

        coords = np.ascontiguousarray(points / self.voxel_size, dtype=np.int32)
        input_coords = [coords]
        input_feats = [points.astype(np.float32)]

        actions = torch.from_numpy(actions).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()
        colors_t, image_coords_t = self.image_processor.preprocess_images(
            colors, image_coords
        )

        ret_dict = {
            "cloud_coords": input_coords,
            "cloud_feats": input_feats,
            "image_coords": image_coords_t,
            "image_feats": colors_t,
            "action": actions,
            "action_normalized": actions_normalized,
        }

        if self.mask_aware_enabled:
            ret_dict["mask01"] = torch.from_numpy(mask01).float()
            ret_dict["mask_path"] = mask_path
        if image_mask_weight is not None:
            ret_dict["image_mask_weight"] = image_mask_weight
        if self.tcp_anchor_enabled:
            ret_dict["arm_mask_path"] = arm_mask_path

        return ret_dict
