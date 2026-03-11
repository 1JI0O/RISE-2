# eval_mask_final.py — 完整推理工作流文档

> 文件路径：`sam2/eval_mask_final.py`
> 最后更新：2026-03-10

---

## 目录

1. [整体架构](#1-整体架构)
2. [启动与配置加载](#2-启动与配置加载)
3. [模型与组件初始化](#3-模型与组件初始化)
4. [SAM2 Runtime 初始化](#4-sam2-runtime-初始化)
5. [Rollout 主循环](#5-rollout-主循环)
6. [SAM2 分割流程详解](#6-sam2-分割流程详解)
   - [6.1 冷启动：首帧交互标注](#61-冷启动首帧交互标注)
   - [6.2 后续帧：在线流式推理](#62-后续帧在线流式推理)
   - [6.3 周期重置机制](#63-周期重置机制)
   - [6.4 最终 mask 合成](#64-最终-mask-合成)
7. [Mask 规范化与下游使用](#7-mask-规范化与下游使用)
8. [3D 点云过滤分支](#8-3d-点云过滤分支)
9. [2D 图像 patch 重权分支](#9-2d-图像-patch-重权分支)
10. [策略推理与动作预测](#10-策略推理与动作预测)
11. [动作投影与 Ensemble](#11-动作投影与-ensemble)
12. [回退逻辑与异常统计](#12-回退逻辑与异常统计)
13. [配置参考](#13-配置参考)
14. [关键数据类型速查](#14-关键数据类型速查)

---

## 1. 整体架构

```
evaluate()
│
├─ 配置加载 (_build_mask_aware_cfg / _build_sam2_cfg)
├─ 策略模型加载 (RISE2)
├─ 投影器 Projector 初始化
├─ ImageProcessor 初始化
├─ Robot Agent 初始化
├─ EnsembleBuffer 初始化
├─ SAM2 VideoPredictor Runtime 初始化 (_init_sam2_runtime)
│
└─ Rollout 主循环 (for t in range(max_steps))
    │
    ├─ [每 num_inference_steps 步执行一次策略推理]
    │   │
    │   ├─ 获取观测 agent.get_global_observation() → colors, depths
    │   │
    │   ├─ SAM2 分割 (_safe_infer_mask)
    │   │   ├─ [首帧] _cold_start_interactive (cv2 交互标注)
    │   │   └─ [后续] propagate_in_video (VideoPredictor 跟踪)
    │   │
    │   ├─ 3D 点云过滤 (mask01 → 零化机械臂深度像素)
    │   ├─ create_input → coords, points, cloud (MinkowskiEngine 稀疏张量)
    │   ├─ 2D patch 重权 (_build_image_mask_weight)
    │   ├─ 策略前向 policy(cloud_data, colors, image_coords, image_mask_weight)
    │   ├─ 动作反归一化 process_state
    │   ├─ 坐标投影 projector.project_tcp_to_base_coord
    │   └─ ensemble_buffer.add_action
    │
    └─ ensemble_buffer.get_action → agent.action (每步执行)
```

---

## 2. 启动与配置加载

### 入口

```bash
python sam2/eval_mask_final.py \
    --type local \
    --calib_rise2 calib_rise2/ \
    --calib_airexo calib_airexo/ \
    --config config/dual_teleop_dino.yaml \
    --ckpt logs/collect_toys
```

### 配置层次

```
config (YAML)
└── mask_aware
    ├── enabled            bool    是否启用 mask-aware 推理
    ├── enable_3d_filter   bool    是否用 mask 过滤深度图（屏蔽机械臂点云）
    ├── enable_2d_reweight bool    是否用 mask 构建 patch 可信度权重
    ├── mask_threshold     float   mask 二值化阈值（默认 0）
    ├── mask_white_is_untrusted bool  像素值 > threshold 为不可信（机械臂区域）
    ├── r_min              float   插值最小半径（1e-3）
    ├── interp_eps / interp_tiny float  数值稳定防零
    ├── infer_none_policy  str     "no_mask_fallback" | "fail_fast"
    ├── empty_cloud_policy str     "warn_and_skip_filter" | "fail_fast"
    └── sam2
        ├── enabled              bool    是否启用 SAM2
        ├── config_file          str     SAM2 模型配置 yaml
        ├── ckpt_path            str     微调权重路径
        ├── device               str     "cuda_if_available" | "cuda" | "cpu" | "mps"
        ├── arm_obj_id           int     机械臂对象 ID（默认 1）
        ├── gripper_obj_id       int     Gripper 对象 ID（默认 2）
        ├── dilate_radius        int     arm/gripper mask 膨胀半径（像素，默认 10）
        └── reset_every_n_steps  int     每 N 帧重置 inference_state（默认 30）
```

`_build_mask_aware_cfg` 和 `_build_sam2_cfg` 负责从 YAML 合并默认值并做类型校验，任何解析异常都会 fallback 为禁用。

---

## 3. 模型与组件初始化

### 策略模型 RISE2

- `args.type == "local"` 时本地加载 `RISE2`
- 支持单臂（`action_dim=10`）和双臂（`action_dim=20`）
- `policy.load_state_dict(..., strict=False)` 加载微调权重
- `policy.eval()` 切入推理模式

### Projector（TCP 坐标投影器）

- `SingleArmProjector` / `DualArmProjector`
- 使用 rise2 标定文件将相机系 TCP 转为机器人基坐标系

### ImageProcessor

- 根据 image_enc 类型（`resnet18` / `dinov2*` / `dinov3*`）选择对应分辨率
- 负责图像预处理：`preprocess_images` → resize + normalize
- `get_image_coordinates(depth, intrinsics, depth_scale)` → 图像坐标
- `image_coord_pooling` → patch 级 mask 比例（用于 2D reweight）

### Robot Agent

- `SingleArmAgent` / `DualArmAgent`
- `agent.get_global_observation()` → `(colors: np.uint8 HWC, depths: np.uint16 HW)`
- `agent.action(step_action)` → 发送关节控制指令
- `agent.intrinsics` → 相机内参矩阵（3×3）
- `agent.camera.depth_scale` → 深度单位换算系数

### EnsembleBuffer

- 存储最近若干步预测动作序列
- `add_action(action, t)` → 缓冲加权
- `get_action()` → 按 ensemble_mode 加权平均当前步动作

---

## 4. SAM2 Runtime 初始化

### 条件

仅在以下条件全部满足时初始化：
- `config.mask_aware.enabled == True`
- `args.type == "local"`
- `config.mask_aware.sam2.enabled == True`

### 流程

```python
_sam2_runtime = _init_sam2_runtime(config.mask_aware.sam2)
```

`_init_sam2_runtime` 内部：

1. `_resolve_sam2_device(device_str)` → 解析设备（cuda / mps / cpu，带可用性检测）
2. `build_sam2_video_predictor(config_file, ckpt_path, device, mode="eval")` → 加载微调后的 SAM2VideoPredictor
3. 返回 runtime 字典（模块级全局变量 `_sam2_runtime`）：

```python
{
    "enabled":               True,
    "cfg":                   sam2_cfg,        # edict
    "device":                torch.device,
    "predictor":             SAM2VideoPredictor,
    "inference_state":       None,            # 冷启动后设置
    "frame_idx":             0,               # 当前帧在 inference_state buffer 中的索引
    "last_arm_mask_raw":     None,            # bool (H, W)，未膨胀
    "last_gripper_mask_raw": None,            # bool (H, W)，未膨胀
    "last_reset_frame_idx":  0,               # 上次重置时的 frame_idx
    "last_fail_reason":      None,            # 最近一次失败原因（字符串）
}
```

### 为何使用 VideoPredictor 而非 ImagePredictor

`SAM2VideoPredictor` 内置跨帧时序 memory，与微调训练分布一致，跟踪稳定性优于无状态的 `SAM2ImagePredictor`。
标准 `init_state(video_path)` 需要全量帧预加载，不适合在线部署，故手动构建 `inference_state`（见第 6 节）。

---

## 5. Rollout 主循环

```python
with torch.inference_mode():
    for t in range(config.deploy.max_steps):
        if t % config.deploy.num_inference_steps == 0:
            # 每 num_inference_steps 步进行一次完整策略推理
            ...
        # 每步均从 ensemble buffer 取动作并执行
        step_action = ensemble_buffer.get_action()
        agent.action(step_action, rotation_rep="rotation_6d")
```

**num_inference_steps 的意义**：策略以较低频率（例如每 5 步）推理一次，输出一段动作序列，ensemble buffer 在更高频率下逐步取出执行。这减少了计算压力，同时通过 ensemble 平滑动作抖动。

---

## 6. SAM2 分割流程详解

分割入口为 `_safe_infer_mask`，内部调用 `infer_mask`，捕获所有异常并转化为 `None`（触发 no_mask_fallback）。

### 6.1 冷启动：首帧交互标注

**触发条件**：`_sam2_runtime["inference_state"] is None`（第一次调用 `infer_mask`）

**步骤**：

1. `_build_video_inference_state_from_np(predictor, color_np)`
   - 对当前帧 resize 到 `predictor.image_size`（通常 1024）
   - ImageNet 均值方差归一化 → `(1, 3, H, W)` float32 tensor，放入 GPU
   - 手动构建 `inference_state` dict（替代 `init_state(video_path)`），包含：
     - `images`：单帧张量
     - `num_frames`：1
     - `video_height / video_width`：原始分辨率
     - 空的 `point_inputs_per_obj`、`mask_inputs_per_obj`、`cached_features` 等
   - 调用 `predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)` 预热视觉骨干并缓存帧 0 特征

2. `_cold_start_interactive(predictor, inference_state, cfg, color_np)`
   弹出 cv2 窗口，显示当前帧，等待用户交互标注：

   | 操作 | 含义 |
   |------|------|
   | `[a]` | 切换到 ARM 模式（obj_id = arm_obj_id，默认 1） |
   | `[g]` | 切换到 GRIPPER 模式（obj_id = gripper_obj_id，默认 2） |
   | 左键点击 | 添加正样本点（前景） |
   | 右键点击 | 添加负样本点（背景） |
   | `[r]` | 清空当前对象的所有标注点 |
   | Enter / Space | 确认标注，关闭窗口继续推理 |
   | ESC | 中止冷启动，本帧返回 None |

   每次点击后实时调用 `predictor.add_new_points_or_box`（`normalize_coords=True`，坐标为图像像素坐标）并刷新 mask 叠加显示。
   标注完成后返回 `(arm_raw: bool HW, gripper_raw: bool HW)`。

3. 将 `inference_state`、`frame_idx=0`、`last_arm_mask_raw`、`last_gripper_mask_raw` 存入 `_sam2_runtime`
4. 立即计算并返回首帧 final mask（`_compute_final_mask`）

### 6.2 后续帧：在线流式推理

**正常路径**（非重置帧）：

1. `_append_frame_to_video_state(predictor, inference_state, color_np)`
   - 预处理新帧 → `(1, 3, H, W)` tensor
   - `torch.cat` 拼接到 `inference_state["images"]`（buffer 逐帧增长）
   - `inference_state["num_frames"] += 1`
   - `frame_idx += 1`

2. `predictor.propagate_in_video(inference_state, start_frame_idx=frame_idx, max_frame_num_to_track=1)`
   - 内部使用跨帧时序 memory 对第 `frame_idx` 帧进行跟踪推理
   - 返回 `(frame_idx, obj_ids, mask_logits)`
   - `mask_logits[i].squeeze() > 0.0` → bool mask
   - 按 obj_id 分配到 `arm_raw` / `gripper_raw`

3. 更新 `_sam2_runtime["last_arm_mask_raw"]` 和 `last_gripper_mask_raw`（存储未膨胀 raw mask，供下次跟踪使用）

4. 返回 `_compute_final_mask(arm_raw, gripper_raw, dilate_radius) * 255`

### 6.3 周期重置机制

**触发条件**：`(frame_idx - last_reset_frame_idx) >= reset_every_n_steps` 且上一帧 arm_raw 有效

**目的**：`images` buffer 随时间无限增长会耗尽显存。每 N 步将 buffer 缩减为单帧，利用上一帧的 raw mask 重新锚定跟踪。

**步骤**：

1. 从当前帧重建 `inference_state`（仅含 1 帧）
2. `predictor.add_new_mask(inference_state, frame_idx=0, obj_id=arm_obj_id, mask=arm_raw_prev)` → 以上一帧 mask 作为 prompt
3. 若 gripper_raw_prev 不为 None，同样 add_new_mask for gripper
4. 重置 `frame_idx = 0`，`last_reset_frame_idx = 0`

重置后继续正常调用 `propagate_in_video`，跟踪状态从当前帧重新建立。

> **注意**：`predictor.reset_state` 不清除 `cached_features`，因此视觉骨干特征缓存在重置后仍然有效。

### 6.4 最终 mask 合成

```python
def _compute_final_mask(arm_raw, gripper_raw, dilate_radius):
    arm_dilated = _dilate_mask_bool(arm_raw, dilate_radius)
    if gripper_raw is not None:
        grp_dilated = _dilate_mask_bool(gripper_raw, dilate_radius)
        return np.logical_and(arm_dilated, np.logical_not(grp_dilated))
    return arm_dilated
```

- **arm_raw** 和 **gripper_raw** 为 raw（未膨胀）bool mask，存入 runtime 供下帧跟踪
- 最终 mask = `dilate(arm_raw)  AND  NOT dilate(gripper_raw)`
- 膨胀使用正方形 kernel，边长 = `2 * dilate_radius + 1`
- 结果转为 `uint8 * 255`：机械臂区域（去除 gripper 后）= 255，其余 = 0

**设计意图**：

| 区域 | 最终 mask 值 | 含义 |
|------|-------------|------|
| arm 膨胀区域（去除 gripper 部分） | 255 | 不可信区域，屏蔽 |
| gripper 膨胀区域 | 0 | 保留（gripper 操作区为有效观测） |
| 背景 | 0 | 保留 |

arm 和 gripper 共用同一 `dilate_radius`，通过 Boolean 差集分离两者。

---

## 7. Mask 规范化与下游使用

`_safe_infer_mask` 调用 `_normalize_mask01(raw_mask, depth.shape[:2], mask_cfg)`：

1. 统一转为 numpy（支持 Tensor / ndarray 输入）
2. 处理多余维度（单通道 `(1, H, W)` 或 `(H, W, 1)`）
3. 若 mask 分辨率与深度图不匹配 → `cv2.INTER_NEAREST` 最近邻 resize
4. 二值化：
   - `mask_white_is_untrusted=True`（默认）：`mask01 = (mask_np > threshold).astype(float32)`
   - `mask01 = 1.0` 表示不可信（机械臂区域），`mask01 = 0.0` 表示可信

返回：`mask01 (H, W) float32`，值域 `{0.0, 1.0}`

---

## 8. 3D 点云过滤分支

```python
if mask_enabled and config.mask_aware.enable_3d_filter and mask01 is not None:
    depths_for_cloud = depths.copy()
    depths_for_cloud[mask01 > 0.5] = 0   # 机械臂区域深度置零
```

将机械臂区域的深度值归零，使后续点云重建时该区域不产生点，避免机械臂遮挡影响场景理解。

**点云构建流程**（`create_input` → `create_point_cloud`）：

1. （可选）rescale depths & colors
2. `o3d.geometry.RGBDImage.create_from_color_and_depth` → RGBD 图像
3. `PointCloud.create_from_rgbd_image(rgbd, intrinsics)` → 稠密点云
4. `cloud.crop(bbox3d)` → 按 workspace 裁剪（`config.deploy.workspace.min/max`）
5. `cloud.voxel_down_sample(config.data.voxel_size)` → 体素降采样

**空点云回退**：

- 若过滤后点云为空 → `warn_and_skip_filter`（用原始深度重建）或 `fail_fast`
- 若点云存在非法数值（NaN/Inf）→ 用原始深度重建，记录 `points_nonfinite`

---

## 9. 2D 图像 patch 重权分支

```python
image_mask_weight = _build_image_mask_weight(mask01, image_processor)
```

1. `mask01` → `(1, H, W)` tensor
2. resize 到 image_processor.img_size（同图像 encoder 输入分辨率）
3. `image_processor.image_coord_pooling(mask_tensor)` → 每个 image patch 内 mask 的平均比例（`mask_ratio`）
4. `image_mask_weight = (1.0 - mask_ratio).clamp(0, 1)` → patch 可信度权重
   - 机械臂比例越高的 patch，权重越低（趋近 0）
   - 无机械臂的 patch，权重 = 1.0

该权重作为 `image_mask_weight` 传入 `policy(...)` 的 attention 模块，降低机械臂遮挡区域对策略的影响。

**异常处理**：构建失败或出现非有限数值时，权重置 None（降级为不加权），记录 `reweight_fallback` / `weight_nonfinite`。

---

## 10. 策略推理与动作预测

```python
pred_raw_action = policy(
    cloud_data,          # MinkowskiEngine SparseTensor
    colors,              # (1, C, H, W) 预处理图像
    image_coords,        # (1, N_pts, 2) 点云到图像的投影坐标
    image_mask_weight,   # (1, N_patch) 或 None
    actions=None,
).squeeze(0).cpu().numpy()
```

- `cloud_data`：3D 点云的稀疏张量（MinkowskiEngine 格式），`coords` 为体素坐标，`feats` 为 XYZ 坐标
- 输出 `pred_raw_action`：归一化动作序列，形状 `(num_action, action_dim)`

### 动作维度

| 机器人类型 | action_dim | 含义 |
|-----------|-----------|------|
| single    | 10        | [trans(3), rot6d(6), gripper(1)] |
| dual      | 20        | left: [trans(3), rot6d(6), gripper(1)], right: [trans(3), rot6d(6), gripper(1)] |

`process_state(..., to_control=True)` 将归一化动作反归一化：
- translation: `[-1, 1]` → `[trans_min, trans_max]`（米）
- gripper: `[-1, 1]` → `[0, max_gripper_width]`（米）
- rotation 6d 保持不变（投影器负责后续换算）

---

## 11. 动作投影与 Ensemble

### 坐标投影

```python
# 单臂
action_tcp = projector.project_tcp_to_base_coord(action[..., :9], rotation_rep="rotation_6d")
action = np.concatenate([action_tcp, action[..., 9:10]], axis=-1)

# 双臂
action_left_tcp  = projector.project_tcp_to_base_coord(action[..., :9],   "left",  rotation_rep="rotation_6d")
action_right_tcp = projector.project_tcp_to_base_coord(action[..., 10:19], "right", rotation_rep="rotation_6d")
action = np.concatenate([action_left_tcp, action[..., 9:10], action_right_tcp, action[..., 19:20]], axis=-1)
```

将相机系 TCP 动作投影到机器人基坐标系（使用 rise2 extrinsics 标定）。

### Ensemble Buffer

- `ensemble_buffer.add_action(action, t)` 每次策略推理后将整段动作序列加入 buffer
- `ensemble_buffer.get_action()` 每步（含策略未推理的步）取出当前时刻的加权平均动作
- ensemble_mode 由配置决定（时序加权平均）
- 若 buffer 为空（首步前）返回 None → 跳过该步执行

---

## 12. 回退逻辑与异常统计

### 回退层次

```
infer_mask 返回 None
  └─ _safe_infer_mask → mask01=None, mask_reason=...
       └─ infer_none_policy
           ├─ "no_mask_fallback"：记录日志，继续（跳过 3D 过滤和 2D reweight）
           └─ "fail_fast"：抛出 RuntimeError，终止 rollout
```

### 统计指标（rollout 结束后打印）

| 统计项 | 含义 |
|--------|------|
| `infer_none` | `infer_mask` 正常返回 None |
| `infer_exception` | `infer_mask` 抛出异常 |
| `mask_invalid` | `_normalize_mask01` 失败（mask 格式异常） |
| `empty_cloud_skip` | 3D 过滤后点云为空，回退原始深度 |
| `reweight_fallback` | 2D 权重构建失败 |
| `points_nonfinite` | 点云含 NaN/Inf，回退原始深度 |
| `weight_nonfinite` | 2D 权重含 NaN/Inf |
| `sam2_propagate_fail` | `propagate_in_video` 异常或 arm mask 为 None |
| `sam2_invalid_color` | 输入颜色图格式错误（非 uint8 HWC-3） |

### SAM2 内部失败原因（`_sam2_runtime["last_fail_reason"]`）

| 值 | 含义 |
|----|------|
| `sam2_invalid_color` | 颜色图格式不符 |
| `sam2_cold_start_exception` | 冷启动初始化抛出异常 |
| `sam2_cold_start_aborted` | 用户按 ESC 中止标注 |
| `sam2_reset_exception` | 周期重置时抛出异常 |
| `sam2_append_exception` | 追加新帧时抛出异常 |
| `sam2_propagate_fail` | propagate 异常或未返回 arm mask |

---

## 13. 配置参考

### 最小可用配置（YAML 节选）

```yaml
mask_aware:
  enabled: true
  enable_3d_filter: true
  enable_2d_reweight: true
  mask_threshold: 0
  mask_white_is_untrusted: true
  infer_none_policy: no_mask_fallback
  empty_cloud_policy: warn_and_skip_filter

  sam2:
    enabled: true
    config_file: configs/sam2.1/sam2.1_hiera_b+.yaml
    ckpt_path: checkpoints/sam2.1_hiera_base_plus.pt  # 微调权重
    device: cuda_if_available
    arm_obj_id: 1
    gripper_obj_id: 2
    dilate_radius: 10
    reset_every_n_steps: 30
```

### 典型参数调优建议

| 参数 | 较小值效果 | 较大值效果 |
|------|-----------|-----------|
| `dilate_radius` | 更精确边界，可能遗漏部分臂区域 | 更保守屏蔽，可能误删有效点 |
| `reset_every_n_steps` | 显存占用更低，跟踪连续性稍弱 | 跟踪更连续，显存随时间增长 |

---

## 14. 关键数据类型速查

| 变量 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `colors` | `(H, W, 3)` | uint8 | RGB 颜色图，来自 agent |
| `depths` | `(H, W)` | uint16 | 深度图（单位：毫米） |
| `arm_raw` / `gripper_raw` | `(H, W)` | bool | SAM2 raw mask，未膨胀，存于 runtime |
| `final` | `(H, W)` | bool | `dilate(arm) AND NOT dilate(gripper)` |
| `raw_mask` | `(H, W)` | uint8 | `final * 255`，`infer_mask` 返回值 |
| `mask01` | `(H, W)` | float32 | 规范化后，1.0=不可信，0.0=可信 |
| `image_mask_weight` | `(1, N_patch)` | float32 | patch 可信度权重 |
| `coords` | `(N, 4)` | int32 | MinkowskiEngine 体素坐标（含 batch dim） |
| `points` | `(N, 3)` | float32 | 点云 XYZ（米） |
| `pred_raw_action` | `(num_action, action_dim)` | float32 | 归一化动作序列 |
| `inference_state["images"]` | `(T, 3, img_size, img_size)` | float32 | SAM2 帧 buffer，逐帧追加 |

---

*本文档由 Claude Code 根据 `sam2/eval_mask_final.py` 源码自动生成。*
