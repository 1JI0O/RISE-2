# SAM2 在 Eval 流程中的完整工作流程

> 文件：`sam2/eval_mask_final.py`，服务端：`sam2/sam2_mask_server.py`
> 最后更新：2026-03

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [启动初始化](#2-启动初始化)
3. [Rollout 主循环结构](#3-rollout-主循环结构)
4. [SAM2 推理详解](#4-sam2-推理详解)
   - 4.1 [冷启动：首帧交互标注](#41-冷启动首帧交互标注)
   - 4.2 [后续帧：流式追帧 + propagate](#42-后续帧流式追帧--propagate)
   - 4.3 [周期重置：显存管理](#43-周期重置显存管理)
   - 4.4 [Mask 合成：arm 膨胀减去 gripper](#44-mask-合成arm-膨胀减去-gripper)
5. [Mask 在下游的两个用途](#5-mask-在下游的两个用途)
   - 5.1 [3D 点云过滤](#51-3d-点云过滤)
   - 5.2 [2D 图像 Patch 权重重加权](#52-2d-图像-patch-权重重加权)
6. [异常处理与回退机制](#6-异常处理与回退机制)
7. [跨 Conda 环境部署：WebSocket 服务模式](#7-跨-conda-环境部署websocket-服务模式)
8. [完整时间线示例](#8-完整时间线示例)
9. [配置参数参考](#9-配置参数参考)
10. [统计计数器说明](#10-统计计数器说明)

---

## 1. 整体架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                      evaluate() 主循环                           │
│                                                                  │
│  for t in range(max_steps):                                      │
│                                                                  │
│    ① agent.get_global_observation()  ← 每步采一次 RGB-D         │
│                                                                  │
│    ② SAM2 推理（每步都执行）                                      │
│       infer_mask(color, depth, ...)                              │
│       → mask01: float32 (H,W)，1=机械臂区域，0=其余             │
│                                                                  │
│    ③ 仅推理步（t % num_inference_steps == 0）执行：              │
│       • 3D: 深度图中机械臂像素置0 → 点云只含场景                  │
│       • policy 预测动作序列 (H, action_dim)                      │
│       • 2D: image patch 权重（机械臂区域权重降低）               │
│       • EnsembleBuffer.add_action()                              │
│                                                                  │
│    ④ 每步从 buffer 取动作并下发                                   │
│       EnsembleBuffer.get_action() → agent.action()              │
└─────────────────────────────────────────────────────────────────┘
```

**关键设计**：SAM2 **每步都运行**（维持时序 memory 连续性），但 policy 推理只在 `t % num_inference_steps == 0` 时触发，使用当步的 mask。

---

## 2. 启动初始化

`evaluate()` 函数在进入 rollout 循环前依次完成：

### 2.1 配置加载

```python
config.mask_aware = _build_mask_aware_cfg(config)
config.mask_aware.sam2 = _build_sam2_cfg(config.mask_aware)
```

SAM2 配置的完整字段（含默认值）：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | SAM2 总开关 |
| `config_file` | `configs/sam2.1/sam2.1_hiera_b+.yaml` | SAM2 模型结构配置 |
| `ckpt_path` | `checkpoints/sam2.1_hiera_base_plus.pt` | 微调权重路径 |
| `device` | `cuda_if_available` | 推理设备 |
| `arm_obj_id` | `1` | VideoPredictor 中机械臂的 obj_id |
| `gripper_obj_id` | `2` | VideoPredictor 中 gripper 的 obj_id |
| `dilate_radius` | `10` | arm/gripper mask 膨胀半径（像素） |
| `reset_every_n_steps` | `100` | 每 N 步重置 inference_state |
| `remote_port` | `null` | 非空时启用 WebSocket 远程模式 |

### 2.2 SAM2 Runtime 初始化

```python
if config.mask_aware.enabled and args.type == "local" and config.mask_aware.sam2.enabled:
    _sam2_runtime = _init_sam2_runtime(config.mask_aware.sam2)
```

**本地模式**（`remote_port=null`）：加载 `SAM2VideoPredictor` 模型，创建 runtime dict：

```python
_sam2_runtime = {
    "enabled": True,
    "cfg": sam2_cfg,
    "device": torch.device("cuda"),
    "predictor": SAM2VideoPredictor,   # 微调后的模型
    "inference_state": None,           # 首帧标注后才创建
    "frame_idx": 0,                    # 当前 buffer 中的帧索引
    "last_arm_mask_raw": None,         # 上一帧 arm mask（未膨胀），用于 reset 时提示
    "last_gripper_mask_raw": None,     # 上一帧 gripper mask（未膨胀）
    "last_reset_frame_idx": 0,         # 上次重置时的 frame_idx
    "last_fail_reason": None,          # 最近一次失败原因
    "last_frame_np": None,             # 上一帧 RGB 图像（用于重置时序对齐）
}
```

**远程模式**（`remote_port=8765`）：不加载模型，只建立 WebSocket 连接：

```python
_sam2_runtime = {
    "mode": "remote",
    "enabled": True,
    "conn": <WebSocket connection>,
    "packer": msgpack_numpy.Packer(),
    "last_fail_reason": None,
}
```

---

## 3. Rollout 主循环结构

```
t=0,1,2,...,max_steps-1（默认 3000 步）
      │
      ├─ ① 采观测（每步）
      │     colors_raw, depths = agent.get_global_observation()
      │     colors_raw: uint8 (H,W,3) RGB
      │     depths:     uint16 (H,W)  毫米深度
      │
      ├─ ② SAM2 推理（每步）
      │     mask01 = _safe_infer_mask(color, depth, ...)
      │     mask01: float32 (H,W)，1.0=机械臂像素，0.0=其余
      │
      ├─ ③ Policy 推理（仅 t % num_inference_steps == 0）
      │     3D: depths_for_cloud[mask01 > 0.5] = 0
      │     点云构建 → MinkowskiEngine SparseTensor
      │     2D: image_mask_weight = 1 - pool(mask01)
      │     policy(cloud, image, image_coords, image_mask_weight)
      │     → pred_raw_action (H, action_dim)
      │     EnsembleBuffer.add_action(action, t)
      │
      └─ ④ 执行动作（每步）
            step_action = EnsembleBuffer.get_action()
            agent.action(step_action)
```

**观测频率 vs 推理频率**：

- `num_inference_steps=20`，`max_steps=3000` → 共 150 次 policy 推理
- 每次 policy 推理输出长度 `H=20` 的动作序列
- 两次 policy 推理之间，机器人按 buffer 里的动作开环执行，同时 SAM2 持续每步追帧
- SAM2 的连续追帧保证了 policy 推理时拿到的 mask 始终是当步的最新结果

---

## 4. SAM2 推理详解

入口函数：`infer_mask(color, depth, proprio, meta, agent=None)`

返回：`np.ndarray (H,W) uint8`（255=机械臂区域，0=其余），失败返回 `None`。

### 4.1 冷启动：首帧交互标注

**触发条件**：`_sam2_runtime["inference_state"] is None`，即首次调用。

**流程**：

```
1. _build_video_inference_state_from_np(predictor, color_np)
   │  手动构建 inference_state dict（复现 SAM2VideoPredictor.init_state() 逻辑）
   │  fields: images(1帧张量), num_frames=1, video_height/width,
   │          point_inputs_per_obj, mask_inputs_per_obj,
   │          cached_features, obj_id_to_idx, output_dict_per_obj, ...
   └→ predictor._get_image_feature(state, frame_idx=0, batch_size=1)
      预热视觉特征缓存

2. _cold_start_interactive(predictor, inference_state, cfg, color_np)
   │  弹出 cv2 窗口，显示当前帧
   │  用户操作：
   │    [a] 切换到 ARM 模式（obj_id=1）
   │    [g] 切换到 GRIPPER 模式（obj_id=2）
   │    左键 = 正点（前景），右键 = 负点（背景）
   │    [r] 清空当前对象的所有点
   │    Enter/Space = 确认，关闭窗口
   │    ESC = 中止
   │  每次点击后实时调用：
   │    predictor.add_new_points_or_box(
   │        inference_state, frame_idx=0, obj_id=oid,
   │        points=pts_np, labels=lbs_np, normalize_coords=True
   │    )
   │  实时渲染 mask 叠加预览（arm=蓝色，gripper=橙色）
   └→ 返回 (arm_raw_bool, gripper_raw_bool)

3. 存入 runtime：
   _sam2_runtime["inference_state"]       = state
   _sam2_runtime["frame_idx"]             = 0
   _sam2_runtime["last_arm_mask_raw"]     = arm_raw
   _sam2_runtime["last_gripper_mask_raw"] = gripper_raw
   _sam2_runtime["last_frame_np"]         = color_np.copy()

4. 返回 _compute_final_mask(arm_raw, gripper_raw, dilate_radius) * 255
```

### 4.2 后续帧：流式追帧 + propagate

每步（非 reset 步）：

```
1. _append_frame_to_video_state(predictor, inference_state, color_np)
   │  将当前帧预处理（resize → img_size，归一化 ImageNet mean/std）
   │  inference_state["images"] = torch.cat([旧buffer, 新帧], dim=0)
   │  inference_state["num_frames"] += 1
   └→ frame_idx += 1

2. predictor.propagate_in_video(
       inference_state,
       start_frame_idx=frame_idx,
       max_frame_num_to_track=1      ← 只处理当前这一帧
   )
   │  for _, obj_ids, mask_logits in ...:
   │      arm_raw     = mask_logits[arm_idx].squeeze().cpu().numpy() > 0
   │      gripper_raw = mask_logits[grp_idx].squeeze().cpu().numpy() > 0
   └→ 利用 SAM2 时序 memory（conditioning frames 的视觉特征）定位当前帧对象

3. 更新 runtime：
   last_arm_mask_raw     = arm_raw
   last_gripper_mask_raw = gripper_raw
   last_frame_np         = color_np.copy()

4. 返回 _compute_final_mask(arm_raw, gripper_raw, dilate_radius) * 255
```

**为何每步追帧而不是只在推理步时运行**：

SAM2VideoPredictor 的时序 memory 依赖连续帧输入。若跳帧（如每 20 步才喂一帧），相邻两帧之间机械臂可能已移动大段距离，time-step gap 超出模型训练分布，导致跟踪漂移或丢失。**每步都追帧确保 SAM2 看到的视频序列与真实控制频率一致**。

### 4.3 周期重置：显存管理

**触发条件**：

```python
need_reset = (
    cfg.reset_every_n_steps > 0
    and (frame_idx - last_reset_frame_idx) >= cfg.reset_every_n_steps
    and arm_raw_prev is not None
)
```

默认每 100 步触发一次。随着 `inference_state["images"]` 不断追帧，buffer 线性增长（每帧约 `3 × img_size² × 4 bytes`），100 帧后积累数百 MB VRAM，需要定期清理。

**重置逻辑（时序对齐版本）**：

```
问题根因：
  若用"当前帧（t）图像 + 上一帧（t-1）mask"作为 conditioning，
  SAM2 对 conditioning frame 直接返回输入 mask（不做视觉校正），
  导致 reset 帧输出的是 t-1 位置的 mask，与当前帧图像无关。

正确实现：

  step 1. src = last_frame_np              ← 上一帧（t-1）的图像
           state = _build_video_inference_state_from_np(predictor, src)
           add_new_mask(state, frame_idx=0, mask=arm_raw_prev)   ← t-1 mask
           → conditioning frame: t-1 图像 + t-1 mask（严格对齐）

  step 2. _append_frame_to_video_state(predictor, state, color_np)
           → 追加当前帧（t）作为 frame_idx=1

  step 3. propagate(start_frame_idx=1, max_frame_num_to_track=1)
           → SAM2 从 t-1 正常传播到 t，利用视觉特征做跨帧跟踪

  step 4. frame_idx = 1, last_reset_frame_idx = 1
           → 下次触发重置的计数从 1 开始累积
```

**重置后效果**：buffer 从 100 帧缩减为 2 帧，VRAM 立即释放，跟踪不中断。

### 4.4 Mask 合成：arm 膨胀减去 gripper

```python
def _compute_final_mask(arm_raw, gripper_raw, dilate_radius):
    arm_dilated = _dilate_mask_bool(arm_raw, dilate_radius)
    if gripper_raw is not None:
        grp_dilated = _dilate_mask_bool(gripper_raw, dilate_radius)
        return arm_dilated AND NOT grp_dilated
    return arm_dilated
```

**设计理由**：

- `arm_raw` / `gripper_raw` 是 SAM2 输出的**原始未处理 mask**，存入 runtime 供下帧跟踪使用（不膨胀存储，保证 SAM2 下帧看到的是真实对象边界）
- 膨胀在合成阶段做，扩大机械臂遮挡区域边界，防止关节边缘点云噪声进入场景点云
- gripper 区域从 arm 中减去：gripper 末端可能接触目标物体，该区域点云有用，应当**保留**而不是过滤

---

## 5. Mask 在下游的两个用途

`mask01`：float32 (H,W)，**1.0 = 机械臂像素（不可信区域），0.0 = 场景**

### 5.1 3D 点云过滤

```python
# 仅在推理步执行
if mask_enabled and config.mask_aware.enable_3d_filter and mask01 is not None:
    depths_for_cloud = depths.copy()
    depths_for_cloud[mask01 > 0.5] = 0   # 机械臂像素深度置0
```

`create_input(colors_raw, depths_for_cloud, ...)` 将深度图转为 3D 点云：

1. 利用相机内参将每个有效深度像素反投影为 3D 点
2. 深度为 0 的像素不会生成点 → 机械臂像素被过滤
3. 裁剪到 workspace bbox，体素下采样
4. 输出的点云**只含场景物体**，不含机械臂自身

若过滤后点云为空（机械臂完全遮挡场景），回退到未过滤的原始深度重建点云，并计入 `empty_cloud_skip` 统计。

### 5.2 2D 图像 Patch 权重重加权

```python
# 仅在推理步执行
if mask_enabled and config.mask_aware.enable_2d_reweight and mask01 is not None:
    image_mask_weight = _build_image_mask_weight(mask01, image_processor)
```

```python
def _build_image_mask_weight(mask01, image_processor):
    mask_tensor = resize_image(mask01, image_processor.img_size, NEAREST)
    mask_ratio  = image_processor.image_coord_pooling(mask_tensor)
    # mask_ratio: 每个 patch 内机械臂像素的占比 ∈ [0,1]
    image_mask_weight = (1.0 - mask_ratio).clamp(0.0, 1.0)
    # 机械臂占比越高 → 权重越低（接近0）
```

传入 policy：

```python
policy(cloud_data, colors_proc, image_coords, image_mask_weight=image_mask_weight)
```

policy 的图像编码器（DINOv2）对每个 patch 的注意力乘以该权重。机械臂遮挡的 patch 权重接近 0，policy 在视觉上"忽略"机械臂区域，更专注于目标物体的外观特征。

---

## 6. 异常处理与回退机制

所有 SAM2 调用通过 `_safe_infer_mask` 包装：

```python
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
```

`mask01 is None` 时的处理策略：

| 情况 | `infer_none_policy` | 行为 |
|------|---------------------|------|
| 非推理步 mask 失败 | 任意 | 仅统计，不影响 policy（该步本就不推理） |
| 推理步 mask 失败 | `no_mask_fallback` | 打印日志，本次推理不用 mask（无 3D 过滤，无 2D 权重） |
| 推理步 mask 失败 | `fail_fast` | 抛出异常，终止 rollout |

**`infer_mask` 内部失败类型**：

| `last_fail_reason` | 原因 |
|--------------------|------|
| `sam2_invalid_color` | 输入 color 不是 3 通道 uint8 |
| `sam2_cold_start_exception` | 构建初始 state 或打开 cv2 窗口时抛异常 |
| `sam2_cold_start_aborted` | 用户按 ESC 中止标注 |
| `sam2_reset_exception` | 周期重置时重建 state 失败 |
| `sam2_append_exception` | 追帧时失败 |
| `sam2_propagate_fail` | `propagate_in_video` 抛异常或返回空 obj_ids |
| `sam2_remote_exception` | WebSocket 通信失败（远程模式） |

---

## 7. 跨 Conda 环境部署：WebSocket 服务模式

`rise2` 环境与 `sam2` 环境存在依赖冲突，无法在同一进程中加载。解决方案：将 SAM2 单独作为 WebSocket 服务运行，复用项目已有的 `remote_eval/msgpack_numpy.py` 序列化层。

### 通信协议

```
rise2 进程  →  sam2 进程:
    msgpack({"color": np.ndarray uint8 HWC})

sam2 进程   →  rise2 进程:
    msgpack({"mask": np.ndarray uint8 HW | None,
             "reason": str | None})
```

- 只传 RGB 图像，不传 depth（深度处理留在 rise2 侧）
- 冷启动 cv2 窗口在 **sam2 进程**弹出，用户标注完毕后才返回第一帧 mask
- 客户端在 `conn.recv()` 处阻塞等待，rollout 暂停直到标注完成

### 启动方式

```bash
# 终端1：sam2 环境启动服务
conda run -n sam2 python sam2/sam2_mask_server.py \
    --config configs/dual_teleop_dino.yaml --port 8765
# 等待打印：[sam2-server] listening on 127.0.0.1:8765

# 终端2：rise2 环境启动 eval
conda activate rise2
python sam2/eval_mask_final.py --config configs/dual_teleop_dino.yaml ...
# 打印：[mask-aware/sam2] connected to remote server at ws://127.0.0.1:8765
```

YAML 激活（加一行即可）：

```yaml
mask_aware:
  sam2:
    enabled: true
    remote_port: 8765   # 加这行 → remote 模式；不加 → local 模式
```

---

## 8. 完整时间线示例

配置：`num_inference_steps=20`，`max_steps=3000`，`reset_every_n_steps=100`

```
t=0  ← 第一步
  ① 采观测 (colors_raw, depths)
  ② SAM2: inference_state is None → 冷启动
          弹出 cv2 窗口，用户标注 arm + gripper
          建 inference_state（1帧），frame_idx=0
          返回 mask01（来自冷启动标点）
  ③ t%20==0 → policy 推理：
          3D 过滤: depths[mask01>0.5]=0 → 点云（无机械臂）
          2D 权重: image_mask_weight = 1 - pool(mask01)
          policy → 动作序列 (20, action_dim)
          EnsembleBuffer.add_action(action, t=0)
  ④ buffer 取动作 → agent.action()

t=1..19  ← 非推理步
  ① 采观测
  ② SAM2: 追帧 → propagate(frame_idx=1..19) → mask01 更新（持续追踪）
  ③ 跳过（非推理步）
  ④ buffer 取动作 → agent.action()

t=20  ← 第二次 policy 推理
  ① 采观测
  ② SAM2: 追帧 → propagate(frame_idx=20) → mask01（已积累20帧时序信息）
  ③ t%20==0 → policy 推理（用 t=20 的 mask01，此时跟踪已稳定）
  ④ buffer 取动作

...

t=100  ← 周期重置（frame_idx=100，满足 100-0 >= 100）
  ① 采观测（color_np = t=100 的图像）
  ② SAM2: need_reset=True
          用 last_frame_np（t=99 图像）+ arm_raw_prev（t=99 mask）
          建新 state（conditioning frame = t=99，图像与 mask 严格对齐）
          追加 color_np（t=100）作为 frame_idx=1
          propagate(start=1) → t=99→t=100 正常视觉传播
          frame_idx=1，last_reset_frame_idx=1
          buffer 从100帧缩减为2帧，VRAM 释放
  ③/④ 正常继续

t=3000  ← rollout 结束
  打印统计：
  [mask-aware] summary infer_none=X infer_exception=X mask_invalid=X
               empty_cloud_skip=X reweight_fallback=X points_nonfinite=X
               weight_nonfinite=X sam2_propagate_fail=X sam2_invalid_color=X
```

---

## 9. 配置参数参考

```yaml
mask_aware:
  enabled: true
  enable_3d_filter: true           # 点云过滤开关
  enable_2d_reweight: true         # 图像权重重加权开关
  mask_threshold: 0                # mask 二值化阈值
  mask_white_is_untrusted: true    # true: mask>threshold 为机械臂（不可信）
  infer_none_policy: no_mask_fallback  # mask 推理失败时的策略
  empty_cloud_policy: warn_and_skip_filter

  sam2:
    enabled: true
    config_file: configs/sam2.1/sam2.1_hiera_b+.yaml
    ckpt_path: checkpoints/sam2.1_hiera_base_plus.pt
    device: cuda
    arm_obj_id: 1
    gripper_obj_id: 2
    dilate_radius: 10
    reset_every_n_steps: 100
    # remote_port: 8765   # 跨环境部署时取消注释
```

---

## 10. 统计计数器说明

| 字段 | 含义 | 正常期望 |
|------|------|----------|
| `infer_none` | `infer_mask` 返回 None（已知失败） | 接近 0 |
| `infer_exception` | `infer_mask` 抛出未预期异常 | 0 |
| `mask_invalid` | mask 格式/尺寸异常，`_normalize_mask01` 失败 | 0 |
| `empty_cloud_skip` | 3D 过滤后点云为空，回退到原始深度 | 偶发可接受 |
| `reweight_fallback` | 2D 权重构建失败，本次不使用 2D 权重 | 0 |
| `points_nonfinite` | 点云中出现 NaN/Inf，回退到原始深度 | 0 |
| `weight_nonfinite` | 2D 权重中出现 NaN/Inf，丢弃权重 | 0 |
| `sam2_propagate_fail` | `propagate_in_video` 失败或未返回 arm mask | 接近 0 |
| `sam2_invalid_color` | 输入 color 图像格式不合法 | 0 |
