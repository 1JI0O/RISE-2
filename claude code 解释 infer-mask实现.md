# infer_mask 实现说明

> 分支：`mask-aware`
> 涉及文件：`eval.py`、`mask/renderer.py`、`mask/__init__.py`

---

## 背景

RISE2 的 mask-aware 管线在 `eval.py` 中已有完整骨架：

- **3D 过滤**：将 mask 区域的深度置零，避免生成含噪点云
- **2D 重加权**：按 patch 级 mask 占比降低对 inpainting 区域的特征信任度
- **加权插值**：`SpatialAligner` 在特征融合时乘以 mask 置信度

但核心函数 `infer_mask()` 原本只是一行 `return None`，导致整条管线在推理期实际上不产生 mask。

本次改动填充了这个缺口：**在推理时用 URDF + Open3D OffscreenRenderer 渲染机械臂当前姿态，得到像素级 mask。**

---

## 新增 / 修改的文件

```
mask/
  __init__.py        ← 新建（空，使 mask 成为 Python 包）
  renderer.py        ← 新建（自包含的 URDF 渲染器）
eval.py              ← 修改（实现 infer_mask，加 get_arm_joints，初始化 renderer）
```

---

## mask/renderer.py

### 设计原则

- **自包含**：不 import airexo 子模块，把需要的常量和 FK 函数内联进来，方便独立维护。
- **参考来源**：逻辑直接适配自 `airexo/airexo/helpers/renderer.py`（`SeparateRobotRenderer`）和 `urdf_robot.py`。

### 内容结构

```
常量
  O3D_RENDER_TRANSFORMATION           # Open3D 渲染坐标翻转
  ROBOT_PREDEFINED_TRANSFORMATION     # URDF base → 物理 base（统一双臂）
  LEFT_ROBOT_PREDEFINED_TRANSFORMATION
  RIGHT_ROBOT_PREDEFINED_TRANSFORMATION

FK 辅助函数（Flexiv Rizon4 + Robotiq 2F-85）
  _calc_robotiq_state(open_length)
  _convert_robotiq_gripper_joint_state(gripper_width)
  _convert_joint_states_single(joint, num_robot_joints, is_rad)
  _forward_kinematic_single(joint, num_robot_joints, is_rad, urdf_file, with_visuals_map)

class SeparateRiseRobotRenderer
  __init__(cam_to_left_base, cam_to_right_base, intrinsic, width, height,
           num_robot_joints, urdf_left, urdf_right, near_plane, far_plane)
  update_joints(left_joint, right_joint)  # 更新姿态并刷新 Open3D 场景
  render_depth() -> np.ndarray            # float32，z_in_view_space，inf=背景
  render_mask(depth=None) -> np.ndarray   # uint8，255=机械臂像素，0=背景
  _update_geometry(joints)               # 内部：FK → mesh transform → 重建场景
```

### 关节角格式

`left_joint` / `right_joint`：`np.ndarray shape (num_robot_joints + 1,)`

| 索引 | 含义 |
|------|------|
| 0–6  | Flexiv 7 个关节角（弧度） |
| 7    | Robotiq 夹爪宽度（米，范围 0–0.085） |

### mask 含义

渲染后：`255` = 机械臂占据的像素，`0` = 背景。

与 `eval.py` 中 `mask_white_is_untrusted: true` 配合：255 的区域会被认为"不可信"，触发 3D 过滤和 2D 降权。

---

## eval.py 改动

### 1. 模块级变量

```python
_arm_renderer = None   # 由 evaluate() 在启动时赋值
```

### 2. get_arm_joints（接口占位）

```python
def get_arm_joints(meta=None):
    # 返回 (left_joint, right_joint)，各为 shape (8,) 的 ndarray
    # 实际部署时接入机械臂 SDK
    return None
```

**这是唯一需要在真实部署时填写的函数。** 示例：

```python
def get_arm_joints(meta=None):
    left_q  = agent.left_robot.get_joint_pos()    # (7,) rad
    left_w  = agent.left_gripper.get_width()      # float, meters
    right_q = agent.right_robot.get_joint_pos()
    right_w = agent.right_gripper.get_width()
    return (np.append(left_q, left_w), np.append(right_q, right_w))
```

### 3. infer_mask（实现）

```python
def infer_mask(color, depth, proprio, meta):
    if _arm_renderer is None:
        return None
    joints = get_arm_joints(meta)
    if joints is None:
        return None
    left_joint, right_joint = joints
    _arm_renderer.update_joints(left_joint, right_joint)
    return _arm_renderer.render_mask()   # uint8 (H, W)
```

### 4. evaluate() 中的 renderer 初始化

在 `_log_mask_aware_summary()` 之后，仅在 `mask_aware.enabled=true` 且 `type=local` 时初始化：

```python
global _arm_renderer
_arm_renderer = None
if config.mask_aware.enabled and args.type == "local":
    _arm_renderer = SeparateRiseRobotRenderer(
        cam_to_left_base  = projector.cam_to_left_base,
        cam_to_right_base = projector.cam_to_right_base,
        intrinsic         = fake_intrinsics,  # 真实部署换为 agent.intrinsics
        width=1280, height=720,
        num_robot_joints  = 7,
        urdf_left         = ...,   # 默认 airexo/airexo/urdf_models/robot/left_robot.urdf
        urdf_right        = ...,
    )
```

初始化失败时自动 fallback（`_arm_renderer = None`），不中断 eval。

---

## 配置项（dual_teleop_dino.yaml）

`mask_aware` 块新增两个**可选**字段：

```yaml
mask_aware:
  enabled: true
  # ... 已有字段不变 ...
  urdf_left:  airexo/airexo/urdf_models/robot/left_robot.urdf   # 可选，有默认值
  urdf_right: airexo/airexo/urdf_models/robot/right_robot.urdf  # 可选，有默认值
```

未配置时使用上述默认路径。

---

## 数据流

```
每个推理步（每 num_inference_steps 帧触发一次）

agent.get_global_observation()
  → colors, depths

get_arm_joints(meta)               ← 【唯一需要接入硬件的接口】
  → (left_joint, right_joint)

SeparateRiseRobotRenderer.update_joints()
  → FK 计算 → Open3D OffscreenRenderer 更新 mesh

render_mask()
  → uint8 mask (H, W)，255=臂

_normalize_mask01()
  → float32 mask01 (H, W)，1.0=臂（不可信）

3D 过滤：depths_for_cloud[mask01 > 0.5] = 0
2D 重加权：image_mask_weight = 1 - pool(mask01)

policy(cloud_data, colors, image_coords, image_mask_weight=...)
  → pred_raw_action
```

---

## 故障排查

| 现象 | 原因 | 处理 |
|------|------|------|
| `infer_none` 计数持续增加 | `get_arm_joints` 返回 None | 正常（占位状态），接入 SDK 后消失 |
| `[mask-aware] renderer init failed` | URDF 路径不存在 / kinpy 未安装 | 检查路径；`pip install kinpy` |
| mask 全黑（全为 0） | 关节角全零时臂在画面外 | 正常；等待真实关节角接入 |
| `mask_aware.enabled` 无效 | 配置文件路径错误或字段拼写错 | 检查 yaml 中 `mask_aware.enabled: true` |

---

## 依赖

| 包 | 用途 | 已有？ |
|----|------|--------|
| `open3d` | OffscreenRenderer | ✓（airexo 已用） |
| `kinpy` | URDF FK | ✓（airexo 已用） |
| `numpy` | 数值计算 | ✓ |
