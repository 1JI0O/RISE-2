# infer_mask URDF 渲染器实现与 Review 修复复盘

> 分支：`mask-aware`
> 涉及 commit：本次实现 + review 修复（同一开发周期内）
> 关联文档：[`mask_aware_inference_maintainer_notes.md`](mask_aware_inference_maintainer_notes.md)

---

## 一、本次实现了什么

### 背景

`eval.py` 已有完整的 mask-aware 推理骨架（3D 过滤、2D 降权、异常统计），但
[`infer_mask()`](../../eval.py#L139) 是占位函数，始终返回 `None`，整条管线实际无效。

### 实现目标

在推理期通过 **URDF + Open3D OffscreenRenderer** 渲染机械臂当前姿态，得到像素级 mask，
使管线真正可用。姿态读取入口保留为可替换的占位接口。

### 新增文件

| 文件 | 内容 |
|------|------|
| [`mask/__init__.py`](../../mask/__init__.py) | 空文件，使 `mask` 成为 Python 包 |
| [`mask/renderer.py`](../../mask/renderer.py) | 自包含的 URDF 渲染器（含 FK 辅助、常量、`SeparateRiseRobotRenderer`） |

### 修改文件

[`eval.py`](../../eval.py)（四处）：

| 位置 | 内容 |
|------|------|
| `import os` 新增 | 路径拼接需要 |
| `_build_mask_aware_cfg` | 新增 `urdf_left`/`urdf_right` 默认键（默认 `None`） |
| `_arm_renderer` + `get_arm_joints()` | 模块级 renderer 引用 + 关节角读取占位接口 |
| `infer_mask()` | 实现：调用 `get_arm_joints` → `update_joints` → `render_mask()` |
| `evaluate()` 初始化块 | mask-aware 启用时构建 `SeparateRiseRobotRenderer` |

[`configs/dual_teleop_dino.yaml`](../../configs/dual_teleop_dino.yaml)：
- `empty_cloud_policy: warn_and_skip` → `warn_and_skip_filter`（bug fix，见下文 P1b）

---

## 二、mask/renderer.py 设计说明

### 为什么不直接 import airexo

airexo 是 git submodule，其路径约定、内部依赖不由本仓库控制。
将渲染逻辑内联到 `mask/renderer.py` 使 RISE2 codebase 自包含，便于独立维护。

### 核心类：SeparateRiseRobotRenderer

适配自 airexo 的 `SeparateRobotRenderer`（左右臂各一份 URDF），主要接口：

```python
renderer.update_joints(left_joint, right_joint)
# left_joint / right_joint: np.ndarray (8,)
# [joint1..7 in radians, gripper_width_in_meters]

mask = renderer.render_mask()
# 返回 uint8 (H, W)，255 = 机械臂像素，0 = 背景
```

### kinpy chain 缓存（P2 修复后的设计）

kinpy chain 表示机器人运动学**结构**（拓扑、连杆），来自静态 URDF，永远不变。
`chain.forward_kinematics(joint_states)` 以当前关节角为参数，返回本帧的各连杆变换矩阵——
chain 本身无状态，可在任意关节角下重复调用。

因此 chain 只在 `__init__` 构建一次（`kp.build_chain_from_urdf`），
`_update_geometry` 每帧仅调用 `chain.forward_kinematics(joint_states)`。

原始 airexo 代码中每帧都重建 chain，属于性能冗余，本次在复制时已修正。

---

## 三、Review 发现的 4 个 Bug 及修复

### P0（功能默认失效）— projector 属性不存在

**问题**

初始实现写了：
```python
cam_to_left_base = projector.cam_to_left_base   # ❌ 不存在
```

`DualArmProjector` 实际结构（见 [`dataset/projector.py`](../../dataset/projector.py#L92)）：
```python
self.projector_left  = ProjectorBase(np.linalg.inv(calib["camera_to_robot_left"][cam]))
self.projector_right = ProjectorBase(np.linalg.inv(calib["camera_to_robot_right"][cam]))
# ProjectorBase.camera_pose 存的是 cam→base 的变换矩阵
```

属性访问异常会被 `except Exception` 吞掉，`_arm_renderer = None`，
随后每帧 `infer_mask` 走 `if _arm_renderer is None: return None`，
**功能静默失效且没有任何报错**。

**修复**（[`eval.py:413`](../../eval.py#L413)）
```python
# 修复前
cam_to_left_base  = projector.cam_to_left_base
cam_to_right_base = projector.cam_to_right_base

# 修复后
cam_to_left_base  = projector.projector_left.camera_pose
cam_to_right_base = projector.projector_right.camera_pose
```

---

### P1a（配置键被丢弃）— urdf 路径配置永远不生效

**问题**

`_build_mask_aware_cfg` 的合并逻辑：
```python
for key in default_cfg:          # 只遍历已知 key
    if key in raw_cfg ...
        merged_cfg[key] = ...
```

`urdf_left`/`urdf_right` 不在 `default_cfg` 里，yaml 里写的值会被丢弃。
后续 `getattr(config.mask_aware, "urdf_left", ...)` 因属性不存在而总是回落默认路径。

**修复**（[`eval.py:60`](../../eval.py#L60)）

在 `default_cfg` 中加入这两个 key：
```python
"urdf_left":  None,
"urdf_right": None,
```

`evaluate()` 中的读取也简化为：
```python
_urdf_left  = config.mask_aware.urdf_left  or os.path.join(...)
_urdf_right = config.mask_aware.urdf_right or os.path.join(...)
```

---

### P1b（配置值与合法集合不一致）— empty_cloud_policy 被强制覆盖

**问题**

[`configs/dual_teleop_dino.yaml`](../../configs/dual_teleop_dino.yaml)：
```yaml
empty_cloud_policy: warn_and_skip      # ❌ 非法值
```

`_build_mask_aware_cfg` 中合法集合是：
```python
valid_empty_cloud_policy = {"warn_and_skip_filter", "fail_fast"}
```

`warn_and_skip` 不在其中，被强制改为 `warn_and_skip_filter` 并打印警告，
用户以为配置了某个值，实际上永远走 `warn_and_skip_filter`。

**修复**（[`configs/dual_teleop_dino.yaml:104`](../../configs/dual_teleop_dino.yaml#L104)）
```yaml
empty_cloud_policy: warn_and_skip_filter   # ✓
```

---

### P2（性能风险）— 每帧重建 URDF kinpy chain

**问题**

初始实现的 `_update_geometry` 调用 `_forward_kinematic_single()`，
该函数内部每次都执行：
```python
model_chain = kp.build_chain_from_urdf(open(urdf_file, "rb").read())
```

每个推理步（每 `num_inference_steps` 帧）都会重新解析 URDF 文件、
重建完整的运动学拓扑结构，这是纯粹的冗余开销。

**修复**（[`mask/renderer.py`](../../mask/renderer.py)）

在 `__init__` 中一次性构建并缓存：
```python
self.chain_left  = kp.build_chain_from_urdf(open(urdf_left,  "rb").read())
self.chain_right = kp.build_chain_from_urdf(open(urdf_right, "rb").read())
self.visuals_map_left  = self.chain_left.visuals_map()
self.visuals_map_right = self.chain_right.visuals_map()
```

`_update_geometry` 每帧只调用：
```python
cur_transforms_left  = self.chain_left.forward_kinematics(left_states)
cur_transforms_right = self.chain_right.forward_kinematics(right_states)
```

---

## 四、修复后各组件职责总结

```
eval.py
  ├── _build_mask_aware_cfg()    配置归一（含 urdf_left/right 键）
  ├── get_arm_joints()           关节角读取接口【唯一需接入硬件的函数】
  ├── infer_mask()               调 get_arm_joints → 渲染 → 返回 mask
  └── evaluate()
        └── 初始化 SeparateRiseRobotRenderer
              cam_to_*_base = projector.projector_{left,right}.camera_pose  ← P0 fix

mask/renderer.py
  └── SeparateRiseRobotRenderer
        ├── __init__: 构建 chain_left/chain_right（一次）               ← P2 fix
        ├── _update_geometry: chain.forward_kinematics(joint_states)
        ├── render_depth → render_mask
        └── update_joints → _update_geometry

configs/dual_teleop_dino.yaml
  └── empty_cloud_policy: warn_and_skip_filter                           ← P1b fix
```

---

## 五、接入真实机械臂的唯一入口

[`eval.py:123`](../../eval.py#L123) 的 `get_arm_joints()`，将 `return None` 替换为：

```python
def get_arm_joints(meta=None):
    left_q  = agent.left_robot.get_joint_pos()   # shape (7,), rad
    left_w  = agent.left_gripper.get_width()     # float, meters
    right_q = agent.right_robot.get_joint_pos()
    right_w = agent.right_gripper.get_width()
    return (np.append(left_q, left_w), np.append(right_q, right_w))
```

接入后 `infer_mask` 会自动产生真实 mask，无需修改其他任何地方。

---

## 六、验证步骤

1. **renderer 未初始化路径**：`mask_aware.enabled=false` 或 `type=remote`，
   `_arm_renderer=None`，`infer_mask` 返回 `None`，fallback 日志计数增加，eval 正常运行。

2. **renderer 初始化验证**：`mask_aware.enabled=true, type=local`，
   日志出现 `[mask-aware] renderer initialized`，无异常。

3. **零姿态 mask 验证**：临时在 `get_arm_joints` 返回全零关节角，
   `render_mask()` 应返回非全黑的 uint8 图像（臂出现在画面边缘/外侧属正常）。

4. **配置项验证**：yaml 中写 `urdf_left: /custom/path/left_robot.urdf`，
   日志应显示该路径而非默认路径。
