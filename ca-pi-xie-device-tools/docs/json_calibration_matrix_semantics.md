# RISE2 中由 JSON 合成的标定矩阵语义分析

本文基于当前仓库的真实消费链路，对两个 JSON 文件中的矩阵语义进行反推分析。结论不是单靠字段名猜测，而是依据下游训练、投影、推理回投以及 Airexo 原始标定链条的矩阵乘法方向得出。

## 结论摘要

对于由 [`build_rise2_calib_from_json.py`](../build_rise2_calib_from_json.py) 读取的左右两个 JSON：

- 左 JSON 中的 [`pose_in_link`](../build_rise2_calib_from_json.py:114) 表示 `T_cam_left_base`
- 右 JSON 中的 [`pose_in_link`](../build_rise2_calib_from_json.py:114) 表示 `T_cam_right_base`

更直白地说：

- 左 JSON 的矩阵是 **全局相机坐标系到左机械臂 base 坐标系的刚体变换**
- 右 JSON 的矩阵是 **全局相机坐标系到右机械臂 base 坐标系的刚体变换**

为了避免自然语言歧义，后文统一写成：

- `T_cam_left_base`
- `T_cam_right_base`

其使用语义是：**把 base 坐标系中的点/位姿变换到 camera 坐标系中**。

同时，7 维 [`pose_in_link`](../build_rise2_calib_from_json.py:114) 的格式为：

- 前 3 维：`[x, y, z]`
- 后 4 维：`[qw, qx, qy, qz]`

这一格式在 [`pose_wxyz_to_mat()`](../build_rise2_calib_from_json.py:51) 中被显式规定。

## 1. 为什么可以从下游使用方式反推出矩阵语义

在 [`DualArmProjector`](../../dataset/projector.py:70) 中，左右两个矩阵被直接作为：

- [`camera_to_robot_left`](../../dataset/projector.py:94)
- [`camera_to_robot_right`](../../dataset/projector.py:95)

也就是：

- 左臂 projector 的 [`camera_pose`](../../dataset/projector.py:8) = `T_cam_left_base`
- 右臂 projector 的 [`camera_pose`](../../dataset/projector.py:8) = `T_cam_right_base`

### 1.1 训练时：base -> camera

训练数据在 [`RealWorldDataset.__getitem__()`](../../dataset/realworld.py:567) 中，会把 lowdim 里的 TCP 从机械臂 base 坐标系投到相机坐标系。

具体调用：

- 左臂调用 [`project_tcp_to_camera_coord(..., robot = "left")`](../../dataset/realworld.py:574)
- 右臂调用 [`project_tcp_to_camera_coord(..., robot = "right")`](../../dataset/realworld.py:579)

真正的核心乘法在 [`ProjectorBase.project_tcp_to_camera_coord()`](../../dataset/projector.py:10)：

```python
np.linalg.inv(self.camera_pose) @ xyz_rot_to_mat(tcp)
```

见 [`dataset/projector.py`](../../dataset/projector.py:12)。

若令：

- `self.camera_pose = T_cam_base`
- `xyz_rot_to_mat(tcp) = T_base_tcp`

那么就得到：

```python
T_cam_tcp = inv(T_cam_base) @ T_base_tcp
```

这正是把 base 下的 TCP 位姿转到 camera 坐标系中的标准形式。

因此，这里要求 [`camera_pose`](../../dataset/projector.py:8) 必须是 `T_cam_base`，而不能是 `T_base_cam`。

### 1.2 推理时：camera -> base

推理评估阶段会把模型输出的相机系动作再投回 base 系，调用位置见：

- [`eval_rise2_dev_dataset.py`](../../eval_rise2_dev_dataset.py:727)
- [`eval_rise2_dev_dataset.py`](../../eval_rise2_dev_dataset.py:728)

真正计算在 [`ProjectorBase.project_tcp_to_base_coord()`](../../dataset/projector.py:23)：

```python
self.camera_pose @ xyz_rot_to_mat(tcp)
```

见 [`dataset/projector.py`](../../dataset/projector.py:25)。

若模型输出是 `T_cam_tcp`，则：

```python
T_base_tcp = T_cam_base @ T_cam_tcp
```

这再次要求 [`self.camera_pose`](../../dataset/projector.py:8) 的真实语义是 `T_cam_base`。

### 1.3 与你的验证结果一致

你已经验证：由 [`build_rise2_calib_from_json.py`](../build_rise2_calib_from_json.py) 生成的标定文件用于训练后，vis 出来的轨迹合理。

这意味着：

- 训练时从 base 投到 camera 的方向是对的
- 推理时从 camera 投回 base 的方向也是对的
- 因而 JSON 中矩阵的方向解释必须与上述乘法链一致

如果 JSON 实际是 `T_base_cam`，那么当前实现会多做一次逆，通常会导致：

- 轨迹整体翻转
- 位姿方向明显错误
- 相机视角下左右臂运动趋势不合理

既然这些没有发生，那么“`pose_in_link` = `T_cam_left_base` / `T_cam_right_base`”就是最符合证据的解释。

## 2. `build_rise2_calib_from_json.py` 实际做了什么

在 [`build_rise2_calib()`](../build_rise2_calib_from_json.py:102) 中，脚本直接把左右 JSON 的 [`pose_in_link`](../build_rise2_calib_from_json.py:114) 填入目标标定字典：

- 左侧写到 [`camera_to_robot_left`](../build_rise2_calib_from_json.py:132)
- 右侧写到 [`camera_to_robot_right`](../build_rise2_calib_from_json.py:135)

对应代码：

```python
"camera_to_robot_left": {
    serial: pose_wxyz_to_mat(left_data["pose_in_link"]).astype(np.float32),
},
"camera_to_robot_right": {
    serial: pose_wxyz_to_mat(right_data["pose_in_link"]).astype(np.float32),
},
```

见：

- [`build_rise2_calib_from_json.py:132`](../build_rise2_calib_from_json.py:132)
- [`build_rise2_calib_from_json.py:135`](../build_rise2_calib_from_json.py:135)

这说明该脚本并没有尝试恢复 Airexo 原始标定的中间链条，而是直接写入 RISE2 下游要消费的最终矩阵。

脚本头部注释也明确说明：

- [`本脚本不会尝试还原 Airexo 原始标定链条`](../build_rise2_calib_from_json.py:12)
- [`本脚本直接生成下游训练/读取所需的 RISE2 风格标定`](../build_rise2_calib_from_json.py:13)

因此，从这个脚本角度，JSON 中的矩阵不是 marker 中间量，而是已经落在最终 `camera -> robot base` 语义上的结果。

## 3. 7 维 pose 的每一部分代表什么

### 3.1 平移部分 `[x, y, z]`

在 [`pose_wxyz_to_mat()`](../build_rise2_calib_from_json.py:51) 中：

- 前 3 维被放入 4x4 矩阵右上角，见 [`mat[:3, 3] = xyz`](../build_rise2_calib_from_json.py:60)

所以对于 `T_cam_base`：

- 平移向量表示 **base 原点在 camera 坐标系中的位置**

写成公式：

```text
p_cam = R_cam_base * p_base + t_cam_base
```

当 `p_base = 0` 时：

```text
p_cam = t_cam_base
```

因此：

- 左 JSON 的 `[x, y, z]` = 左臂 base 原点在全局相机坐标系中的位置
- 右 JSON 的 `[x, y, z]` = 右臂 base 原点在全局相机坐标系中的位置

### 3.2 四元数部分 `[qw, qx, qy, qz]`

在 [`pose_wxyz_to_mat()`](../build_rise2_calib_from_json.py:57) 中，后四维作为 `quat_wxyz`，再由：

- [`quat_wxyz_to_mat3()`](../build_rise2_calib_from_json.py:26)

转为旋转矩阵，见 [`mat[:3, :3] = quat_wxyz_to_mat3(quat_wxyz)`](../build_rise2_calib_from_json.py:59)。

它表示的是 `R_cam_base`，也就是：

- 一个在 base 坐标系中表示的向量，乘上这个旋转后，会变成在 camera 坐标系中的表示

因此，这个四元数表达的是：

- **机械臂 base 姿态相对于 global camera 的旋转关系**

## 4. 与 Airexo 原始标定链条的关系

Airexo 原始 `type='robot'` 标定不是直接保存 `camera_to_robot_left/right`，而是通过 `extrinsics + inhand + tcp_pose` 推导它。

这在 [`CalibrationInfo.get_camera_to_robot_left_base()`](../../airexo/airexo/calibration/calib_info.py:98) 中非常清楚：

```python
left_cam_to_base = self.extrinsics[serial] \
    @ np.linalg.inv(self.extrinsics[self.camera_serial_inhand_left]) \
    @ ROBOT_LEFT_CAM_TO_TCP \
    @ np.linalg.inv(xyz_rot_to_mat(self.robot_left["tcp_pose"], rotation_rep = "quaternion"))
```

见 [`airexo/airexo/calibration/calib_info.py:101`](../../airexo/airexo/calibration/calib_info.py:101)。

右臂同理，见 [`airexo/airexo/calibration/calib_info.py:109`](../../airexo/airexo/calibration/calib_info.py:109)。

同样的推导在简化适配器中也出现：

- [`_camera_to_robot_left_base()`](../../airexo/airexo/adaptor/calib_transform_tool.py:79)
- [`_camera_to_robot_right_base()`](../../airexo/airexo/adaptor/calib_transform_tool.py:97)

其中变量命名已经说明很多问题：

- [`cam_to_global_marker`](../../airexo/airexo/adaptor/calib_transform_tool.py:84)
- [`inhand_to_global_marker`](../../airexo/airexo/adaptor/calib_transform_tool.py:85)

这表明原始 [`extrinsics`](../../airexo/airexo/calibration/calib_info.py:41) 更像是：

- 各个相机到同一个全局 marker 的位姿

然后再借助：

- 手眼相机到 TCP 的固定变换
- 当前时刻 TCP pose

最终推导出 `camera -> robot base`。

而你的 JSON 合成脚本并没有保存这条中间链，而是直接把最终结果写到了 RISE2 训练真正消费的：

- [`camera_to_robot_left`](../../dataset/projector.py:94)
- [`camera_to_robot_right`](../../dataset/projector.py:95)

## 5. 仓库中另一份脚本给出的独立旁证

在 [`build_fake_airexo_robot_calib.py`](../build_fake_airexo_robot_calib.py) 中，作者已经写了非常直接的说明：

- [`这两个 JSON 就是 camera -> left_real_base / camera -> right_real_base`](../build_fake_airexo_robot_calib.py:70)

对应变量也是：

- [`cam_to_left_real_base`](../build_fake_airexo_robot_calib.py:71)
- [`cam_to_right_real_base`](../build_fake_airexo_robot_calib.py:72)

虽然这不是下游消费代码本身，但它与前面根据 [`dataset/projector.py`](../../dataset/projector.py) 反推出的结论完全一致，因此可作为一条独立旁证。

## 6. 为什么字段名 `pose_in_link` 不能单独决定语义

单看 [`pose_in_link`](../build_rise2_calib_from_json.py:114) 这个名字，并不能唯一决定方向。

因为它只说明：

- 有一个 pose 是在某个 link 坐标系中表达的

但不能自动判断：

- 是 camera pose expressed in link
- 还是 link pose expressed in camera

因此，真正可靠的方式只能是看下游矩阵怎么参与乘法。

而当前仓库的训练/推理链已经清楚表明：

- 它们必须被当作 `T_cam_left_base` / `T_cam_right_base` 使用

## 7. 为什么自然语言“相机相对于机械臂”容易误导

[`build_rise2_calib_from_json.py`](../build_rise2_calib_from_json.py) 顶部注释写道：

- [`左右两个 JSON 已经分别给出“global 相机相对于左/右机械臂”的外参`](../build_rise2_calib_from_json.py:6)

这句中文在工程实践里常有歧义。因为“X 相对于 Y”可能被不同人解释为：

- `T_Y_X`
- 或 `T_X_Y`

但矩阵乘法没有歧义。

在本仓库中，真正被要求成立的是：

```text
T_cam_tcp = inv(T_cam_base) @ T_base_tcp
T_base_tcp = T_cam_base @ T_cam_tcp
```

所以应以矩阵乘法链为准，而不是以自然语言表述为准。

## 8. 最终结论

基于当前仓库代码证据，可以给出如下稳定结论：

### 左 JSON

- [`pose_in_link`](../build_rise2_calib_from_json.py:114) = `T_cam_left_base`
- 表示左机械臂 base 相对于全局相机的 6DoF 位姿
- 其逆可用于把左臂 base 下的 TCP 投到相机系中，见 [`ProjectorBase.project_tcp_to_camera_coord()`](../../dataset/projector.py:10)

### 右 JSON

- [`pose_in_link`](../build_rise2_calib_from_json.py:114) = `T_cam_right_base`
- 表示右机械臂 base 相对于全局相机的 6DoF 位姿
- 其逆可用于把右臂 base 下的 TCP 投到相机系中，见 [`ProjectorBase.project_tcp_to_camera_coord()`](../../dataset/projector.py:10)

### 向量内部语义

- 前 3 维：base 原点在相机坐标系中的位置
- 后 4 维：base 坐标轴相对于相机坐标轴的旋转，四元数顺序是 [`qw,qx,qy,qz`](../build_rise2_calib_from_json.py:54)

## 9. 推荐命名

为了避免未来混淆，建议在后续分析文档和脚本中尽量避免使用“某某相对于某某”的自然语言，而改用：

- `T_cam_left_base`
- `T_cam_right_base`
- `T_base_tcp`
- `T_cam_tcp`

这样可以直接从下标读出“输入坐标系”和“输出坐标系”，减少方向误判。

## 10. 简图

```text
左臂训练时:   T_cam_tcp_left  = inv(T_cam_left_base)  @ T_left_base_tcp
右臂训练时:   T_cam_tcp_right = inv(T_cam_right_base) @ T_right_base_tcp

左臂推理时:   T_left_base_tcp  = T_cam_left_base  @ T_cam_tcp_left
右臂推理时:   T_right_base_tcp = T_cam_right_base @ T_cam_tcp_right
```

这正是：

- [`dataset/projector.py`](../../dataset/projector.py)
- [`dataset/realworld.py`](../../dataset/realworld.py)
- [`eval_rise2_dev_dataset.py`](../../eval_rise2_dev_dataset.py)

共同实现的坐标变换链。