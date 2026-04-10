# 从现有 JSON / CSV 构造 Airexo 风格 `type='robot'` 标定文件

这份说明对应实现脚本：

- [`build_airexo_robot_calib_equivalent.py`](../build_airexo_robot_calib_equivalent.py)

目标不是唯一恢复历史标定软件内部的“原始真值”，而是构造一份**结构与语义兼容 [`CalibrationInfo`](../../airexo/airexo/calibration/calib_info.py:14) 的 `type='robot'` 标定文件**。

---

## 1. 原始 Airexo 方程

Airexo 对 `type='robot'` 真正依赖的是下面两条方程：

左臂：

```text
T_cam_left_base_real
= extrinsics[global]
  @ inv(extrinsics[inhand_left])
  @ ROBOT_LEFT_CAM_TO_TCP
  @ inv(tcp_pose_left)
```

见 [`airexo/airexo/calibration/calib_info.py:101`](../../airexo/airexo/calibration/calib_info.py:101)。

右臂：

```text
T_cam_right_base_real
= extrinsics[global]
  @ inv(extrinsics[inhand_right])
  @ ROBOT_RIGHT_CAM_TO_TCP
  @ inv(tcp_pose_right)
```

见 [`airexo/airexo/calibration/calib_info.py:109`](../../airexo/airexo/calibration/calib_info.py:109)。

之后 [`CalibrationInfo.get_camera_to_base()`](../../airexo/airexo/calibration/calib_info.py:81) 还会继续：

1. 把左右 `individual real base` 提升到 shared real base
2. 做平均
3. 再右乘 [`inv(ROBOT_PREDEFINED_TRANSFORMATION)`](../../airexo/airexo/calibration/calib_info.py:95)

---

## 2. 为什么现在要考虑“整个方向可能反了”

你最新给出的现象是：

- **base 出现在相机前面，但它其实应该在后面**

这类错误比“偏 90°”更基础，通常意味着：

- 我们把 [`result.json`](result.json) 当成了 `camera -> base`
- 但它实际可能是 `base -> camera`

也就是说，问题不只是 fixed transform 是否多/少一层，而可能是**整个 4x4 目标矩阵方向反了**。

因此脚本现在新增了“先对 JSON 目标取逆，再做反解”的候选模式。

---

## 3. 当前支持的 `target_mode`

### 3.1 `real_base`

- 直接把 JSON 当成 `camera -> real_base`

### 3.2 `predefined_base`

- 把 JSON 当成 `camera -> predefined_base`
- 再乘 [`ROBOT_PREDEFINED_TRANSFORMATION`](../../airexo/airexo/helpers/constants.py:91) 映回 real-base 层

### 3.3 `predefined_base_rz_pos90`

- 在 `predefined_base` 基础上，再补一个绕 Z 轴 `+90°`

### 3.4 `predefined_base_rz_neg90`

- 在 `predefined_base` 基础上，再补一个绕 Z 轴 `-90°`

### 3.5 `inverse_real_base`

- **先把 JSON 目标整体求逆**
- 再把它解释成 `camera -> real_base`

也就是测试：

```text
T_target_real = inv(T_json)
```

### 3.6 `inverse_predefined_base`

- **先把 JSON 目标整体求逆**
- 再把它当作 `camera -> predefined_base`
- 再乘 [`ROBOT_PREDEFINED_TRANSFORMATION`](../../airexo/airexo/helpers/constants.py:91) 映回 real-base 层

也就是测试：

```text
T_target_real = inv(T_json) @ ROBOT_PREDEFINED_TRANSFORMATION
```

---

## 4. 当前脚本使用哪些输入

### 必需输入

- 左臂 [`result.json`](result.json)
- 右臂 [`result.json`](result.json)
- 左臂 [`pose_b2e.csv`](pose_b2e.csv)
- 右臂 [`pose_b2e.csv`](pose_b2e.csv)
- [`intrinsics.npy`](intrinsics.npy)

### 当前版本未直接使用

- [`pose_h2w.csv`](pose_h2w.csv)

原因是当前方案走的是“等效构造”而不是“重做 hand-eye 求解”。

---

## 5. 当前脚本做了什么

[`build_airexo_robot_calib_equivalent.py`](../build_airexo_robot_calib_equivalent.py) 的处理流程是：

1. 从左右 [`result.json`](result.json) 读取 `pose_in_link`
2. 根据 [`--target-mode`](../build_airexo_robot_calib_equivalent.py) 决定：
   - 是否先整体求逆
   - 是否补 [`ROBOT_PREDEFINED_TRANSFORMATION`](../../airexo/airexo/helpers/constants.py:91)
   - 是否额外补 Z 轴 `±90°`
3. 把目标矩阵映射到 real-base 层
4. 从左右 [`pose_b2e.csv`](pose_b2e.csv) 各取一行作为 `tcp_pose`
5. 使用 Airexo 常量：
   - [`ROBOT_LEFT_CAM_TO_TCP`](../../airexo/airexo/helpers/constants.py:35)
   - [`ROBOT_RIGHT_CAM_TO_TCP`](../../airexo/airexo/helpers/constants.py:36)
6. 反解得到 `extrinsics[inhand_left/right]`
7. 构造 `type='robot'` 字典并保存为 `.npy`

---

## 6. 现在最值得试的命令

既然你已经观察到“base 应在后面，却跑到前面”，当前最值得优先试的是：

### 6.1 `inverse_real_base`

```bash
python ca-pi-xie-device-tools/build_airexo_robot_calib_equivalent.py \
  --left-json /data/haoxiang/data/task0012_260321/calib/left_global_20260104/result.json \
  --right-json /data/haoxiang/data/task0012_260321/calib/right_global_20260104/result.json \
  --left-pose-b2e /data/haoxiang/data/task0012_260321/calib/left_global_20260104/pose_b2e.csv \
  --right-pose-b2e /data/haoxiang/data/task0012_260321/calib/right_global_20260104/pose_b2e.csv \
  --intrinsics-npy /data/haoxiang/data/task0012_260321/1736320913189/intrinsics.npy \
  --global-serial 105422061350 \
  --inhand-left-serial 104122064161 \
  --inhand-right-serial 104122061330 \
  --intrinsic-selector first \
  --left-row 0 \
  --right-row 0 \
  --target-mode inverse_real_base \
  --output /data/haoxiang/data/task0012_260321/calib/1234567.npy
```

### 6.2 `inverse_predefined_base`

```bash
python ca-pi-xie-device-tools/build_airexo_robot_calib_equivalent.py \
  --left-json /data/haoxiang/data/task0012_260321/calib/left_global_20260104/result.json \
  --right-json /data/haoxiang/data/task0012_260321/calib/right_global_20260104/result.json \
  --left-pose-b2e /data/haoxiang/data/task0012_260321/calib/left_global_20260104/pose_b2e.csv \
  --right-pose-b2e /data/haoxiang/data/task0012_260321/calib/right_global_20260104/pose_b2e.csv \
  --intrinsics-npy /data/haoxiang/data/task0012_260321/1736320913189/intrinsics.npy \
  --global-serial 105422061350 \
  --inhand-left-serial 104122064161 \
  --inhand-right-serial 104122061330 \
  --intrinsic-selector first \
  --left-row 0 \
  --right-row 0 \
  --target-mode inverse_predefined_base \
  --output /data/haoxiang/data/task0012_260321/calib/1234567.npy
```

如果这两种模式比之前明显更合理，就说明：

- 你的 [`result.json`](result.json) 方向确实不是 `camera -> base`
- 而更像 `base -> camera`

---

## 7. notebook 侧最小改动

你现在 notebook 里这段：

```python
calib_info = CalibrationInfo(
    calib_path=calib_path,
    calib_timestamp=1234567
)
```

仍然可以保持不变。

你只需要：

- 用不同 [`target_mode`](../build_airexo_robot_calib_equivalent.py) 重写同一个 [`1234567.npy`](ca-pi-xie-device-tools/reconstruct_two_arm.ipynb)
- 再 rerun notebook 看 overlay

---

## 8. 当前目标

这一步的目标很明确：

- 先判断是不是**整个 4x4 方向反了**
- 再去细调 fixed transform 层级

因为“base 在前面，但应该在后面”已经是一个比“偏 90°”更强的信号：优先应该检查 `camera->base` 与 `base->camera` 是否搞反。