# Renderer 变换链条代码分析：只看真实 Robot 渲染分支

本文档只分析 **真实 robot 渲染分支**。

这里说的“airexo 风格标定”，是指你当前使用的那套 **标定文件格式 / 数据组织风格**，用于和 RISE2 风格区分；**不是要分析 AirExo 硬件本体的渲染链条**。

因此本文档刻意删除了：
- AirExo 本体 renderer 分支
- `airexo_left.h5` / `airexo_right.h5` 的可视化链条
- `AIREXO_PREDEFINED_TRANSFORMATION` 对外骨骼 mesh 的渲染解释

本文只保留：
- 真实 robot 的标定矩阵如何被读取
- `camera -> robot base` 如何被构造
- renderer 如何使用这些矩阵把 **robot URDF** 渲染到图像中

---

## 1. 真正需要看的入口

你关心的真实 robot 渲染链，主要对应下面这些代码：

- 标定读取与 `camera -> base` 计算：
  - [`CalibrationInfo`](../airexo/airexo/calibration/calib_info.py:14)
  - [`get_camera_to_robot_left_base()`](../airexo/airexo/calibration/calib_info.py:98)
  - [`get_camera_to_robot_right_base()`](../airexo/airexo/calibration/calib_info.py:106)
  - [`get_camera_to_base()`](../airexo/airexo/calibration/calib_info.py:81)

- 机器人渲染器：
  - [`RobotRenderer`](../airexo/airexo/helpers/renderer.py:150)
  - [`SeparateRobotRenderer`](../airexo/airexo/helpers/renderer.py:269)

- 真实渲染入口脚本：
  - [`airexo/airexo/adaptor/render.py`](../airexo/airexo/adaptor/render.py:1)

- 关节变换（当输入仍是 exo 风格编码器时）：
  - [`transform_arm()`](../airexo/airexo/helpers/transform.py:45)

---

## 2. 标定文件从哪里进入 renderer 链条

标定文件的读取入口是 [`CalibrationInfo.__init__()`](../airexo/airexo/calibration/calib_info.py:22)。

### 2.1 先定位到具体标定文件
初始化时，代码先根据：
- [`calib_path`](../airexo/airexo/calibration/calib_info.py:24)
- [`calib_timestamp`](../airexo/airexo/calibration/calib_info.py:25)

拼出标定文件路径：
- [`self.calib_file_path = os.path.join(calib_path, "{}.npy".format(calib_timestamp))`](../airexo/airexo/calibration/calib_info.py:30)

然后做存在性检查：
- [`assert os.path.exists(self.calib_file_path)`](../airexo/airexo/calibration/calib_info.py:31)

这一步意味着：
- renderer 并不是直接吃某个配置里硬编码的矩阵
- 而是先通过 [`CalibrationInfo`](../airexo/airexo/calibration/calib_info.py:14) 找到一个 `.npy` 标定文件
- 后续所有 `camera -> base` 计算都从这个 `.npy` 的内容开始

### 2.2 把 `.npy` 整体读成一个字典
真正读入发生在：
- [`calib_file = np.load(self.calib_file_path, allow_pickle = True).item()`](../airexo/airexo/calibration/calib_info.py:32)

也就是说，标定文件被当作一个 Python dict 使用。

这一步之后，代码并没有立刻计算 renderer 需要的 `cam_to_base`，而是先把原始字段缓存到 [`CalibrationInfo`](../airexo/airexo/calibration/calib_info.py:14) 对象里。

### 2.3 被缓存下来的核心字段
最关键的是下面这些：

- 标定类型：
  - [`self.calib_type = calib_file["type"]`](../airexo/airexo/calibration/calib_info.py:33)

- 相机身份信息：
  - [`self.camera_serials`](../airexo/airexo/calibration/calib_info.py:36)
  - [`self.camera_serials_global`](../airexo/airexo/calibration/calib_info.py:37)
  - [`self.camera_serial_inhand_left`](../airexo/airexo/calibration/calib_info.py:38)
  - [`self.camera_serial_inhand_right`](../airexo/airexo/calibration/calib_info.py:39)

- 相机参数：
  - [`self.intrinsics`](../airexo/airexo/calibration/calib_info.py:40)
  - [`self.extrinsics`](../airexo/airexo/calibration/calib_info.py:41)

- 如果是 robot 类型标定：
  - [`self.robot_left = calib_file["robot_left"]`](../airexo/airexo/calibration/calib_info.py:46)
  - [`self.robot_right = calib_file["robot_right"]`](../airexo/airexo/calibration/calib_info.py:47)

这几类字段的职责完全不同：

#### A. [`intrinsics`](../airexo/airexo/calibration/calib_info.py:40)
只负责 **投影**，也就是 Open3D 相机如何把 3D 物体投到 2D 图像上。
它不参与 base 恢复。

#### B. [`extrinsics`](../airexo/airexo/calibration/calib_info.py:41)
这是 **renderer 变换链最底层的外参输入之一**。
它描述的是各个相机相对于公共参考系（代码约定下通常是 marker/global marker）的位姿关系。

#### C. [`robot_left["tcp_pose"]`](../airexo/airexo/calibration/calib_info.py:101) / [`robot_right["tcp_pose"]`](../airexo/airexo/calibration/calib_info.py:109)
这两个字段不是相机参数，而是**标定时刻机器人 TCP 在各自 robot base 下的位姿**。
后面恢复 `camera -> robot_base` 时，要用它把 “camera -> tcp” 接回 “camera -> base”。

#### D. [`camera_serial_inhand_left`](../airexo/airexo/calibration/calib_info.py:38) / [`camera_serial_inhand_right`](../airexo/airexo/calibration/calib_info.py:39)
这两个字段非常关键，因为代码需要知道：
- 哪台相机是左臂手上的 inhand camera
- 哪台相机是右臂手上的 inhand camera

后面的公式里，会通过：
- [`self.extrinsics[self.camera_serial_inhand_left]`](../airexo/airexo/calibration/calib_info.py:101)
- [`self.extrinsics[self.camera_serial_inhand_right]`](../airexo/airexo/calibration/calib_info.py:109)

把当前全局相机，和 inhand 相机接到同一个公共参考系里。

### 2.4 为什么这里说“最关键的输入矩阵”是这些
对真实 robot 渲染链来说，最终目的是得到：
- 左臂的 [`camera -> left_robot_base`](../airexo/airexo/calibration/calib_info.py:98)
- 右臂的 [`camera -> right_robot_base`](../airexo/airexo/calibration/calib_info.py:106)
- 或者进一步合成的共享 [`camera -> shared_base`](../airexo/airexo/calibration/calib_info.py:81)

而这些矩阵并不是直接存放在标定文件里的；它们是由下面几部分**现场拼出来**的：

1. [`extrinsics`](../airexo/airexo/calibration/calib_info.py:41)
   - 提供 `当前相机 -> 公共参考系`
   - 也提供 `inhand相机 -> 公共参考系`

2. [`robot_left["tcp_pose"]`](../airexo/airexo/calibration/calib_info.py:101) / [`robot_right["tcp_pose"]`](../airexo/airexo/calibration/calib_info.py:109)
   - 提供 `base -> tcp`
   - 求逆后可得到 `tcp -> base`

3. 常量 [`ROBOT_LEFT_CAM_TO_TCP`](../airexo/airexo/helpers/constants.py:35) / [`ROBOT_RIGHT_CAM_TO_TCP`](../airexo/airexo/helpers/constants.py:36)
   - 提供 `inhand camera -> tcp`
   - 这是硬件安装的固定几何关系

4. 常量 [`ROBOT_LEFT_REAL_BASE_TO_REAL_BASE`](../airexo/airexo/helpers/constants.py:42) / [`ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE`](../airexo/airexo/helpers/constants.py:50)
   - 提供左右真实基座到共享真实基座的固定关系
   - 用于双臂共享 base 的合成

5. 常量 [`ROBOT_PREDEFINED_TRANSFORMATION`](../airexo/airexo/helpers/constants.py:91)
   - 提供 real base 与 URDF base 之间的坐标系对齐
   - 这是 renderer 真正落到 mesh 之前必须补的一层

所以更准确地说：
- 标定文件里最关键的“原始输入”是 [`extrinsics`](../airexo/airexo/calibration/calib_info.py:41) 和 [`robot_left/right["tcp_pose"]`](../airexo/airexo/calibration/calib_info.py:101)
- 常量里最关键的是 [`ROBOT_LEFT_CAM_TO_TCP`](../airexo/airexo/helpers/constants.py:35)、[`ROBOT_RIGHT_CAM_TO_TCP`](../airexo/airexo/helpers/constants.py:36)、[`ROBOT_PREDEFINED_TRANSFORMATION`](../airexo/airexo/helpers/constants.py:91)、[`ROBOT_LEFT_REAL_BASE_TO_REAL_BASE`](../airexo/airexo/helpers/constants.py:42)、[`ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE`](../airexo/airexo/helpers/constants.py:50)
- 这两类信息结合起来，才组成 renderer 真正使用的 `cam_to_base`

### 2.5 从“读文件”到“进入渲染链”的真正过渡点
标定文件虽然是在 [`CalibrationInfo.__init__()`](../airexo/airexo/calibration/calib_info.py:22) 中读入的，但它真正进入渲染链的转折点，是下面这几个函数被调用时：

- [`get_camera_to_robot_left_base()`](../airexo/airexo/calibration/calib_info.py:98)
- [`get_camera_to_robot_right_base()`](../airexo/airexo/calibration/calib_info.py:106)
- [`get_camera_to_base()`](../airexo/airexo/calibration/calib_info.py:81)

因为只有到这里，原始字段才被拼装成 renderer 直接使用的刚体变换矩阵。

换句话说：
- [`__init__()`](../airexo/airexo/calibration/calib_info.py:22) 只是“把原料搬进来”
- [`get_camera_to_robot_left_base()`](../airexo/airexo/calibration/calib_info.py:98) / [`get_camera_to_robot_right_base()`](../airexo/airexo/calibration/calib_info.py:106) 才是“真正开始配方计算”
- [`get_camera_to_base()`](../airexo/airexo/calibration/calib_info.py:81) 则是在双臂共享 renderer 语义下，进一步把左右结果合并

所以，如果你要追 `renderer` 最终为什么对不齐，真正应该顺着查的是：
- 标定文件里 [`extrinsics`](../airexo/airexo/calibration/calib_info.py:41) 是否方向正确
- 标定文件里 [`robot_left["tcp_pose"]`](../airexo/airexo/calibration/calib_info.py:101) / [`robot_right["tcp_pose"]`](../airexo/airexo/calibration/calib_info.py:109) 的语义是否真的是 `base -> tcp`
- inhand 相机 serial 是否填对
- 常量 [`ROBOT_LEFT_CAM_TO_TCP`](../airexo/airexo/helpers/constants.py:35) / [`ROBOT_RIGHT_CAM_TO_TCP`](../airexo/airexo/helpers/constants.py:36) 是否对应当前硬件安装

---

## 3. 左右臂单独的 `camera -> robot_base` 是如何构造的

### 3.1 左臂

左臂核心公式在 [`get_camera_to_robot_left_base()`](../airexo/airexo/calibration/calib_info.py:98)：

```python
left_cam_to_base = (
    self.extrinsics[serial]
    @ np.linalg.inv(self.extrinsics[self.camera_serial_inhand_left])
    @ ROBOT_LEFT_CAM_TO_TCP
    @ np.linalg.inv(xyz_rot_to_mat(self.robot_left["tcp_pose"], rotation_rep = "quaternion"))
)
```

见 [`airexo/airexo/calibration/calib_info.py:101`](../airexo/airexo/calibration/calib_info.py:101)。

拆解如下：

1. [`self.extrinsics[serial]`](../airexo/airexo/calibration/calib_info.py:101)
   - 当前全局相机到公共参考系

2. [`np.linalg.inv(self.extrinsics[self.camera_serial_inhand_left])`](../airexo/airexo/calibration/calib_info.py:101)
   - 公共参考系到左手 inhand 相机

3. [`ROBOT_LEFT_CAM_TO_TCP`](../airexo/airexo/calibration/calib_info.py:101)
   - 左手 inhand camera 到 TCP

4. [`np.linalg.inv(xyz_rot_to_mat(self.robot_left["tcp_pose"], rotation_rep = "quaternion"))`](../airexo/airexo/calibration/calib_info.py:101)
   - TCP 到左臂 base

因此左臂链条是：

```text
T_cam_left_base
= T_cam_marker
  @ T_marker_left_inhand_cam
  @ T_left_inhand_cam_tcp
  @ T_tcp_left_base
```

### 3.2 右臂

右臂完全同理，见 [`get_camera_to_robot_right_base()`](../airexo/airexo/calibration/calib_info.py:106)：

```python
right_cam_to_base = (
    self.extrinsics[serial]
    @ np.linalg.inv(self.extrinsics[self.camera_serial_inhand_right])
    @ ROBOT_RIGHT_CAM_TO_TCP
    @ np.linalg.inv(xyz_rot_to_mat(self.robot_right["tcp_pose"], rotation_rep = "quaternion"))
)
```

见 [`airexo/airexo/calibration/calib_info.py:109`](../airexo/airexo/calibration/calib_info.py:109)。

---

## 4. `ROBOT_LEFT_CAM_TO_TCP` / `ROBOT_RIGHT_CAM_TO_TCP` 是怎么来的

这两个常量不是拍脑袋来的，是由 flange 与相机的固定位姿，以及 TCP 到 flange 的固定位姿组合得到。

见 [`airexo/airexo/helpers/constants.py`](../airexo/airexo/helpers/constants.py:5)：
- [`ROBOT_LEFT_FLANGE_TO_CAM`](../airexo/airexo/helpers/constants.py:5)
- [`ROBOT_RIGHT_FLANGE_TO_CAM`](../airexo/airexo/helpers/constants.py:14)
- [`ROBOT_TCP_TO_FLANGE`](../airexo/airexo/helpers/constants.py:23)

然后构造：
- [`ROBOT_LEFT_TCP_TO_CAM = ROBOT_TCP_TO_FLANGE @ ROBOT_LEFT_FLANGE_TO_CAM`](../airexo/airexo/helpers/constants.py:32)
- [`ROBOT_RIGHT_TCP_TO_CAM = ROBOT_TCP_TO_FLANGE @ ROBOT_RIGHT_FLANGE_TO_CAM`](../airexo/airexo/helpers/constants.py:33)
- 再求逆得到：
  - [`ROBOT_LEFT_CAM_TO_TCP`](../airexo/airexo/helpers/constants.py:35)
  - [`ROBOT_RIGHT_CAM_TO_TCP`](../airexo/airexo/helpers/constants.py:36)

也就是说：

```text
camera -> tcp = inverse(tcp -> flange -> camera)
```

这一步非常关键，因为 renderer / calib 的链条里需要把 inhand camera 的外参，接到 robot TCP 上。

---

## 5. 为什么还要区分 real base 和 URDF base

在 [`get_camera_to_robot_left_base()`](../airexo/airexo/calibration/calib_info.py:98) 和 [`get_camera_to_robot_right_base()`](../airexo/airexo/calibration/calib_info.py:106) 中，都会看到：

```python
if not real_base:
    left_cam_to_base = left_cam_to_base @ np.linalg.inv(ROBOT_PREDEFINED_TRANSFORMATION)
```

见 [`airexo/airexo/calibration/calib_info.py:102`](../airexo/airexo/calibration/calib_info.py:102) 和 [`airexo/airexo/calibration/calib_info.py:110`](../airexo/airexo/calibration/calib_info.py:110)。

这说明：
- 一开始恢复出来的是 **真实机器人控制系里的 base**
- 但 renderer 中使用 URDF 时，mesh / FK 所在的是 **URDF base**
- 需要通过 [`ROBOT_PREDEFINED_TRANSFORMATION`](../airexo/airexo/helpers/constants.py:91) 在二者之间做坐标系对齐

常量定义见：
- [`ROBOT_PREDEFINED_TRANSFORMATION`](../airexo/airexo/helpers/constants.py:91)

所以这里有两层 base：
- real base：控制/标定意义下的 base
- URDF base：robot.urdf 里的 base link 所在系

renderer 真正使用的是 **URDF base 版本** 的 `cam_to_base`。

---

## 6. 共享 `cam_to_base` 是怎样由左右臂合成的

对普通双臂 robot renderer，代码并不总是使用左右臂独立外参；很多时候是使用共享的 [`cam_to_base`](../airexo/airexo/helpers/renderer.py:158)。

这个共享矩阵由 [`get_camera_to_base()`](../airexo/airexo/calibration/calib_info.py:81) 生成：

```python
left_cam_to_base = self.get_camera_to_robot_left_base(serial, real_base = True) @ ROBOT_LEFT_REAL_BASE_TO_REAL_BASE
right_cam_to_base = self.get_camera_to_robot_right_base(serial, real_base = True) @ ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE
cam_to_base = average_xyz_rot_quat(left_cam_to_base, right_cam_to_base, rotation_rep = "matrix")
if not real_base:
    cam_to_base = cam_to_base @ np.linalg.inv(ROBOT_PREDEFINED_TRANSFORMATION)
```

见 [`airexo/airexo/calibration/calib_info.py:91`](../airexo/airexo/calibration/calib_info.py:91)。

这里做了两件事：

1. 先把左、右单臂相机到各自 real base 的矩阵，转换到共享 real base：
   - 左侧乘 [`ROBOT_LEFT_REAL_BASE_TO_REAL_BASE`](../airexo/airexo/helpers/constants.py:42)
   - 右侧乘 [`ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE`](../airexo/airexo/helpers/constants.py:50)

2. 再用 [`average_xyz_rot_quat()`](../airexo/airexo/calibration/calib_info.py:93) 对左右结果做平均

于是得到一个共享的 `camera -> shared_base`。

最后如果 renderer 需要 URDF base，就再右乘 [`np.linalg.inv(ROBOT_PREDEFINED_TRANSFORMATION)`](../airexo/airexo/calibration/calib_info.py:95)。

---

## 7. 真实 robot 渲染分支是怎样启动的

在 [`airexo/airexo/adaptor/render.py`](../airexo/airexo/adaptor/render.py:99) 中，robot 渲染器初始化为：

```python
robot_renderer = hydra.utils.instantiate(
    cfg.robot_renderer,
    cam_to_base = calib_info.get_camera_to_base(cfg.camera_serial),
    intrinsic = calib_info.get_intrinsic(cfg.camera_serial)
)
```

见 [`airexo/airexo/adaptor/render.py:100`](../airexo/airexo/adaptor/render.py:100)。

也就是说，对普通 robot renderer：
- 输入的相机到 base 外参，来自 [`get_camera_to_base()`](../airexo/airexo/calibration/calib_info.py:81)
- 这是一个**共享 base** 版本，不区分左右

如果是标定标注器那条链，会用 [`SeparateRobotRenderer`](../airexo/airexo/helpers/renderer.py:269)，分别传：
- [`cam_to_left_base`](../airexo/airexo/helpers/renderer.py:277)
- [`cam_to_right_base`](../airexo/airexo/helpers/renderer.py:278)

例如 [`airexo/airexo/calibration/annotator.py`](../airexo/airexo/calibration/annotator.py:178)。

---

## 8. joint 是怎样进入 renderer 的

### 8.1 如果输入已经是 robot joints
那就直接调用：
- [`robot_renderer.update_joints(left_joint, right_joint)`](../airexo/airexo/helpers/renderer.py:247)

### 8.2 如果输入仍然是 exo 风格传感器数据
则先通过 [`transform_arm()`](../airexo/airexo/helpers/transform.py:45) 转成 robot joints。

在 [`airexo/airexo/adaptor/render.py`](../airexo/airexo/adaptor/render.py:109) 中：

```python
robot_left_joint = transform_arm(...)
robot_right_joint = transform_arm(...)
robot_renderer.update_joints(robot_left_joint, robot_right_joint)
```

见：
- [`transform_arm()`](../airexo/airexo/adaptor/render.py:109)
- [`transform_arm()`](../airexo/airexo/adaptor/render.py:115)
- [`update_joints()`](../airexo/airexo/adaptor/render.py:122)

`transform_arm()` 的逻辑在 [`airexo/airexo/helpers/transform.py`](../airexo/airexo/helpers/transform.py:45)：
- 逐 joint 调用 [`transform_joint()`](../airexo/airexo/helpers/transform.py:12)
- 根据 calib config 类型执行：
  - [`fixed`](../airexo/airexo/helpers/transform.py:20)
  - [`scaling`](../airexo/airexo/helpers/transform.py:28)
  - [`mapping`](../airexo/airexo/helpers/transform.py:33)

所以：
- exo 风格输入不会直接喂给 robot FK
- 必须先映射为 robot 关节空间

---

## 9. FK 之后，mesh 如何被送到图像里

### 9.1 普通 `RobotRenderer`

在 [`RobotRenderer`](../airexo/airexo/helpers/renderer.py:150) 中，初始化和更新时都使用同一条核心公式：

```python
tf = O3D_RENDER_TRANSFORMATION @ self.cam_to_base @ ROBOT_PREDEFINED_TRANSFORMATION @ transform.matrix() @ v.offset.matrix()
```

见：
- 初始化 [`airexo/airexo/helpers/renderer.py:207`](../airexo/airexo/helpers/renderer.py:207)
- 更新 [`airexo/airexo/helpers/renderer.py:241`](../airexo/airexo/helpers/renderer.py:241)

逐项解释：

1. [`transform.matrix()`](../airexo/airexo/helpers/renderer.py:207)
   - robot FK 结果，表示 URDF base 到当前 link 的变换

2. [`v.offset.matrix()`](../airexo/airexo/helpers/renderer.py:207)
   - 该 visual mesh 相对 link 的局部 offset

3. [`ROBOT_PREDEFINED_TRANSFORMATION`](../airexo/airexo/helpers/constants.py:91)
   - 把 robot URDF base 对齐到 renderer / 标定链条使用的机器人 base 语义

4. [`self.cam_to_base`](../airexo/airexo/helpers/renderer.py:207)
   - 把 base 系中的物体送到 camera 系

5. [`O3D_RENDER_TRANSFORMATION`](../airexo/airexo/helpers/constants.py:120)
   - 把 camera 系再转换到 Open3D renderer 所期待的坐标系

因此最终链条是：

```text
mesh_local
-> visual offset
-> FK(link in URDF base)
-> ROBOT_PREDEFINED_TRANSFORMATION
-> cam_to_base
-> O3D_RENDER_TRANSFORMATION
-> Open3D scene
```

写成总变换：

```text
T_render_mesh = T_o3d_render
              @ T_cam_base
              @ T_robot_predefined
              @ T_fk_link
              @ T_visual_offset
```

### 9.2 `SeparateRobotRenderer`

左右臂分离版则是：

左臂：
```python
tf = O3D_RENDER_TRANSFORMATION @ self.cam_to_left_base @ ROBOT_PREDEFINED_TRANSFORMATION @ LEFT_ROBOT_PREDEFINED_TRANSFORMATION @ transform.matrix() @ v.offset.matrix()
```

见 [`airexo/airexo/helpers/renderer.py:337`](../airexo/airexo/helpers/renderer.py:337)。

右臂：
```python
tf = O3D_RENDER_TRANSFORMATION @ self.cam_to_right_base @ ROBOT_PREDEFINED_TRANSFORMATION @ RIGHT_ROBOT_PREDEFINED_TRANSFORMATION @ transform.matrix() @ v.offset.matrix()
```

见 [`airexo/airexo/helpers/renderer.py:351`](../airexo/airexo/helpers/renderer.py:351)。

这里多出来的：
- [`LEFT_ROBOT_PREDEFINED_TRANSFORMATION`](../airexo/airexo/helpers/constants.py:98)
- [`RIGHT_ROBOT_PREDEFINED_TRANSFORMATION`](../airexo/airexo/helpers/constants.py:105)

是为了让左右单臂 URDF 的局部 base 方向对齐到双臂共享的机器人定义。

---

## 10. Open3D 最后如何投影成图像

mesh 放进 scene 之后，还需要相机投影。

这个由 renderer 初始化时的：
- [`self.renderer.scene.camera.set_projection(intrinsic, near_plane, far_plane, float(width), float(height))`](../airexo/airexo/helpers/renderer.py:216)
- 或 [`set_projection(...)`](../airexo/airexo/helpers/renderer.py:360)

来完成。

这里的 [`intrinsic`](../airexo/airexo/adaptor/render.py:82) 来自：
- [`calib_info.get_intrinsic(cfg.camera_serial)`](../airexo/airexo/adaptor/render.py:82)

然后通过：
- [`render_image()`](../airexo/airexo/helpers/renderer.py:255)
- [`render_depth()`](../airexo/airexo/helpers/renderer.py:258)
- [`render_mask()`](../airexo/airexo/helpers/renderer.py:261)

输出最终结果。

所以最终能够渲染出来，是因为：
1. joint 经过 FK 得到 link 位姿
2. link 位姿乘上 base / camera / Open3D 各级变换
3. 内参投影被正确设置
4. Open3D offscreen renderer 输出图像

---

## 11. 这条链条里最核心的几组矩阵

### 11.1 标定恢复 `camera -> base`
- [`extrinsics`](../airexo/airexo/calibration/calib_info.py:41)
- [`ROBOT_LEFT_CAM_TO_TCP`](../airexo/airexo/helpers/constants.py:35)
- [`ROBOT_RIGHT_CAM_TO_TCP`](../airexo/airexo/helpers/constants.py:36)
- [`robot_left["tcp_pose"]`](../airexo/airexo/calibration/calib_info.py:101)
- [`robot_right["tcp_pose"]`](../airexo/airexo/calibration/calib_info.py:109)

### 11.2 坐标系修正
- [`ROBOT_PREDEFINED_TRANSFORMATION`](../airexo/airexo/helpers/constants.py:91)
- [`LEFT_ROBOT_PREDEFINED_TRANSFORMATION`](../airexo/airexo/helpers/constants.py:98)
- [`RIGHT_ROBOT_PREDEFINED_TRANSFORMATION`](../airexo/airexo/helpers/constants.py:105)
- [`O3D_RENDER_TRANSFORMATION`](../airexo/airexo/helpers/constants.py:120)

### 11.3 joint -> FK
- [`transform_arm()`](../airexo/airexo/helpers/transform.py:45)
- [`forward_kinematic()`](../airexo/airexo/helpers/renderer.py:189)
- [`forward_kinematic_single()`](../airexo/airexo/helpers/renderer.py:314)

---

## 12. 最终结论：真实 robot renderer 的完整链条

如果只看你真正关心的 **真实 robot 渲染分支**，最简洁的链条是：

### 12.1 外参恢复
对于每只手：

```text
T_cam_robot_base
= extrinsics[global_cam]
  @ inv(extrinsics[inhand_cam])
  @ T_cam_tcp
  @ inv(T_base_tcp)
```

对应代码：
- 左臂 [`airexo/airexo/calibration/calib_info.py:101`](../airexo/airexo/calibration/calib_info.py:101)
- 右臂 [`airexo/airexo/calibration/calib_info.py:109`](../airexo/airexo/calibration/calib_info.py:109)

### 12.2 如果需要共享 base
再通过：
- [`ROBOT_LEFT_REAL_BASE_TO_REAL_BASE`](../airexo/airexo/helpers/constants.py:42)
- [`ROBOT_RIGHT_REAL_BASE_TO_REAL_BASE`](../airexo/airexo/helpers/constants.py:50)
- [`average_xyz_rot_quat()`](../airexo/airexo/calibration/calib_info.py:93)

得到共享的 [`cam_to_base`](../airexo/airexo/calibration/calib_info.py:93)。

### 12.3 渲染时真正乘到 mesh 上的矩阵
普通 robot renderer：

```text
T_render_mesh
= O3D_RENDER_TRANSFORMATION
  @ cam_to_base
  @ ROBOT_PREDEFINED_TRANSFORMATION
  @ FK
  @ visual_offset
```

分离左右臂 renderer：

左臂：

```text
T_render_mesh_left
= O3D_RENDER_TRANSFORMATION
  @ cam_to_left_base
  @ ROBOT_PREDEFINED_TRANSFORMATION
  @ LEFT_ROBOT_PREDEFINED_TRANSFORMATION
  @ FK_left
  @ visual_offset
```

右臂：

```text
T_render_mesh_right
= O3D_RENDER_TRANSFORMATION
  @ cam_to_right_base
  @ ROBOT_PREDEFINED_TRANSFORMATION
  @ RIGHT_ROBOT_PREDEFINED_TRANSFORMATION
  @ FK_right
  @ visual_offset
```

### 12.4 所以最终渲染之所以成立
本质上是：
- 标定文件提供了 [`extrinsics`](../airexo/airexo/calibration/calib_info.py:41) 和 TCP pose
- 代码恢复出 `camera -> robot base`
- joint 经 FK 变成 link pose
- link pose 经过若干坐标系对齐矩阵
- 最后进入 Open3D 的投影与渲染

---

## 13. 如果你后续要排查错位，最优先核查哪几处

如果真实 robot 渲染结果不对，最值得优先核查的地方是：

1. [`get_camera_to_robot_left_base()`](../airexo/airexo/calibration/calib_info.py:98) / [`get_camera_to_robot_right_base()`](../airexo/airexo/calibration/calib_info.py:106)
   - 尤其是 [`robot_left["tcp_pose"]`](../airexo/airexo/calibration/calib_info.py:101) / [`robot_right["tcp_pose"]`](../airexo/airexo/calibration/calib_info.py:109) 的语义是否和你手上的标定一致

2. [`ROBOT_LEFT_CAM_TO_TCP`](../airexo/airexo/helpers/constants.py:35) / [`ROBOT_RIGHT_CAM_TO_TCP`](../airexo/airexo/helpers/constants.py:36)
   - inhand 相机安装位姿是否与你的数据采集硬件一致

3. [`ROBOT_PREDEFINED_TRANSFORMATION`](../airexo/airexo/helpers/constants.py:91)
   - real base 与 URDF base 的对齐是否正确

4. 如果使用单独左右臂渲染：
   - [`LEFT_ROBOT_PREDEFINED_TRANSFORMATION`](../airexo/airexo/helpers/constants.py:98)
   - [`RIGHT_ROBOT_PREDEFINED_TRANSFORMATION`](../airexo/airexo/helpers/constants.py:105)

5. [`transform_arm()`](../airexo/airexo/helpers/transform.py:45)
   - 如果 joint 来源不是直接 robot joint，而是 exo 风格输入，这里非常容易引入系统性偏差

这几项里，最容易引发整体刚体错位的，通常是第 1、2、3 项。