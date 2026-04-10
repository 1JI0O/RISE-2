# `inspect_json_robot_render_inline.ipynb` 里的 frame 是怎么来的

本文专门解释 [`ca-pi-xie-device-tools/inspect_json_robot_render_inline.ipynb`](../inspect_json_robot_render_inline.ipynb) 里那几组坐标轴（尤其是 `camera`、`left_base`、`right_base`、`left_tcp_fk`、`right_tcp_fk`）到底代表什么、是如何算出来的，以及为什么它们看起来有时“不在你以为的底座位置上”。

---

## 1. 先讲结论

在这个 notebook 里，所谓的 **frame**，本质上就是一个 `4x4` 刚体变换矩阵的可视化：

- 平移部分决定坐标系原点在哪里；
- 旋转部分决定这个坐标系的 `x/y/z` 轴朝向。

所以你看到的“一个 frame”，并不是什么特殊对象，它只是某个矩阵 `T` 的画图结果。

更具体地说：

- `camera`：当前 Plotly 世界里的相机参考坐标系；
- `left_base`：当前代码认为的“左臂 base 所在参考系”；
- `right_base`：当前代码认为的“右臂 base 所在参考系”；
- `left_tcp_fk` / `right_tcp_fk`：由当前 joint 通过 FK 算出来的 TCP 参考系。

这里最容易误解的一点是：

> **你在图里看到的 `left_base/right_base`，不一定等于你肉眼理解的“机械臂底座几何中心”。**

这正是很多“为什么 frame 看起来不在底座上”的来源。

---

## 2. frame 在代码里到底怎么画出来

画 frame 的函数是 [`add_frame()`](../inspect_json_robot_render_inline.ipynb)。

它的核心逻辑可以概括成：

1. 取矩阵 `T[:3, 3]` 作为原点；
2. 取矩阵 `T[:3, :3]` 的三列作为三个轴方向；
3. 从原点沿每个轴方向画一条线：
   - 红色：`x`
   - 绿色：`y`
   - 蓝色：`z`

也就是说，如果你传进去的是某个矩阵 `T_world_A`，那么画出来的就是：

- 坐标系 `A` 的原点在 `world` 里哪里；
- 坐标系 `A` 的 `x/y/z` 轴在 `world` 里朝向哪里。

所以：

> **frame 不是“机械臂模型的一部分”，而是“某个坐标变换矩阵”的可视化。**

---

## 3. notebook 里具体画了哪些 frame

在 notebook 的 frame 区域，代码逻辑大致是：

- 画 `camera`
- 画 `left_base`
- 画 `right_base`
- 如果打开 TCP 开关，再画 `left_tcp_fk` 和 `right_tcp_fk`

因此，如果你只开了 `SHOW_FRAME_AXES`，通常会看到三组 frame：

1. `camera`
2. `left_base`
3. `right_base`

如果还开了 `SHOW_TCP_FRAME`，则还会多看到：

4. `left_tcp_fk`
5. `right_tcp_fk`

很多时候你会主观说“有三个坐标轴”，其实更准确的说法是：

> **有三组坐标系，每组坐标系各自有三根轴。**

---

## 4. `camera` frame 是什么

`camera` frame 是最简单的一个，它直接来自单位阵：

- 单位阵的平移是 `0`
- 单位阵的旋转是标准坐标轴

所以 `camera` 本质上表示的是：

> 当前 Plotly 场景所采用的世界参考系原点。

这里还有一个非常重要但容易忽略的细节：

### 4.1 这个 “camera” 不是原始 pinhole camera 裸坐标

因为点云在构建时，已经先乘过 [`O3D_RENDER_TRANSFORMATION`](../../airexo/airexo/helpers/constants.py)。

所以 notebook 里的 Plotly 世界，其实已经不是最原始的相机坐标，而是：

> **经过 Open3D / renderer 对齐后的相机可视化坐标系。**

这意味着：

- 你看到的 `camera` frame 是“渲染世界里的 camera frame”；
- 它是后续所有 mesh / frame / point cloud 对齐的共同参考。

---

## 5. `left_cam_to_base` / `right_cam_to_base` 是怎么来的

这一步是整个 notebook 最核心、也是最容易混乱的地方。

### 5.1 从 JSON 里读出 `pose_in_link`

notebook 会先读 JSON 标定文件里的 `pose_in_link`。

但问题在于：

> `pose_in_link` 这个字段本身的**方向语义并没有靠名字自动说明白**。

你必须额外决定它到底表示：

- `base_to_cam` ？
- 还是 `cam_to_base` ？
- 还是某个 real-base / predefined-base 层级下的 base？

### 5.2 notebook 当前做的事

notebook 里有一个函数专门把 JSON 结果转成 renderer 需要的 `cam_to_base`。

它当前的语义是：

1. 假设 JSON 给的是 `base_to_cam`；
2. 所以先求逆，得到 `cam_to_base`；
3. 再乘一个 `inv(ROBOT_PREDEFINED_TRANSFORMATION)`，试图把 JSON 所在的 base 层级调整到 renderer 使用的 base 层级。

最终得到两个矩阵：

- `left_cam_to_base`
- `right_cam_to_base`

此时这两个矩阵的角色可以理解为：

> **从 camera 世界，走到机器人 base 参考系的变换。**

注意这里的 “base” 还不是最终 mesh 里的每个 link，也不是肉眼意义上的底座壳体中心，而只是“renderer/FK 入口层”的 base。

---

## 6. 为什么还要乘 `ROBOT_PREDEFINED_TRANSFORMATION`

这一步最容易让人困惑，因为它看起来像“又多补了一个神秘矩阵”。

其实它不是补丁，而是一个**固定坐标系对齐矩阵**。

在这套 `airexo` renderer 逻辑里，存在至少两层“base”概念：

1. **真实控制/标定层的 base**
2. **URDF 模型层的 base**

URDF 在建模时，base 原点和轴向是建模者定义的；
而真实机器人控制和标定时，采用的 base 原点和轴向则来自系统约定。

这两者并不保证完全一致。

所以：

> `ROBOT_PREDEFINED_TRANSFORMATION` 的作用，就是把“控制/真实 base 层”和“URDF base 层”做固定对齐。

换句话说，它不是在表达“这个机械臂此时此刻动到了哪里”，而是在表达：

> “同一台机械臂，在不同参考坐标定义之间，如何固定地旋转/对齐。”

---

## 7. 为什么还要乘 `LEFT_ROBOT_PREDEFINED_TRANSFORMATION` / `RIGHT_ROBOT_PREDEFINED_TRANSFORMATION`

除了通用的 `ROBOT_PREDEFINED_TRANSFORMATION` 外，左右臂还有各自额外的固定矩阵：

- [`LEFT_ROBOT_PREDEFINED_TRANSFORMATION`](../../airexo/airexo/helpers/constants.py)
- [`RIGHT_ROBOT_PREDEFINED_TRANSFORMATION`](../../airexo/airexo/helpers/constants.py)

这表示：

> 左右臂在 URDF 建模和参考轴定义上，还有一层 side-specific 的固定差异。

原因通常包括：

- 左右臂是镜像装配；
- 左右臂在统一世界中的建模参考不完全对称；
- 为了让左右臂都复用类似的渲染/FK 流程，需要额外补一个固定旋转。

所以当你看到 mesh 变换链中既有：

- `ROBOT_PREDEFINED_TRANSFORMATION`
- 又有 `LEFT/RIGHT_ROBOT_PREDEFINED_TRANSFORMATION`

这不是重复，而是：

1. 一层是“通用 robot base ↔ URDF base”的对齐；
2. 一层是“左/右臂各自”的额外对齐。

---

## 8. 为什么 mesh 的链条比 frame 更长

### 8.1 frame 只画到 base 这一层

当我们画 `left_base` / `right_base` 时，本质上是在可视化一个“base 层”的矩阵。

也就是说，它只回答：

- 左/右 base 参考系的原点在哪；
- 左/右 base 的轴朝向如何；

### 8.2 mesh 还要继续往下走到每个 link / visual

而机械臂 mesh 的渲染链则还包含：

- FK 计算出的每个 `link` 的位姿
- `visual` 自己相对 link 的偏移

也就是说，mesh 的最终位置是：

1. 相机 / 渲染世界
2. → base 层
3. → 某个 link 层
4. → 某个 visual mesh 层

所以你看到的整条机械臂，是一个“base + 所有 link + 所有 mesh offset”叠加出来的结果。

这就解释了一个非常重要的现象：

> **就算 `left_base` frame 看起来不在你肉眼认为的“底座几何中心”，机械臂 mesh 仍然可能整体是对的。**

因为 mesh 后面还有更多层的补偿和偏移。

---

## 9. 为什么 `left_base/right_base` 看起来可能“不在底座上”

这正是你现在最困惑的问题。

这里有几种非常常见的原因。

### 9.1 你画的是“参考 base”，不是“底座几何中心”

`left_base/right_base` 所代表的是当前 renderer 链路里定义的某一层 base 坐标系。

但你肉眼看到的“底座位置”，通常指的是：

- 机械臂金属底盘的几何中心
- 或者底座壳体某个平面中心
- 或者地面安装点

而 URDF / renderer 里的 base 原点往往并不选在这些地方。

所以：

> **“frame 不在底座中心”并不自动说明它错了。**

它有可能只是 base 原点定义得不符合你的视觉直觉。

### 9.2 你画的 frame 是某一层 base，但你期待的是另一层 base

你脑中“base”这个词，可能在无意识里混了几种不同对象：

- real base
- URDF base
- predefined 对齐后的 base
- 左右臂 side-specific 对齐后的 base

而 notebook 当前画的是其中某一层。

这会导致你看到 frame 时产生强烈不适感：

> “这不是我理解的那个 base 啊。”

这不是错觉，而是很可能你确实拿“不同层的 base”在比较。

### 9.3 少/多乘某个固定矩阵时，会产生“像是镜像过去了”的视觉感受

如果一个 frame 看起来似乎“绕 camera 对称一下就能回去”，那在代码语义上通常意味着：

- 这个矩阵方向可能取反了（`cam_to_base` / `base_to_cam`）
- 或者少/多乘了一个固定轴翻转矩阵
- 或者少/多乘了某一层 predefined 变换

所以你的视觉直觉并不玄学，它其实就是在提示：

> **frame 的来龙去脉，和你想象中的层级不一致。**

---

## 10. 为什么有时会出现“mesh 对，但 frame 不对”的诡异现象

这个现象完全可能成立，而且并不矛盾。

原因如下：

### 10.1 mesh 用的是整条链，frame 只看中间一层

mesh 的最终变换是：

- 渲染世界
- → base 层
- → FK link 层
- → visual offset 层

frame 只是把“某个中间层矩阵”单独拿出来画。

因此，哪怕这层 frame 在视觉上不贴壳体中心，mesh 依然可能通过后续的 link / offset 表现得很合理。

### 10.2 你对 “姿势对” 和 “base 对” 的判断标准不同

你看 mesh 时，通常关注的是：

- 整条机械臂姿态是否和点云对齐；
- TCP 朝向是否大致合理；
- 左右机械臂是否落在工作空间中正确位置。

但你看 frame 时，关注的是：

- 原点是不是恰好在底座位置；
- 轴方向是不是直觉上对。

这两个标准不是同一个东西。

所以：

> mesh “看起来对” 不自动推出 frame “一定在你期待的位置”。

---

## 11. 你现在最应该如何理解这个 frame

最稳妥的理解方式不是：

> “它是不是机械臂底座的真实物理中心？”

而是：

> **“它是当前 notebook / renderer 在进入 FK 之前，所采用的那个 base 参考坐标系。”**

这个表述更准确，也更接近代码事实。

它说明的是：

- renderer 认为机器人 base 应该在什么位置；
- renderer 认为 base 的轴应该朝向哪里；
- 然后所有 link 和 mesh 都会在这个参考上继续展开。

---

## 12. 你现在的几个直觉，分别在代码里对应什么

### 直觉 A：
> `left_base/right_base` 看起来不在底座上。

代码含义：
- 你画的这个 frame 对应的 base 层，不一定是你主观想看的那个 base 层；
- 或者 URDF base 原点本来就不在壳体几何中心。

### 直觉 B：
> 好像绕 camera 对称一下就能回去。

代码含义：
- 有可能 `cam_to_base` / `base_to_cam` 方向理解错；
- 或者少/多乘了一个固定镜像/翻转矩阵；
- 或者你比较的是不同层级的 base。

### 直觉 C：
> mesh 看起来是对的，但 frame 又不对。

代码含义：
- mesh 比 frame 多走了 FK 和 visual offset 两层；
- 你看到的是不同层对象，不矛盾。

---

## 13. 最后的简化版理解

如果把整个 notebook 想成一条流水线，可以这么理解：

1. 从 JSON 得到一个相机与机器人之间的外参；
2. 把它转换到 renderer 所需的 base 语义层；
3. 这个转换结果，就是 `left_base/right_base` frame 的来源；
4. 再在这个 base 上叠加 joint 的 FK；
5. 再叠加 visual mesh offset；
6. 最后画出机械臂 mesh。

所以：

> **frame 是 mesh 的“上游参考层”，不是 mesh 本身。**

也正因为如此，它既重要，又不一定长得像你眼里看到的“底座中心”。

---

## 14. 一句话总结

`inspect_json_robot_render_inline.ipynb` 里的 `frame`，来源于：

**JSON 外参 → notebook 当前假设下的 `cam_to_base` → predefined / side-predefined 坐标系对齐 → 将这个 4x4 变换矩阵的原点和三轴画出来。**

它表示的是：

> **renderer 进入 FK 之前所采用的参考坐标系层。**

而不是“肉眼意义上的机械臂底座几何中心”。
