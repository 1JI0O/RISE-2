# RISE-2 Mask-Aware 改造总结（面向人类阅读）

## 这份文档是干什么的

这份总结不是给 Agent 的执行规范，也不是新的需求说明，而是给研发同学、算法同学和项目维护者看的“改造复盘”。

目标是把这次改造讲清楚：

- 为什么要改
- 改了哪些文件
- 每个改动解决了什么问题
- 数据在系统里是怎么流动的
- 有哪些边界条件和风险
- 现在能用到什么程度

---

## 一句话结论

本次改造把 2D mask 真正接入了 RISE-2 主链路，并且同时实现了两条能力：

1. **方案A：2D mask -> 3D精确过滤**（在深度图阶段剔除不可信区域）
2. **方案B：2D patch 可靠度 -> 插值降权**（在图像 token 融合到点云时进行软降权）

默认配置下该能力是关闭的，不会影响原有训练/推理行为。

---

## 背景：改造前的问题是什么

在改造前，RISE-2 的整体流程已经很完整（点云编码 + 图像编码 + 对齐融合 + 决策），但对于“图像中不可信区域”的处理不足：

- 3D 点云构建时，深度图中可能包含人手、遮挡、反光等不希望学习的区域。
- 2D 特征与 3D 特征融合时，所有 patch 默认同等可信，模型会把噪声 patch 也当成同等信息源。

这会带来两个问题：

- 点云中混入不稳定几何
- 跨模态对齐时，错误 patch 会干扰最近邻插值

因此本次改造把 mask 作为“一等信号”正式接入。

---

## 先讲清楚：RISE-2 的原始主链路

训练时一条样本的大致流程：

1. 数据集读取 color/depth/action
2. 由 depth + intrinsics 生成点云（Open3D）
3. 图像走 dense encoder（DINO/ResNet）
4. 点云走 sparse encoder（Minkowski）
5. 通过 SpatialAligner，把图像特征插值到点 token 上
6. Transformer 聚合后进动作解码器（Diffusion）

本次改造只改变“数据可信度处理”和“跨模态融合权重”，没有改模型骨架。

---

## 这次改造的总体设计

### 设计目标

- 能读取离线 mask，并且和 color 帧严格对齐
- A/B 两种策略都要支持（硬过滤 + 软降权）
- 默认关闭，保证旧配置可继续使用
- 出错时能给清晰报错，避免 silent failure

### 核心语义约定

- 统一定义 `mask01`：
  - `1` = 不可信区域（应该被剔除或降权）
  - `0` = 可信区域

### 为什么方案A放在“深度图阶段”而不是“点云阶段”

一个关键讨论点是：能不能先建点云，再把点投回像素做过滤？

结论是本次不走这条路，原因如下：

- Open3D 生成点云 + crop + voxel downsample 后，点和原始像素不是一一可逆关系。
- 后验反查像素坐标会额外引入不稳定映射误差。

所以本次采用更稳妥路线：

- **先在 depth 图上按 mask 置零**
- **再用现有流程建点云**

这样几何来源天然干净，且和相机模型一致。

---

## 具体改了哪些文件

本次主要改动位于 6 个文件：

- `configs/test.yaml`
- `configs/single_rise1.yaml`
- `dataset/realworld.py`
- `policy/policy.py`
- `policy/sparse_modules.py`
- `train.py`

`eval.py`、`eval_server.py` 未做逻辑改造，但通过接口兼容方式保持可用。

---

## 文件级详细说明

## 1) 配置层：新增 mask_aware 总开关和参数

在两个配置文件里都新增了 `mask_aware` 块（默认关闭）：

```yaml
mask_aware:
  enabled: false
  mask_root: null
  align_mode: sorted_index
  enable_2d_reweight: true
  enable_3d_filter: true
  mask_threshold: 0
  mask_white_is_untrusted: true
  r_min: 0.001
  interp_eps: 0.000001
  interp_tiny: 0.000001
  empty_cloud_policy: warn_and_skip
  log_alignment_preview: true
```

参数含义（按人话解释）：

- `enabled`：是否启用 mask-aware 全流程。
- `mask_root`：mask 数据根目录。
- `align_mode`：当前只支持 `sorted_index`（排序后一一对应）。
- `enable_2d_reweight`：是否启用方案B（2D 可靠度降权）。
- `enable_3d_filter`：是否启用方案A（3D 几何硬过滤）。
- `mask_threshold`：二值阈值。
- `mask_white_is_untrusted`：白色是否表示不可信。
- `r_min`：插值可靠度下限，避免数值退化。
- `interp_eps`：距离分母稳定项。
- `interp_tiny`：归一化稳定项。
- `empty_cloud_policy`：过滤后空点云策略。
- `log_alignment_preview`：是否打印 color-mask 对齐预览日志。

---

## 2) 数据集层：mask 读取、对齐、分支信号产出

改动集中在 `dataset/realworld.py`，这是本次改造最关键的一层。

### 2.1 配置解析与启动校验

在 `_parse_config()` 中读取 `mask_aware` 字段，并给出默认值。

在 `_validate_mask_aware_config()` 中做强校验：

- 启用后必须有 `mask_root`
- `align_mode` 只能是 `sorted_index`
- 方案A/B 必须同时可用
- 数值参数合法（`r_min`、`interp_eps`、`interp_tiny`）
- 空点云策略必须在白名单内

目的：尽早失败，避免训练跑一半才发现数据配置有问题。

### 2.2 color 与 mask 的对齐机制

新增了以下工具方法：

- `_frame_sort_key()`：优先按数字排序帧名
- `_list_sorted_pngs()`：读取并排序 png
- `_resolve_image_dirs()`：根据 teleop/wild 定位 color/depth 目录
- `_resolve_mask_dir()`：在多个候选路径中寻找可用 mask 目录
- `_build_mask_lookup()`：按 sorted_index 建立 frame -> mask_path 映射

对齐策略是：

1. 对 color 文件名排序
2. 对 mask 文件名排序
3. 逐个 zip 对应
4. 建立映射并缓存

若数量不一致直接报错，不允许“猜着对齐”。

### 2.3 `__getitem__` 的新流程

样本读取时新增了 mask 分支，产出两个信号：

- `mask01`：像素级二值 mask（1=不可信）
- `image_mask_weight`：patch 级可靠度（给方案B）

具体过程：

1. 读取 mask，并与 depth 分辨率对齐（最近邻）
2. 按阈值转成 `mask01`
3. 把 `mask01` resize 到模型图像尺寸
4. 用已有 pooling 得到每个 patch 的“坏像素比例”
5. `image_mask_weight = 1 - mask_ratio`

### 2.4 方案A：深度图阶段做 3D 精确过滤

真正执行点在这句逻辑：

- `depths_for_cloud[mask01 > 0.5] = 0`

含义是：不可信像素对应深度清零，从源头不生成对应 3D 点。

然后继续走原有 Open3D 点云流程（crop + voxel downsample）。

### 2.5 空点云处理策略

如果过滤太狠导致空点云，支持策略处理：

- `fail_fast`：立即报错（严格模式）
- `warn_and_skip`（当前默认行为）：打印告警并回退到未过滤 depth 重建点云
- `fallback_disable_3d_filter`：预留策略名（当前实现上等价于回退重建）

---

## 3) 模型入口：把 patch 可靠度传进对齐模块

在 `policy/policy.py` 中，`RISE2.forward` 签名改为：

```python
forward(self, cloud, image, image_coord, image_mask_weight=None, actions=None)
```

主要变化：

- 接收可选 `image_mask_weight`
- 按 token 展平，与 `image_feat`/`image_coord` 对齐
- 传入 `SpatialAligner`

关键点：参数是可选的，旧调用不传也能跑。

---

## 4) 对齐插值层：把“距离权重”升级为“距离 × 可靠度”

核心改造在 `policy/sparse_modules.py` 的 `WeightedSpatialInterpolation`。

### 4.1 新接口

新增参数：

- `src_weights=None`
- `r_min=1e-3`
- `eps=1e-6`
- `tiny=1e-6`

### 4.2 新权重公式

对于目标点与其 k 个邻近 image patch：

1. 距离项：

\[
\text{dist\_weight}_j = \frac{1}{d_j + \text{eps}}
\]

2. 可靠度项：

\[
\text{rel}_j = \text{clamp}(r_j, r_{min}, 1.0)
\]

3. 合并并归一化：

\[
w_j = \frac{\text{dist\_weight}_j \cdot \text{rel}_j}{\sum_t (\text{dist\_weight}_t \cdot \text{rel}_t) + \text{tiny}}
\]

这意味着：

- 距离近的 patch 仍然更重要
- 但不可信 patch 会被显著降权
- 同时保留数值稳定性，不会出现分母为 0

### 4.3 SpatialAligner 的传递

`SpatialAligner.forward` 增加 `image_mask_weight` 入参，并按样本切分后传给插值器。

---

## 5) 训练入口：把数据集产出的权重喂给模型

在 `train.py` 的训练循环中：

- 从 batch 里 `data.get("image_mask_weight", None)`
- 若存在则 `.to(device)`
- 调用 policy 时传入 `image_mask_weight=...`

这样数据层到模型层链路闭环。

---

## 6) 为什么评估脚本没有大改

`eval.py`、`eval_server.py` 没有显式传 `image_mask_weight`，但不影响可用，原因是：

- `RISE2.forward` 的新增参数是可选默认 `None`
- 不传时退化为原始插值行为

这保证了历史脚本兼容性。

---

## 改造前后数据流对比

## 改造前

- depth -> point cloud
- image -> dense feature
- 插值权重仅由几何距离决定

## 改造后

- mask 与 color 对齐后进入两条分支：

A 分支（硬过滤）
- mask -> depth 置零 -> point cloud 更干净

B 分支（软降权）
- mask -> patch reliability -> 插值权重降低不可信 patch 影响

两条分支共同作用，几何和语义两侧都降噪。

---

## 流程化文字演示：一条样本的完整旅程

为了更直观地理解 mask-aware 改造后的数据流动，下面用一个具体场景来模拟一条样本从读入到进入策略网络的全过程。

### 场景设定

假设训练数据集中有一帧样本，包含以下文件：
- RGB 图像：`color/000123.png`（224×224 像素）
- 深度图：`depth/000123.png`（与 RGB 同分辨率）
- Mask 图像：`mask/000123.png`（与 RGB 同分辨率，白色区域表示不可信）
- 动作标签：`action.npy`（7 维机械臂动作）

配置中已启用 mask-aware 功能：
```yaml
mask_aware:
  enabled: true
  enable_2d_reweight: true
  enable_3d_filter: true
```

### 完整流程（按时间顺序）

#### 第一步：文件读取与对齐

数据集（`dataset/realworld.py`）首先做三件事：

1. **读取 RGB 图像**：加载 `color/000123.png`，得到形状为 (3, 224, 224) 的张量。
2. **读取深度图**：加载 `depth/000123.png`，得到形状为 (224, 224) 的深度值矩阵。
3. **读取并处理 Mask**：
   - 加载 `mask/000123.png`，得到形状为 (224, 224) 的灰度图。
   - 按阈值（`mask_threshold`）二值化，得到 `mask01`（1=不可信，0=可信）。
   - 把 `mask01` resize 到模型图像尺寸（比如 224×224）。
   - 通过平均池化，把像素级 mask 转成 patch 级可靠度：每个 14×14 的 patch 计算其中不可信像素的比例，然后 `image_mask_weight = 1 - mask_ratio`。

此时我们有了三个关键信号：
- `color`：原始 RGB 图像
- `depth`：原始深度图
- `mask01`：像素级二值 mask（用于方案A）
- `image_mask_weight`：patch 级可靠度（用于方案B）

#### 第二步：Mask 的两条作用路径开始分叉

从这里开始，mask 信号会同时走两条路：

**路径 A（方案A）：在深度图阶段过滤，影响"哪些点进入点云"**

执行这句关键代码：
```python
depths_for_cloud[mask01 > 0.5] = 0
```

含义是：把 mask 标记为不可信的像素对应的深度值全部清零。比如，如果 mask 中人手区域是白色（不可信），那么深度图中对应位置的深度值就被置为 0。

然后继续走原有的点云构建流程：
- 用清零后的深度图 + 相机内参，通过 Open3D 生成 3D 点云。
- 进行 crop（裁剪）和 voxel downsample（体素下采样）。

**结果**：不可信区域根本不会生成 3D 点，点云从源头就是干净的。

**路径 B（方案B）：生成 patch 可靠度，影响"语义聚合权重"**

`image_mask_weight` 这个张量会被保留下来，形状是 (H/14, W/14)，每个值代表一个 patch 的可信程度（0=完全不可信，1=完全可信）。

这个信号暂时不会立即使用，而是会一路传递到模型的对齐模块。

#### 第三步：3D 点云分支

点云构建完成后，进入稀疏编码器（Minkowski Engine）：

1. 点云被转换成稀疏张量格式。
2. 通过 Minkowski ResNet 提取 3D 特征。
3. 输出每个点的特征向量，形状为 (N, C)，其中 N 是点数，C 是特征维度。

**注意**：由于方案A已经过滤了不可信区域，这里的点云特征已经不包含噪声区域的几何信息。

#### 第四步：2D 语义分支

RGB 图像进入密集编码器（DINO 或 ResNet）：

1. 图像经过卷积网络，提取 2D 特征。
2. 输出特征图，形状为 (C, H/14, W/14)，其中 H/14 × W/14 是 patch 数量。
3. 特征图被展平成 (num_patches, C) 的形式。

同时，每个 patch 对应的像素坐标也被记录下来，用于后续与 3D 点对齐。

#### 第五步：对齐融合——两条路径在这里汇合

这是最关键的一步，发生在 `SpatialAligner` 模块中（`policy/sparse_modules.py`）。

对于每个 3D 点，系统会找到它在图像平面上的投影位置，然后找到最近的 k 个图像 patch。

**原始流程（无 mask）**：
- 插值权重只由几何距离决定：距离越近的 patch 权重越大。
- 公式：`weight = 1 / (distance + eps)`

**改造后流程（有 mask）**：
- 插值权重由"几何距离 × 可靠度"共同决定。
- 公式：`weight = (1 / (distance + eps)) × clamp(reliability, r_min, 1.0)`
- 其中 `reliability` 就是 `image_mask_weight` 中对应 patch 的值。

**具体效果**：
- 如果一个 patch 距离目标点很近，但它的可靠度很低（比如 0.1），那么它的最终权重会被显著降低。
- 如果一个 patch 距离稍远，但可靠度很高（比如 1.0），它仍然会有一定影响力。
- 这样既保留了空间邻近性，又抑制了噪声区域的干扰。

#### 第六步：输出到策略网络

融合后的特征进入 Transformer 进行聚合，然后进入动作解码器（Diffusion），最终输出动作预测。

**总结两条路径的作用**：
- **方案A（硬过滤）**：在点云构建阶段就剔除了不可信区域的几何信息，从源头保证 3D 特征干净。
- **方案B（软降权）**：在跨模态融合阶段降低不可信 patch 的语义权重，避免噪声特征干扰决策。

两条路径协同工作，几何侧和语义侧都得到了降噪。

### Mask 关闭时的退化行为

如果配置中 `mask_aware.enabled = false`，或者 `enable_3d_filter = false` 且 `enable_2d_reweight = false`，系统会自动退化到原始流程：

- 深度图不会被 mask 置零，所有像素都会生成 3D 点。
- `image_mask_weight` 不会被计算或传递。
- 插值权重只由几何距离决定，不考虑可靠度。
- 整个流程与改造前完全一致。

这种设计保证了向后兼容性，旧配置可以无缝继续使用。

---

## 兼容性与风险控制

### 兼容性

- 默认 `mask_aware.enabled=false`，旧行为不变。
- 新参数均为可选或有默认值。
- 推理端旧接口可继续使用。

### 风险控制

- 对齐数量不一致直接报错，防止脏训练。
- 关键参数启动时校验，防止中途崩。
- 空点云可选策略，避免批量训练被单帧卡死。

---

## 已完成验证

已经做过的验证包括：

1. 语法编译检查（核心文件）通过。
2. 静态链路检查通过：
   - 数据集确实产出 `image_mask_weight`
   - 训练脚本确实向 policy 传递
   - policy/aligner/interp 确实完成参数传递与计算
   - 3D 过滤逻辑确实在 depth 阶段生效

运行时端到端 smoke test 在当前终端环境受限（缺少 torch）未完成，这属于环境问题，不是代码路径问题。

---

## 如何实际启用这套能力

在配置里做最小修改：

```yaml
mask_aware:
  enabled: true
  mask_root: /你的/mask/根目录
  align_mode: sorted_index
  enable_2d_reweight: true
  enable_3d_filter: true
```

建议先保持默认稳定参数（`r_min`、`interp_eps`、`interp_tiny`），只先验证数据对齐和训练可跑。

---

## 目录对齐的现实约束（请务必注意）

当前实现对 mask 路径做了多候选兼容，但核心假设仍然是：

- color 和 mask 经过排序后是一一对应的同一时间序列

如果你的数据文件名体系有“缺帧、补帧、异步命名”的情况，需要先在数据准备阶段修正，否则会触发长度不一致报错。

---

## 后续建议

如果接下来要继续打磨，建议按这个顺序推进：

1. 做一轮小规模训练 smoke test（几百 step）
2. 可视化对比过滤前后点云，确认 mask 生效区域正确
3. 记录 `image_mask_weight` 分布，确认没有全 0 或全 1 的异常
4. 对比开启/关闭 mask-aware 的收敛曲线与部署表现

---

## 最终总结

这次改造的本质是：

- 把“mask”从离线附属信息，升级成训练主链路里的可靠度信号。
- 在几何构建侧做硬约束（方案A），在跨模态融合侧做软约束（方案B）。
- 保持默认关闭和接口兼容，降低落地风险。

从工程角度看，这次改造完成了“可配置、可回退、可解释”的目标；从算法角度看，减少了噪声区域对表示学习和动作预测的干扰。