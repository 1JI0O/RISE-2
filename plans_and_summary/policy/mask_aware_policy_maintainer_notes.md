# RISE-2 Mask-Aware 训练改造维护手册（面向维护 Agent）

> 文档定位：给后续接手 Agent 的“维护索引 + 不变量 + 回归检查”。
> 参考背景：[`plans/mask_aware_policy_human_summary.md`](plans/mask_aware_policy_human_summary.md:1)。

## 1) 目标与范围（In/Out of scope）

### In scope

- 训练链路中 mask-aware 相关改造与维护点，覆盖：
  - [`dataset/realworld.py`](dataset/realworld.py:1)
  - [`policy/policy.py`](policy/policy.py:1)
  - [`policy/sparse_modules.py`](policy/sparse_modules.py:1)
  - [`train.py`](train.py:1)
  - [`configs/test.yaml`](configs/test.yaml:1)
  - [`configs/single_rise1.yaml`](configs/single_rise1.yaml:1)
- 重点维护对象：配置读取、数据集对齐、3D 过滤、2D 降权、前向接线、异常回退。

### Out of scope

- 本地推理链路维护（见 [`eval.py`](eval.py:1) 的独立文档）。
- 远程推理链路维护（[`eval_server.py`](eval_server.py:1)、[`remote_eval/websocket_client_policy.py`](remote_eval/websocket_client_policy.py:1)、[`remote_eval/websocket_policy_server.py`](remote_eval/websocket_policy_server.py:1)）。

---

## 2) 改动总览（文件与职责）

| 文件 | 职责 | 本次落地点 |
|---|---|---|
| [`dataset/realworld.py`](dataset/realworld.py:1) | 训练样本构建主入口 | 增加 `mask_aware` 解析与校验、color-mask 对齐映射、`mask01` 构建、`image_mask_weight` 构建、深度阶段 3D 过滤、空点云回退。|
| [`policy/policy.py`](policy/policy.py:1) | 策略前向入口 | [`RISE2.forward()`](policy/policy.py:52) 新增可选入参 `image_mask_weight` 并传入对齐器。|
| [`policy/sparse_modules.py`](policy/sparse_modules.py:1) | 2D-3D 对齐插值 | [`WeightedSpatialInterpolation.forward()`](policy/sparse_modules.py:72) 支持 `src_weights` + 稳定项；[`SpatialAligner.forward()`](policy/sparse_modules.py:196) 增加样本级权重透传。|
| [`train.py`](train.py:1) | 训练循环 | 从 batch 读取 `image_mask_weight`，并在 policy 调用时透传（[`train()`](train.py:29) 内调用点见 [`loss = policy(...)`](train.py:160)）。|
| [`configs/test.yaml`](configs/test.yaml:81) | 测试配置模板 | 增加 `mask_aware` 块及默认值。|
| [`configs/single_rise1.yaml`](configs/single_rise1.yaml:81) | 单臂配置模板 | 增加 `mask_aware` 块及默认值。|

补充：batch 聚合键位在 [`TO_TENSOR_KEYS`](dataset/realworld.py:18) 和 [`collate_fn()`](dataset/realworld.py:571) 中已纳入 `mask01`、`image_mask_weight`。

---

## 3) 接口/签名变化

1. 策略前向签名变化
   - [`RISE2.forward()`](policy/policy.py:52)
   - 新增可选参数：`image_mask_weight=None`。
   - 兼容性：不传该参数时退化到旧行为。

2. 对齐模块签名变化
   - [`SpatialAligner.forward()`](policy/sparse_modules.py:196)
   - 新增可选参数：`image_mask_weight=None`。

3. 插值层签名变化
   - [`WeightedSpatialInterpolation.forward()`](policy/sparse_modules.py:72)
   - 新增参数：`src_weights=None`、`r_min=1e-3`、`eps=1e-6`、`tiny=1e-6`。

4. 数据字典输出变化
   - [`RealWorldDataset.__getitem__()`](dataset/realworld.py:407) 在 mask-aware 打开时额外返回：`mask01`、`mask_path`、`image_mask_weight`。

---

## 4) 数据流与关键语义不变量

### 4.1 数据流（训练单样本）

1. 配置解析：[`RealWorldDataset._parse_config()`](dataset/realworld.py:269) 读取 `mask_aware`。
2. 启动校验：[`RealWorldDataset._validate_mask_aware_config()`](dataset/realworld.py:237) 执行强校验。
3. 对齐映射：[`RealWorldDataset._build_mask_lookup()`](dataset/realworld.py:198) 以 `sorted_index` 建立 color->mask 映射。
4. 样本读取：[`RealWorldDataset.__getitem__()`](dataset/realworld.py:407) 读取 color/depth/mask。
5. 二值化语义：`mask01` 在 [`mask01 = (mask_img > threshold)` 逻辑](dataset/realworld.py:442) 下输出（取决于 `mask_white_is_untrusted`）。
6. 3D 分支：深度阶段过滤见 [`depths_for_cloud[mask01 > 0.5] = 0.0`](dataset/realworld.py:456)，再进入 [`load_point_cloud()`](dataset/realworld.py:375)。
7. 2D 分支：patch 降权见 [`image_mask_weight = 1.0 - mask_ratio`](dataset/realworld.py:450)。
8. 训练前向：[`train()`](train.py:29) 中读取并透传 [`image_mask_weight`](train.py:156) 到 [`policy(...)`](train.py:160)。
9. 插值融合：[`WeightedSpatialInterpolation.forward()`](policy/sparse_modules.py:72) 计算距离权重与可靠度权重乘积后归一化。

### 4.2 维护必须守住的不变量

- 不变量 A：`mask01 == 1` 表示不可信，`mask01 == 0` 表示可信（训练/推理保持一致）。
- 不变量 B：3D 过滤必须发生在深度图阶段，不可后移到点云阶段。
- 不变量 C：2D 降权公式固定为 `image_mask_weight = 1 - mask_ratio`，并且权重语义是“越大越可信”。
- 不变量 D：插值必须支持 `src_weights`，并保留稳定项（`r_min` 下限、`eps` 距离稳定项、`tiny` 归一化稳定项），实现见 [`dist_weight`](policy/sparse_modules.py:103)、[`torch.clamp(..., min=r_min)`](policy/sparse_modules.py:107)、[`norm + tiny`](policy/sparse_modules.py:112)。
- 不变量 E：对齐模式仅允许 `sorted_index`，且 color/mask 数量必须一致，不允许猜测式对齐。

---

## 5) 配置项与默认值（重点：mask_aware）

配置入口：[`mask_aware`](configs/test.yaml:81)、[`mask_aware`](configs/single_rise1.yaml:81)。
解析实现：[`RealWorldDataset._parse_config()`](dataset/realworld.py:269)。

| 键 | 默认值 | 维护语义 |
|---|---:|---|
| `enabled` | `false` | 总开关，关闭时应回归旧路径。|
| `mask_root` | `null` | 开启时必填且目录必须存在。|
| `align_mode` | `sorted_index` | 当前唯一支持模式。|
| `enable_2d_reweight` | `true` | 2D 降权分支。|
| `enable_3d_filter` | `true` | 深度阶段 3D 过滤分支。|
| `mask_threshold` | `0` | 二值阈值。|
| `mask_white_is_untrusted` | `true` | 白色是否表示不可信。|
| `r_min` | `0.001` | 可靠度下限（校验层读取）。|
| `interp_eps` | `0.000001` | 距离稳定项（校验层读取）。|
| `interp_tiny` | `0.000001` | 归一化稳定项（校验层读取）。|
| `empty_cloud_policy` | `warn_and_skip` | 空点云处理策略。|
| `log_alignment_preview` | `true` | 是否打印 color-mask 对齐预览。|

注意：当前训练代码中 `r_min`/`interp_eps`/`interp_tiny` 在数据集层会读取并校验，但插值层运行值来自 [`WeightedSpatialInterpolation.forward()`](policy/sparse_modules.py:72) 的函数默认参数；若你要让配置值“真实驱动插值”，需要新增显式传参接线。

---

## 6) 异常与回退矩阵（默认策略 vs fail_fast）

| 场景 | 默认行为 | fail_fast 行为 |
|---|---|---|
| `mask_aware.enabled=false` | 完全走旧路径 | 同左 |
| `mask_root` 缺失/不存在 | 启动时报错（阻断） | 同左 |
| `align_mode` 非 `sorted_index` | 启动时报错（阻断） | 同左 |
| `enable_2d_reweight=false` 或 `enable_3d_filter=false` 且 `enabled=true` | 启动时报错（阻断） | 同左 |
| color/mask 数量不一致 | 构建映射时报错（阻断） | 同左 |
| 样本缺失对齐 mask | `KeyError`（阻断） | 同左 |
| mask 与 depth 尺寸不一致 | 最近邻 resize 后继续 | 同左 |
| 3D 过滤后空点云 | 默认 `warn_and_skip`：打印告警并用原始 depth 重建 | `fail_fast`：立即抛错 |
| 回退重建后仍空点云 | 抛错（阻断） | 同左 |

补充：`fallback_disable_3d_filter` 在当前实现中与默认回退路径行为等价（都会回退到原始 depth 重建）。

---

## 7) 维护时最容易踩坑的点

1. 对齐策略不能“看起来差不多就行”。
   - 必须保持 [`_build_mask_lookup()`](dataset/realworld.py:198) 的严格一一对齐语义。
2. 不要改动 `mask01` 语义方向。
   - 一旦把 `1/0` 含义反过来，会同时污染 3D 过滤与 2D 降权。
3. 不要把 3D 过滤挪到点云之后。
   - 过滤位置必须保持在 [`depths_for_cloud[mask01 > 0.5] = 0.0`](dataset/realworld.py:456)。
4. `mask_aware.enabled=true` 下要求 2D/3D 两分支都可用。
   - 这是当前校验硬约束，见 [`_validate_mask_aware_config()`](dataset/realworld.py:247)。
5. 配置项与实际执行值可能“同名不同源”。
   - 当前 `r_min/eps/tiny` 的真实插值执行值默认来自 [`WeightedSpatialInterpolation.forward()`](policy/sparse_modules.py:72) 的默认参数。
6. batch 维度修改需同步多处。
   - `image_mask_weight` 形状改动会影响 [`collate_fn()`](dataset/realworld.py:571)、[`RISE2.forward()`](policy/policy.py:52)、[`SpatialAligner.forward()`](policy/sparse_modules.py:196)。

---

## 8) 最小回归检查清单（可执行）

### A. 语法与导入检查

```bash
python -m py_compile dataset/realworld.py policy/policy.py policy/sparse_modules.py train.py
```

### B. 关键链路静态检查

```bash
grep -n "mask_aware" configs/test.yaml configs/single_rise1.yaml
grep -n "image_mask_weight" dataset/realworld.py train.py policy/policy.py policy/sparse_modules.py
grep -n "depths_for_cloud\[mask01 > 0.5\]" dataset/realworld.py
```

### C. 行为检查（最小步骤）

1. 关闭开关：`mask_aware.enabled=false`，确认训练可正常进入迭代（旧路径回归）。
2. 打开开关：`mask_aware.enabled=true` 且提供有效 `mask_root`，确认：
   - 启动阶段可看到 [`[mask-align]` 日志](dataset/realworld.py:224)；
   - batch 中存在 `image_mask_weight`（读取点见 [`data.get("image_mask_weight")`](train.py:156)）。
3. 构造高遮挡 mask，确认空点云策略：
   - 默认 `warn_and_skip` 不中断；
   - `fail_fast` 触发异常中断。

---

## 9) 后续扩展建议（按优先级）

### P0（优先）

1. 把配置中的 `r_min`/`interp_eps`/`interp_tiny` 显式接入 [`SpatialAligner.forward()`](policy/sparse_modules.py:196) -> [`WeightedSpatialInterpolation.forward()`](policy/sparse_modules.py:72)，避免“配置修改无效果”。
2. 为 [`_build_mask_lookup()`](dataset/realworld.py:198) 增加最小单元测试（数量不一致、命名异常、非数字帧名）。

### P1（中优先）

1. 将空点云策略分支细化为真正不同语义（当前 `fallback_disable_3d_filter` 与默认路径等价）。
2. 增加 `masked_ratio` 的训练期统计（按 batch 输出分位数），用于快速发现 mask 质量异常。

### P2（次优先）

1. 若未来需要，可考虑放宽“2D/3D 必须同时开启”的校验约束，并补齐对应回归集。
2. 将训练与推理的 mask-aware 配置键定义统一化，减少维护心智负担。
