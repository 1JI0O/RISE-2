# RISE-2 本地推理 Mask-Aware 改造总结（面向人类阅读）

## 这份文档是干什么的

这份文档是“实现复盘”，不是执行规范，也不是新需求。

目标是让研发同学、算法同学、维护同学快速理解这次本地推理改造：

- 为什么要改
- 改动范围在哪里
- 每一步具体改了什么
- 出现异常时系统会怎么处理
- 当前可用性和后续建议是什么

---

## 一句话结论

这次改造把 **Mask-Aware 能力完整接入本地推理**，并且严格限制在 `eval.py`：

1. **3D 硬过滤**：在深度图阶段把不可信区域置零，再构建点云。
2. **2D 软降权**：把 mask 转成 patch 级可靠度 `image_mask_weight`，在本地前向传给策略网络。
3. **异常可回退**：默认“不中断推理”，只在显式 `fail_fast` 时中断。
4. **可观测**：启动摘要、帧级回退日志、结束统计摘要都已具备。

---

## 背景：改造前的问题

本地推理在改造前可以跑通 RGB-D -> 点云/图像 -> policy 前向，但缺少 mask-aware 机制，存在两类问题：

- 深度图中的不可信区域（遮挡、反光、伪深度）会直接进入点云。
- 图像 token 融合时，所有 patch 被默认同等可信，噪声 patch 也会参与融合。

这会导致：

- 3D 几何噪声干扰下游动作预测。
- 2D-3D 融合权重对坏 patch 不敏感。

---

## 本次改造范围（非常重要）

本次只做 **本地推理**，并且只改一个文件：

- `eval.py`

明确不在本次范围内：

- 远程推理链路（`eval_server.py`、`remote_eval/*`）
- 训练链路（`train.py`、`dataset/realworld.py`）

---

## 关键语义约定（与训练侧保持一致）

- `mask01 == 1`：不可信区域
- `mask01 == 0`：可信区域

核心语义：

- 3D 过滤必须在 **深度图阶段** 执行
- 图像坐标 `image_coords` 必须继续由 **原始深度图** 计算
- 2D 降权公式固定为：
  - `image_mask_weight = 1 - mask_ratio`
  - 并限制在 `[0, 1]`

---

## 改造前后数据流对比

### 改造前

1. 采集 `colors/depths`
2. 用 `depths` 直接构建点云
3. 用 `depths` 计算 `image_coords`
4. 本地前向 `policy(cloud, image, image_coords)`

### 改造后

1. 采集 `colors/depths`
2. 调用 `infer_mask(...)` 获取 `mask01 | None`
3. 若可用且开启 3D 过滤：
   - `depths_for_cloud[mask01 > 0.5] = 0`
4. 点云分支使用 `depths_for_cloud`（必要时回退原始 `depths`）
5. 图像坐标分支仍使用原始 `depths`
6. 若可用且开启 2D 降权：构建 `image_mask_weight`
7. 本地前向 `policy(cloud, image, image_coords, image_mask_weight=...)`

---

## 阶段化实施复盘

## 阶段1：配置契约接入与启动摘要

目标：让推理期拥有稳定的 `mask_aware` 配置读取与默认值。

完成内容：

- 新增配置构建逻辑，统一补齐默认值（如 `enabled`、`infer_none_policy`、`empty_cloud_policy` 等）。
- 配置非法时自动降级到安全默认（禁用 mask-aware）。
- 启动时打印一次摘要，便于排查当前生效策略。

收益：

- 即使配置缺字段也不会崩。
- 可以一眼看出当前是否启用、走什么策略。

---

## 阶段2：推理期 mask 接口与规范化

目标：把任意来源的 mask 输出规范成推理可用格式。

完成内容：

- 新增 `infer_mask(color, depth, proprio, meta)` 占位接口（当前默认返回 `None`）。
- 新增 mask 标准化流程：
  - 输入类型统一为 numpy
  - 允许 `(H,W)` 或单通道 3D 形状，其他形状判非法
  - 尺寸不一致时最近邻 resize 到 depth 尺寸
  - 按阈值转成 `float32` 的 0/1
- 新增安全包装，把异常收口为原因码，不让异常直接炸穿主流程。

收益：

- 后续替换真实 mask 模型时，只需实现 `infer_mask`。
- 管线内部只处理统一语义的 `mask01`。

---

## 阶段3：3D 深度阶段过滤 + 空点云回退

目标：把“2D mask -> 3D 过滤”落在正确位置，并保证可回退。

完成内容：

- 在点云构建前生成 `depths_for_cloud`。
- 开启时执行：`depths_for_cloud[mask01 > 0.5] = 0`。
- 点云分支使用 `depths_for_cloud`；图像分支保持原始 `depths`。
- 若过滤后空点云：
  - `fail_fast`：立即中断
  - 默认：告警并回退到原始深度重建

收益：

- 3D 过滤语义与训练侧一致。
- 极端遮挡场景不会默认崩溃。

---

## 阶段4：2D 降权与本地前向接线

目标：把 patch 级可靠度真正传进本地策略前向。

完成内容：

- 新增 `image_mask_weight` 构建逻辑：
  1. `mask01` -> tensor
  2. resize 到图像分支尺寸（最近邻）
  3. `image_coord_pooling` 得 `mask_ratio`
  4. `image_mask_weight = 1 - mask_ratio`
  5. clamp 到 `[0,1]`
- 在本地分支调用 policy 时传入 `image_mask_weight`。
- 不满足条件或构建失败时回退为 `None`。

收益：

- 2D 降权链路闭环，已与 `RISE2.forward` 的可选参数对接。
- 不影响旧调用（不传时仍兼容）。

---

## 阶段5：异常矩阵收口、日志闭环、数值护栏

目标：让异常处理从“多处分散”变成“可预测、可观察、可回退”。

完成内容：

1. 统一回退日志格式
- 增加统一日志函数，输出 `step + reason + action`。

2. 统一原因码统计
- 对 infer 失败、mask 非法、空点云、权重构建失败、数值异常等情况进行计数。

3. 数值稳定护栏
- 点云 `points` 非有限值（NaN/Inf）时：
  - `fail_fast`：中断
  - 默认：回退原始深度重建
- `image_mask_weight` 非有限值时：
  - 置 `None`，走旧前向

4. 结束摘要
- rollout 结束后输出一次总览统计，帮助判断整段运行稳定性。

收益：

- 异常不再 silent failure。
- 默认策略“保活优先”得到完整落实。

---

## 逐帧流程（人话版）

每一帧在本地推理大致按这个顺序执行：

1. 读观测：`colors/depths`
2. 尝试 `infer_mask`
3. 成功则规范化 `mask01`，失败则产生回退原因
4. 生成 `depths_for_cloud`（可选 3D 过滤）
5. 构建点云（必要时回退）
6. 用原始 `depths` 计算图像坐标
7. 计算图像特征并可选构建 `image_mask_weight`
8. 本地前向，输出动作
9. 记录统计，进入下一帧

---

## 流程化文字演示：一条样本的完整旅程

为了更直观理解本次“本地推理”改造，下面用一个实际可对应到当前代码路径的例子，演示一帧数据如何从观测进入策略网络。

### 场景设定

假设我们正在本地模式下跑 [`evaluate()`](eval.py:315)，并且当前配置如下：

```yaml
mask_aware:
  enabled: true
  enable_3d_filter: true
  enable_2d_reweight: true
  infer_none_policy: no_mask_fallback
  empty_cloud_policy: warn_and_skip_filter
```

当前帧观测来自机器人或测试输入，记为：

- `colors`：一张 RGB 图，shape 约为 `(H, W, 3)`
- `depths`：一张深度图，shape 为 `(H, W)`
- `t=42`：当前 rollout step

### 完整流程（按执行顺序）

#### 第一步：拿到 mask（或拿不到）

在 [`evaluate()`](eval.py:315) 每帧会先调用 [`infer_mask()`](eval.py:119)（经由 [`_safe_infer_mask()`](eval.py:166) 包装）：

- 如果拿到合法 mask，进入下一步。
- 如果返回 `None` 或抛异常：
  - 当前配置是 `no_mask_fallback`，所以本帧继续跑，但不做 3D 过滤和 2D 降权。
  - 同时会记录原因并打印一条统一回退日志。

假设本例里拿到了合法 mask，规范化后得到 `mask01`：

- `mask01 == 1` 的区域代表不可信（例如手臂遮挡区域）
- `mask01 == 0` 的区域代表可信

#### 第二步：方案A 生效，在深度图阶段做硬过滤

如果开启了 3D 过滤，会先构造 `depths_for_cloud`，然后执行：

```python
depths_for_cloud[mask01 > 0.5] = 0
```

这一步发生在点云生成前，含义是“把不可信像素的深度清零，让这些像素不再生成 3D 点”。

随后调用 [`create_input()`](eval.py:259) 用 `depths_for_cloud` 构建点云输入。

#### 第三步：空点云和非法数值保护

点云构建后会做两类保护：

1. 点云数值检查（NaN/Inf）
- 若出现非有限值：
  - `fail_fast` 配置下直接中断
  - 默认策略下回退到原始 `depths` 重建点云

2. 空点云检查
- 若过滤后点数为 0：
  - `fail_fast` 下中断
  - 默认 `warn_and_skip_filter` 下回退到原始 `depths` 重建

所以本例即使 mask 过强，也不会默认把整个 rollout 直接跑崩。

#### 第四步：图像分支保持“原始深度”语义

无论 3D 过滤是否发生，图像坐标都仍使用原始深度计算：

- 调用 [`ImageProcessor.get_image_coordinates()`](dataset/data_utils.py:243)
- 输入仍是原始 `depths`，不是 `depths_for_cloud`

这保证与 spec 约束一致：3D 过滤只影响点云分支，不改变图像几何坐标语义。

#### 第五步：方案B 生效，构建 patch 级可靠度

如果 `mask01` 可用且开启 2D 降权，会调用 [`_build_image_mask_weight()`](eval.py:186)：

1. `mask01` 转 tensor
2. 最近邻 resize 到图像分支尺寸
3. 经过 `image_coord_pooling` 得到 patch 的 `mask_ratio`
4. 计算 `image_mask_weight = 1 - mask_ratio`
5. clamp 到 `[0,1]`

举个直观例子：

- 某 patch 内 80% 像素被 mask 标成不可信
- 那么该 patch 的 `mask_ratio=0.8`
- 对应 `image_mask_weight=0.2`

这表示该 patch 仍可参与，但权重会明显降低。

#### 第六步：进入本地 policy 前向

在本地分支调用 [`RISE2.forward()`](policy/policy.py:52) 时，会把 `image_mask_weight` 作为可选参数传入：

- 有效时传真实权重
- 构建失败或非有限值时传 `None`，自动退化到旧行为

最终输出动作预测，不影响原有调用兼容性。

### 这个实例体现了什么

这条链路说明了两件事：

1. **几何侧降噪**：方案A 在深度阶段剔除不可信区域，减少坏点进入点云。
2. **语义侧降噪**：方案B 在融合前降低坏 patch 影响，减少噪声语义干扰。

并且在异常场景下默认“保活优先”，不会因为单帧 mask 问题导致整个推理流程中断。

---

## 异常处理矩阵（简化版）

| 场景 | 默认行为 | fail_fast 行为 |
|---|---|---|
| `infer_mask` 返回 `None` | 关闭本帧 mask 分支继续 | 立即中断 |
| `infer_mask` 抛异常 | 关闭本帧 mask 分支继续 | 立即中断 |
| mask 输出非法 | 关闭本帧 mask 分支继续 | 立即中断 |
| 3D 过滤后空点云 | 回退原始深度重建 | 立即中断 |
| 点云出现 NaN/Inf | 回退原始深度重建 | 立即中断 |
| 2D 权重构建失败 | `image_mask_weight=None` | `image_mask_weight=None` |
| 2D 权重出现 NaN/Inf | `image_mask_weight=None` | `image_mask_weight=None` |

说明：2D 权重失败不会直接中断，因为它是“增强项”，回退到旧路径即可。

---

## 与训练侧实现的关系

这次做的是 **推理侧补齐**，不是重复造轮子：

- 训练侧早已有 patch 可靠度语义与前向接口。
- 推理侧现在补齐了同语义的输入构建与传参。
- 因此训练-推理在 mask-aware 语义上已对齐。

---

## 兼容性与风险控制

## 兼容性

- `mask_aware.enabled=false` 时，应退化为旧行为。
- 本次不改远程协议，不影响远程链路。
- 本次不改训练脚本，不影响既有训练任务。

## 风险控制

- mask 尺寸不一致：最近邻对齐并告警。
- 强遮挡导致空点云：默认回退，不崩溃。
- 非有限值污染：检测后回退，避免把坏值送入网络。

---

## 已完成验证

- 语法检查：`python -m py_compile eval.py` 通过。
- 代码路径检查：新增分支均在 `eval.py` 本地链路内。
- 审查结论：阶段1-5已完成并通过阶段验收。

---

## 如何接入真实在线 mask（给后续同学）

当前 `infer_mask` 是占位函数，后续只需在一个位置替换即可：

- 在 `infer_mask(color, depth, proprio, meta)` 内接入真实模型/规则。

建议约束：

1. 返回 `None` 表示本帧不可用（允许回退）。
2. 返回 mask 时尽量给单通道二维数组。
3. 语义保持 `1=不可信，0=可信`，避免与现有分支冲突。

---

## 当前仍未覆盖的内容

以下内容有意不在本轮：

- 远程推理 mask 字段扩展与协议改造
- `eval_server.py` 与 websocket 客户端/服务端联动
- 在线 mask 时序平滑与稳定性优化

这些建议作为后续独立阶段推进。

---

## 总结

这次本地推理改造的核心价值不是“多了几个 if 分支”，而是把 mask-aware 从“训练期能力”扩展成了“推理期可运行能力”：

- 语义一致
- 回退可控
- 日志可查
- 兼容旧链路

从工程角度看，当前版本已经满足“可上线试跑”的最低闭环条件，并为后续接入真实在线 mask 模块预留了稳定接口。