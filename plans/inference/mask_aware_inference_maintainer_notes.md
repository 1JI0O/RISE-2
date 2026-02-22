# RISE-2 Mask-Aware 本地推理改造维护手册（面向维护 Agent）

> 文档定位：给后续接手 Agent 的“维护索引 + 不变量 + 回归检查”。
> 参考背景：[`plans/mask_aware_inference_human_summary.md`](plans/mask_aware_inference_human_summary.md:1)、[`plans/mask_aware_inference_spec_local_v1.md`](plans/mask_aware_inference_spec_local_v1.md:1)。

## 1) 目标与范围（In/Out of scope）

### In scope

- 本轮已落地的 mask-aware 推理改造仅在 [`eval.py`](eval.py:1)（本地分支）内。
- 覆盖内容：
  - 配置接入与默认值归一
  - `infer_mask` 占位接口
  - mask 规范化
  - 3D 深度阶段过滤
  - 空点云回退
  - 2D patch 降权构建
  - 本地前向传参
  - 异常收口与统计日志闭环

### Out of scope

- 远程推理链路（[`eval_server.py`](eval_server.py:1)、[`remote_eval/websocket_client_policy.py`](remote_eval/websocket_client_policy.py:1)、[`remote_eval/websocket_policy_server.py`](remote_eval/websocket_policy_server.py:1)）。
- 训练链路（[`train.py`](train.py:1)、[`dataset/realworld.py`](dataset/realworld.py:1)）。

---

## 2) 改动总览（文件与职责）

仅改动一个逻辑文件：[`eval.py`](eval.py:1)。

| 模块/函数 | 职责 | 已落地行为 |
|---|---|---|
| [`_build_mask_aware_cfg()`](eval.py:46) | 配置读取与默认值补齐 | 统一生成推理期 `mask_aware` 配置，非法值回落安全默认。|
| [`_log_mask_aware_summary()`](eval.py:99) | 启动摘要 | 输出一次生效策略摘要。|
| [`_log_mask_fallback()`](eval.py:114) | 回退日志统一格式 | 统一 `step/reason/action` 日志。|
| [`infer_mask()`](eval.py:119) | 推理 mask 接口占位 | 当前默认返回 `None`，供后续在线模型替换。|
| [`_normalize_mask01()`](eval.py:131) | mask 规范化 | 支持 shape 修复、最近邻 resize、阈值化、语义归一到 `float32` 0/1。|
| [`_safe_infer_mask()`](eval.py:166) | 异常收口 | 把异常映射为可回退原因码，避免异常炸穿主流程。|
| [`_build_image_mask_weight()`](eval.py:184) | 2D 降权构建 | `mask01 -> mask_ratio -> image_mask_weight=1-mask_ratio`，并 clamp。|
| [`evaluate()`](eval.py:312) | 主流程接线 | 本地分支完成 mask 获取、3D 过滤、空点云回退、2D 权重透传、统计输出。|

---

## 3) 接口/签名变化

本轮未新增跨文件接口，仅在 [`eval.py`](eval.py:1) 内新增/扩展 helper。

关键调用契约：

1. 推理期占位接口
   - [`infer_mask(color, depth, proprio, meta)`](eval.py:119) -> `mask01 | None`

2. 本地前向调用
   - [`policy(...)`](eval.py:527) 调用 [`RISE2.forward()`](policy/policy.py:52) 时传入可选参数 `image_mask_weight`。
   - 兼容性：`image_mask_weight=None` 时退化到旧行为。

---

## 4) 数据流与关键语义不变量

### 4.1 本地推理单帧数据流

1. 配置构建：[`config.mask_aware = _build_mask_aware_cfg(config)`](eval.py:323)。
2. mask 推断：[`_safe_infer_mask(...)`](eval.py:429) 调用 [`infer_mask()`](eval.py:119)。
3. mask 不可用时：按 [`infer_none_policy`](eval.py:440) 执行 `fail_fast` 或 `no_mask_fallback`。
4. 3D 分支：若可用则在深度阶段执行 [`depths_for_cloud[mask01 > 0.5] = 0`](eval.py:449)。
5. 点云构建：[`create_input(...)`](eval.py:460) 使用 `depths_for_cloud`。
6. 空点云/非法点保护：检查 [`points.shape[0] == 0`](eval.py:484) 与 [`np.isfinite(points)`](eval.py:467)，必要时回退到原始 `depths` 重建。
7. 图像坐标分支：始终使用原始深度 [`get_image_coordinates(depths, ...)`](eval.py:499)。
8. 2D 权重：[`_build_image_mask_weight()`](eval.py:505) 构建 patch 级 `image_mask_weight`。
9. 本地前向：传入 [`image_mask_weight`](eval.py:531)；失败或非法值时传 `None`。
10. 结束摘要：输出 [`mask_stats` 汇总](eval.py:583)。

### 4.2 维护必须守住的不变量

- 不变量 A：`mask01 == 1` 不可信，`mask01 == 0` 可信。
- 不变量 B：3D 过滤必须在深度阶段执行，不可改为点云后过滤。
- 不变量 C：图像坐标分支必须继续使用原始深度，不能用 `depths_for_cloud`。
- 不变量 D：2D 降权公式固定为 `image_mask_weight = 1 - mask_ratio`，并限制在 `[0,1]`。
- 不变量 E：默认回退策略是“不中断推理”（除非显式 `fail_fast`）。
- 不变量 F：异常必须可观测（统一 fallback 日志 + 结束统计摘要）。

---

## 5) 配置项与默认值（重点：mask_aware）

配置归一入口：[`_build_mask_aware_cfg()`](eval.py:46)。

| 键 | 默认值 | 维护语义 |
|---|---:|---|
| `enabled` | `false` | 总开关，关闭时应回归旧本地推理路径。|
| `enable_3d_filter` | `true` | 是否启用深度阶段 3D 过滤。|
| `enable_2d_reweight` | `true` | 是否启用 2D patch 降权。|
| `mask_threshold` | `0` | 二值化阈值。|
| `mask_white_is_untrusted` | `true` | 白色是否为不可信。|
| `r_min` | `1e-3` | 与插值稳定性语义保持一致（推理配置预留）。|
| `interp_eps` | `1e-6` | 与插值稳定性语义保持一致（推理配置预留）。|
| `interp_tiny` | `1e-6` | 与插值稳定性语义保持一致（推理配置预留）。|
| `infer_allow_none` | `true` | 允许 `infer_mask` 返回 `None`。|
| `infer_none_policy` | `no_mask_fallback` | `infer_mask` 不可用时策略。|
| `empty_cloud_policy` | `warn_and_skip_filter` | 空点云默认回退策略。|

策略合法值白名单：
- `infer_none_policy` ∈ `{no_mask_fallback, fail_fast}`（见 [`valid_none_policy`](eval.py:69)）
- `empty_cloud_policy` ∈ `{warn_and_skip_filter, fail_fast}`（见 [`valid_empty_cloud_policy`](eval.py:70)）

---

## 6) 异常与回退矩阵（默认策略 vs fail_fast）

| 场景 | 默认行为 | fail_fast 行为 |
|---|---|---|
| `infer_mask` 返回 `None` | 记录 `infer_none`，禁用本帧 mask 分支继续 | 立即中断 |
| `infer_mask` 抛异常 | 记录 `infer_exception`，禁用本帧 mask 分支继续 | 立即中断 |
| mask shape/内容非法 | 记录 `mask_invalid`，禁用本帧 mask 分支继续 | 立即中断 |
| 3D 过滤后空点云 | 记录 `empty_cloud_skip`，回退原始 depth 重建 | 立即中断 |
| 点云含 NaN/Inf | 记录 `points_nonfinite`，回退原始 depth 重建 | 立即中断 |
| 2D 权重构建失败 | 记录 `reweight_fallback`，`image_mask_weight=None` | 同左（不中断） |
| 2D 权重含 NaN/Inf | 记录 `weight_nonfinite`，`image_mask_weight=None` | 同左（不中断） |

统计计数器定义见 [`mask_stats`](eval.py:403)。

---

## 7) 维护时最容易踩坑的点

1. “本地改造”边界不要破坏。
   - 本轮强约束是仅 [`eval.py`](eval.py:1)；不要把逻辑扩散到远程路径。
2. 不要改动 `mask01` 语义方向。
   - 语义颠倒会导致 3D 过滤和 2D 降权同时失真。
3. 不要让 `image_coords` 用过滤后深度。
   - 当前明确要求图像坐标继续由原始深度计算，见 [`image_coords = ...get_image_coordinates(depths, ...)`](eval.py:499)。
4. `fail_fast` 仅用于显式策略场景。
   - 默认必须保活，不能因单帧 mask 问题使 rollout 中断。
5. 2D 分支是“增强项”。
   - 构建失败时应退化 `None`，不能反向拖垮主流程。
6. `infer_mask` 尚为占位。
   - 后续接真实模型时，优先复用 [`_safe_infer_mask()`](eval.py:166) 与 [`_normalize_mask01()`](eval.py:131) 约束，不要绕开。

---

## 8) 最小回归检查清单（可执行）

### A. 语法检查

```bash
python -m py_compile eval.py
```

### B. 关键语义静态检查

```bash
grep -n "_build_mask_aware_cfg\|infer_none_policy\|empty_cloud_policy" eval.py
grep -n "depths_for_cloud\[mask01 > 0.5\]" eval.py
grep -n "get_image_coordinates(depths" eval.py
grep -n "image_mask_weight = \(1.0 - mask_ratio\)" eval.py
grep -n "_log_mask_fallback\|mask_stats\|summary" eval.py
```

### C. 最小行为检查（建议顺序）

1. 开关回归：`mask_aware.enabled=false`，确认本地推理路径可运行。
2. 占位回退：保持 [`infer_mask()`](eval.py:119) 返回 `None`，确认不中断并有 fallback 日志。
3. `fail_fast` 验证：将 `infer_none_policy` 设为 `fail_fast`，确认 `infer_none` 触发中断。
4. 空点云回退：注入高占比 mask（或临时 mock），确认默认回退到原始 depth 重建。
5. 2D 降权回退：制造 `_build_image_mask_weight` 失败场景，确认 `image_mask_weight=None` 且流程继续。

---

## 9) 后续扩展建议（按优先级）

### P0（优先）

1. 实现真实在线 mask 推理：替换 [`infer_mask()`](eval.py:119) 占位实现，保持返回契约不变。
2. 增加本地 E2E 回归脚本：覆盖 `enabled=false`、`no_mask_fallback`、`fail_fast`、空点云回退四个最小场景。

### P1（中优先）

1. 将 `masked_ratio/cloud_points_before_after` 作为可选帧级指标输出（默认限流关闭）。
2. 为 `_normalize_mask01` 和 `_safe_infer_mask` 增加单元测试（shape 修复、异常路径、阈值语义）。

### P2（次优先）

1. 评估是否在远程协议中引入 mask 字段并做对齐改造（独立里程碑，不并入当前文档范围）。
2. 加入在线 mask 时序平滑与稳定性策略（例如短窗滤波、置信度门控）。
