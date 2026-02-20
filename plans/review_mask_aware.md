# Mask-Aware 改造代码审查报告

> 审查范围：mask-aware 功能改造涉及的新增/修改代码。
> 参考规范：[plans/mask_aware_policy_spec_v4.md](mask_aware_policy_spec_v4.md)、[plans/mask_aware_inference_maintainer_notes.md](mask_aware_inference_maintainer_notes.md)、[plans/mask_aware_training_maintainer_notes.md](mask_aware_training_maintainer_notes.md)、[plans/AGENTS.md](AGENTS.md)

---

## 一、潜在 Bug

### 1.1 配置中的 `r_min`/`eps`/`tiny` 不驱动实际插值行为

**严重程度**：高（与 maintainer notes §9 P0 一致）

**涉及文件**：[dataset/realworld.py:317-319](../dataset/realworld.py#L317)、[policy/sparse_modules.py:72](../policy/sparse_modules.py#L72)

`_parse_config` 读取并校验了 `mask_aware.r_min`、`interp_eps`、`interp_tiny`，但 `WeightedSpatialInterpolation.forward()` 使用的是函数签名中的硬编码默认值：

```python
def forward(self, tgt, src, tgt_feats, src_feats, src_weights=None, k=3, r_min=1e-3, eps=1e-6, tiny=1e-6):
```

`SpatialAligner.forward()` 调用 `self.interp(...)` 时未显式传入这三个参数，导致修改 YAML 配置不影响实际插值行为。这既是 Bug，也使配置字段形同虚设。

**修改方案**：在 `SpatialAligner.__init__` 中存储这三个参数，并在调用 `self.interp(...)` 时显式传入。

---

### 1.2 `mask_stats` 统计键与日志 reason 不一致

**严重程度**：低（不影响运行，影响可追溯性）

**涉及文件**：[eval.py:510-511](../eval.py#L510)

```python
mask_stats["weight_nonfinite"] += 1
_log_mask_fallback(t, "reweight_nonfinite", "disable_2d_reweight")
```

统计键为 `"weight_nonfinite"`，但传入 `_log_mask_fallback` 的 reason 是 `"reweight_nonfinite"`。两个字符串不同，导致日志和汇总摘要无法对照，排查时产生困惑。

**修改方案**：统一为同一字符串，建议统一为 `"weight_nonfinite"`。

---

### 1.3 异常被完全吞没，无可观测信息

**严重程度**：中（违反 maintainer notes §4.2 不变量 F"异常必须可观测"）

**涉及文件**：[eval.py:195](../eval.py#L195)、[eval.py:171](../eval.py#L171)

`_build_image_mask_weight` 捕获所有异常后直接返回 `None`，未打印任何异常信息：

```python
except Exception:
    return None
```

`_safe_infer_mask` 仅记录原因码字符串，原始异常堆栈丢失：

```python
except Exception:
    return None, "infer_exception"
```

当推理期 mask 构建异常时，维护者无法定位根因。

**修改方案**：在两处捕获时补充打印 `exc` 信息，例如 `print(f"[mask-aware] ... failed: {exc}")`。

---

## 二、设计偏差（相对 v4 规范）

### 2.1 `_validate_mask_aware_config` 强制要求 3D 过滤开启

**涉及文件**：[dataset/realworld.py:247-248](../dataset/realworld.py#L247)

```python
if not self.mask_enable_2d_reweight or not self.mask_enable_3d_filter:
    raise ValueError("mask_aware requires both enable_2d_reweight and enable_3d_filter")
```

v4 规范 §1.3.2 明确指出：

> "2D 降权路径 MUST；3D 过滤路径 SHOULD，并保持独立开关。"

当前实现将 3D 过滤也作为强制项，无法单独开启 2D 降权。这与规范所述的分阶段迁移路线（先 2D 后 3D）相矛盾，使阶段二的安全验证通道被阻断。

maintainer notes 将此项标为 P2，但此处影响的是规范声明的核心迁移路线，建议优先处理。

**修改方案**：将校验分拆，`enable_2d_reweight` 保持强制，`enable_3d_filter` 仅在为 false 时打印告警而非抛错。

---

### 2.2 `_resolve_mask_dir` 多候选路径搜索，但不记录选中路径

**涉及文件**：[dataset/realworld.py:172-195](../dataset/realworld.py#L172)

函数按顺序尝试 6 个候选目录，找到第一个非空的即返回，但未打印选中的路径。v4 规范 §3 要求"记录一次对齐摘要日志，包含 scene 名、数量、首尾映射"，当前 `_build_mask_lookup` 中确实打印了首尾映射，但选中的 mask 目录本身（candidate 中的哪一个）没有日志记录。

当目录结构不符合预期时，可能静默选中错误目录，而维护者无从察觉。

**修改方案**：在 `_resolve_mask_dir` 找到目录后，打印一行 `[mask-align] resolved mask_dir=...` 日志。

---

## 三、AGENTS.md 规范违规

### 3.1 行尾注释（严禁）

**涉及文件**：[dataset/realworld.py:457](../dataset/realworld.py#L457)

```python
depths_for_cloud[mask01 > 0.5] = 0.0  # 用>0.5判断，避免浮点等号比较
```

AGENTS.md 明确要求"严禁使用行尾注释，必须使用单行注释"。

**修改方案**：将注释移至该行上方作为独立注释行。

---

## 四、代码可读性问题

### 4.1 `mask_lookup` 双重查找冗余

**涉及文件**：[dataset/realworld.py:119](../dataset/realworld.py#L119)

```python
mask_path = mask_lookup.get(frame_id, mask_lookup.get(str(frame_id), None))
```

`_build_mask_lookup` 在构建时已同时以 int 和 str 为键写入字典（lines 214-219），因此此处双重查找是冗余的，直接 `mask_lookup.get(frame_id)` 即可命中。当前写法会让读者误以为两种 key 类型的覆盖逻辑仅在此处处理。

---

### 4.2 `image_mask_weight` 的 shape 契约缺乏显式说明

**涉及文件**：[policy/policy.py:74-75](../policy/policy.py#L74)

```python
if image_mask_weight is not None:
    image_mask_weight = image_mask_weight.flatten(2).squeeze(1)
```

此处 `flatten(2).squeeze(1)` 隐含 input shape 为 `(B, 1, H, W)`。当输入为其他形状时会静默产生错误结果而不报错（例如 `(B, H, W)` 时 `flatten(2)` 是 no-op，`squeeze(1)` 视 H 大小可能也是 no-op）。

训练路径和推理路径均已验证实际输出形状为 `(B, 1, coord_h, coord_w)`，逻辑正确，但缺少 `assert` 或注释说明 shape 前提，给后续维护留下隐患。

**修改方案**：在该操作前加一行 `assert image_mask_weight.dim() == 4` 作为防御性检查。

---

### 4.3 `eval.py` 中 3D 过滤深度赋值与 `realworld.py` 不一致

**涉及文件**：[eval.py:449](../eval.py#L449) vs [dataset/realworld.py:456](../dataset/realworld.py#L456)

```python
# eval.py
depths_for_cloud[mask01 > 0.5] = 0      # int 0

# realworld.py
depths_for_cloud[mask01 > 0.5] = 0.0    # float 0.0
```

语义相同（numpy 会做类型转换），但写法不一致，降低了代码的对称性和可核查性。建议统一为 `0.0`。

---

## 五、改动优先级汇总

| 优先级 | 问题 | 文件 | 行号 |
|---|---|---|---|
| P0 | 配置 `r_min`/`eps`/`tiny` 不驱动实际插值 | `realworld.py`, `sparse_modules.py` | 317-319, 72 |
| P1 | 异常被完全吞没，缺少 exc 信息输出 | `eval.py` | 171, 195 |
| P1 | `_validate_mask_aware_config` 强制 3D 过滤，违反 v4 规范 SHOULD 语义 | `realworld.py` | 247-248 |
| P1 | 规范违规：行尾注释 | `realworld.py` | 457 |
| P1 | `mask_stats` key 与日志 reason 不一致 | `eval.py` | 510-511 |
| P2 | `_resolve_mask_dir` 未记录选中路径 | `realworld.py` | 172-195 |
| P2 | `mask_lookup` 双重查找冗余 | `realworld.py` | 119 |
| P2 | `image_mask_weight` shape 契约无显式检查 | `policy/policy.py` | 74-75 |
| P2 | 3D 过滤赋值写法不一致（`0` vs `0.0`） | `eval.py`, `realworld.py` | 449, 456 |
