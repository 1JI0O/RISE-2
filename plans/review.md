# Mask-Aware RISE-2 代码审查报告

> 审查范围：上一轮 Agent 生成的 mask-aware 改造代码，涉及文件：
> `eval.py`、`dataset/realworld.py`、`policy/policy.py`、`policy/sparse_modules.py`、`train.py`

---

## 一、潜在 Bug 与正确性问题

### 1.1 严重：`agent` 对象未定义即被调用（必修）

**文件**：[eval.py:420](../eval.py#L420)、[eval.py:596](../eval.py#L596)

Agent 的初始化代码被注释掉：

```python
# Agent = SingleArmAgent if config.robot_type == "single" else DualArmAgent
# agent = Agent(**config.deploy.agent)
```

但随后仍然调用了 `agent.get_global_observation()`（line 420）和 `agent.stop()`（line 596）。运行时会立即抛出 `NameError: name 'agent' is not defined`，整个推理流程无法启动。

**修改方案**：恢复 agent 初始化，或将对 agent 的调用替换为开发阶段的 mock。

---

### 1.2 严重：`torch.load` 缺少 `weights_only=True`（安全问题）

**文件**：[eval.py:355](../eval.py#L355)

```python
policy.load_state_dict(torch.load(args.ckpt, map_location = device), strict = False)
```

未传入 `weights_only=True`，允许 checkpoint 文件通过 pickle 执行任意代码，存在安全风险。对比 `train.py:109` 已正确使用该参数。

**修改方案**：

```python
policy.load_state_dict(
    torch.load(args.ckpt, map_location=device, weights_only=True), strict=False
)
```

---

### 1.3 重要：配置中的插值参数不驱动实际插值行为

**文件**：[dataset/realworld.py:317-319](../dataset/realworld.py#L317)、[policy/sparse_modules.py:72](../policy/sparse_modules.py#L72)

`_parse_config` 读取并校验了 `mask_aware.r_min`、`interp_eps`、`interp_tiny`，但 `WeightedSpatialInterpolation.forward()` 使用的是函数签名里的硬编码默认值：

```python
def forward(self, tgt, src, tgt_feats, src_feats, src_weights=None, k=3, r_min=1e-3, eps=1e-6, tiny=1e-6):
```

调用链 `SpatialAligner.forward()` → `self.interp(...)` 也未将配置值传入。修改 YAML 中的 `r_min` 等参数不会产生任何实际效果，与配置文件的语义声明相矛盾。

维护手册已在 §9 P0 中标记此项为优先修复项。

**修改方案**：在 `SpatialAligner` 初始化时存储这些参数，并在调用 `self.interp(...)` 时显式传入。

---

### 1.4 重要：异常信息被完全吞没

**文件**：[eval.py:195](../eval.py#L195)

```python
except Exception:
    return None
```

`_build_image_mask_weight` 内部发生异常时，没有任何日志输出，开发者无法得知失败原因。同样，[eval.py:171](../eval.py#L171) 的 `_safe_infer_mask` 仅记录了原因码，未打印原始异常堆栈。

**修改方案**：在捕获异常时加入日志：

```python
except Exception as exc:
    print(f"[mask-aware] _build_image_mask_weight failed: {exc}")
    return None
```

---

### 1.5 轻微：空点云的 `fail_fast` 策略在 dataset 侧不受 `empty_cloud_policy` 完整控制

**文件**：[dataset/realworld.py:478-483](../dataset/realworld.py#L478)

回退后重建的点云若仍为空，会无条件抛出 `RuntimeError`，不区分 `warn_and_skip` 与 `fail_fast`。在 `warn_and_skip` 策略下，这里实际执行的是 fail 行为，与异常矩阵声明不一致。但这属于极端边界情况，影响范围有限。

---

### 1.6 轻微：`mask_lookup` 的双重查找冗余

**文件**：[dataset/realworld.py:119](../dataset/realworld.py#L119)

```python
mask_path = mask_lookup.get(frame_id, mask_lookup.get(str(frame_id), None))
```

`_build_mask_lookup` 在构建时已同时以 int 和 str 作为键存入字典（lines 214-219），因此这里的双重查找是多余的。虽然不是 bug，但增加了维护负担。

---

## 二、AGENTS.md 规范违规

### 2.1 行尾注释（严禁）

**文件**：[dataset/realworld.py:457](../dataset/realworld.py#L457)

```python
depths_for_cloud[mask01 > 0.5] = 0.0  # 用>0.5判断，避免浮点等号比较
```

AGENTS.md 明确规定"严禁使用行尾注释，必须使用单行注释"。

**修改方案**：将注释移到该行前面作为独立行。

---

### 2.2 数字序号注释（禁止）

**文件**：[eval.py:202](../eval.py#L202)、[eval.py:209](../eval.py#L209)

```python
# 1. 加载彩色图并转为 RGB (OpenCV 默认读入是 BGR)
...
# 2. 加载深度图
```

AGENTS.md 规定"注释中绝对不允许出现 Step xx、步骤 xx、Phase xx 或者序号 x"。

**修改方案**：将数字序号去掉，改为直接描述性注释。

---

### 2.3 异常消息使用中文（应用英文）

**文件**：[eval.py:204](../eval.py#L204)、[eval.py:211](../eval.py#L211)

```python
raise ValueError(f"无法加载图片: {color_path}")
raise ValueError(f"无法加载深度图: {depth_path}")
```

AGENTS.md 规定"print 等日志输出使用简明的英文"。异常消息属于对外可见输出，应当使用英文。

**修改方案**：

```python
raise ValueError(f"Failed to load color image: {color_path}")
raise ValueError(f"Failed to load depth image: {depth_path}")
```

---

### 2.4 遗留无意义注释

**文件**：[dataset/realworld.py:20](../dataset/realworld.py#L20)

```python
# need to be modified
```

这是遗留的占位 TODO，对维护者无实际信息价值，应当删除。

---

## 三、代码可读性问题

### 3.1 `eval.py` 整体处于半完成的开发状态

以下几处残留的开发桩使生产代码的可读性和可维护性下降：

- **硬编码本地路径**（[eval.py:24-25](../eval.py#L24)）：

  ```python
  test_color = "/home/haoxiang/RISE-2/saved_test_data/color_0000.png"
  test_depth = "/home/haoxiang/RISE-2/saved_test_data/depth_0000.png"
  ```

- **`fake_intrinsics` 和 `fake_depth_scale`**（[eval.py:27-33](../eval.py#L27)）：模块级常量，并在 `evaluate()` 中被直接使用（lines 454-458, 499），混淆了真实推理流程。

- **`load_test_obs` 函数**（[eval.py:201](../eval.py#L201)）：定义了但从未调用（对应调用点被注释掉在 line 422），是死代码。

这些内容应当在代码完成后清理，或用明显的 `DEV_MODE` 标志隔离。

---

### 3.2 `realworld.py` 中引用赋值模式可读性差

**文件**：[dataset/realworld.py:453](../dataset/realworld.py#L453)

```python
depths_for_cloud = depths
if self.mask_aware_enabled and self.mask_enable_3d_filter and mask01 is not None:
    depths_for_cloud = depths.copy()
    depths_for_cloud[mask01 > 0.5] = 0.0
```

初始赋值 `depths_for_cloud = depths` 是一个共享引用，仅在满足条件时才会执行 `.copy()`。这个模式需要读者仔细追踪才能确认安全性。建议改为先判断条件，再决定是否复制，可以消除认知负担。

---

### 3.3 预先存在的 `depth_scale` 硬编码问题（非 Agent 引入，但值得记录）

**文件**：[dataset/data_utils.py:253](../dataset/data_utils.py#L253)

```python
def get_image_coordinates(self, depths, intrinsics, depth_scale, fill_hole=True):
    ...
    depth_scale = 1000  # 函数参数被局部变量覆盖
```

`depth_scale` 参数被函数内部硬编码覆盖，传入任何值均无效。这个 bug 预先存在，但 mask-aware 改造中在两条路径下都调用了该函数，如果未来需要使用非 1000 的 depth scale，必须先修复这里。

---

## 四、正确性验证（形状分析）

以下关键数据流的 shape 分析验证均正确，无问题：

| 位置 | 张量 | 预期形状 |
|---|---|---|
| `realworld.py` mask_ratio 输出 | `(1, coord_h, coord_w)` | 正确 |
| collate 后 `image_mask_weight` | `(B, 1, coord_h, coord_w)` | 正确 |
| `policy.py` flatten/squeeze 后 | `(B, coord_h * coord_w)` | 正确 |
| `SpatialAligner` 按样本切片 | `(1, coord_h * coord_w)` | 正确 |
| `WeightedSpatialInterpolation` gather | `(n, k)` | 正确 |
| eval.py unsqueeze(0) 后传入 policy | `(1, 1, coord_h, coord_w)` | 正确 |

---

## 五、改动优先级汇总

| 优先级 | 问题 | 文件 | 行号 |
|---|---|---|---|
| P0（必修） | `agent` 未定义即调用导致运行崩溃 | `eval.py` | 420, 596 |
| P0（安全） | `torch.load` 缺少 `weights_only=True` | `eval.py` | 355 |
| P0（正确性） | 配置中 `r_min/eps/tiny` 不驱动实际插值 | `realworld.py`, `sparse_modules.py` | 317-319, 72 |
| P1 | 异常信息被完全吞没，无日志可查 | `eval.py` | 171, 195 |
| P1 | 规范违规：行尾注释 | `realworld.py` | 457 |
| P1 | 规范违规：数字序号注释 | `eval.py` | 202, 209 |
| P1 | 规范违规：中文异常消息 | `eval.py` | 204, 211 |
| P2 | 开发桩残留（硬编码路径、fake 参数） | `eval.py` | 24-33, 452-458 |
| P2 | 遗留无意义注释 `# need to be modified` | `realworld.py` | 20 |
| P2 | `mask_lookup` 双重查找冗余 | `realworld.py` | 119 |
| P2 | `load_test_obs` 死代码 | `eval.py` | 201-217 |
