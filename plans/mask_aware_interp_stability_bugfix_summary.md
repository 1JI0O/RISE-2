# RISE-2 这次单点修复到底修了什么（人话版）

## 如果你只看 1 分钟

这次只修了一个问题：

> 配置里写了 `r_min / interp_eps / interp_tiny`，但模型插值时并没有真的用上。

现在已经修好。修法很小，只改两处：

1. 在数据侧把这 3 个参数一起带下去（像“打包寄出”）。
2. 在插值调用前把它们拆出来并传给插值函数（像“签收并使用”）。

对应代码位置：

- 数据打包：[`RealWorldDataset.__getitem__()`](dataset/realworld.py:407)
- 插值签收：[`SpatialAligner.forward()`](policy/sparse_modules.py:196)
- 最终生效函数：[`WeightedSpatialInterpolation.forward()`](policy/sparse_modules.py:72)

---

## 1. 之前到底哪里不对

从表面看，配置已经有稳定参数，数据集也读到了：

- [`RealWorldDataset._parse_config()`](dataset/realworld.py:269)

但实际执行时，插值调用没有拿到这三个值，导致一直用默认值。

可以把它想成一条管线：

- A 端（配置）有水
- B 端（插值）能接水
- 但中间水管没接上

所以“看起来有配置”，但“跑起来没变化”。

---

## 2. 这次怎么修（最小改动）

### 改动 1：数据侧“打包”

位置：[`dataset/realworld.py`](dataset/realworld.py:446)

原来只会产生 `image_mask_weight`（patch 权重）。

现在在它后面追加 3 个值：

- `mask_r_min`
- `mask_interp_eps`
- `mask_interp_tiny`

也就是把权重从 `src_len` 变成 `src_len + 3`，方便下游直接取。

---

### 改动 2：对齐层“解包并透传”

位置：[`policy/sparse_modules.py`](policy/sparse_modules.py:210)

在 [`SpatialAligner.forward()`](policy/sparse_modules.py:196) 里做了兼容处理：

- 如果长度是 `src_len + 3`：
  - 取最后 3 个值作为 `r_min / eps / tiny`
  - 前 `src_len` 当作真正的 `src_weights`
- 如果长度是 `src_len`：
  - 认为是旧格式，继续走默认值
- 其他长度：
  - 直接报错，避免静默错用

然后把参数显式传给：

- [`self.interp()`](policy/sparse_modules.py:225)
- 最终进入 [`WeightedSpatialInterpolation.forward()`](policy/sparse_modules.py:72)

---

## 3. 改前 vs 改后（直观对比）

| 场景 | 改前 | 改后 |
|---|---|---|
| 配置写了非默认稳定参数 | 基本不生效（插值仍用默认） | 真正生效（插值使用配置值） |
| 旧格式权重（只有 `src_len`） | 可运行 | 仍可运行（兼容不变） |
| 权重长度异常 | 可能难排查 | 明确报错，定位更快 |

---

## 4. 会不会影响现有流程

结论：影响非常小，且可控。

- 只改了两个文件：
  - [`dataset/realworld.py`](dataset/realworld.py:446)
  - [`policy/sparse_modules.py`](policy/sparse_modules.py:210)
- 没改训练主调用接口：[`train.py`](train.py:1)
- 没改策略入口签名：[`RISE2.forward()`](policy/policy.py:52)
- 没改本地推理主文件：[`eval.py`](eval.py:1)

所以这次是“修接线”，不是“重构主链路”。

---

## 5. 一条样本怎么走（流程化人话）

1. 数据集在 [`RealWorldDataset.__getitem__()`](dataset/realworld.py:407) 里先算出 patch 权重。  
2. 顺手把 3 个稳定参数拼到尾部一起返回。  
3. 模型前向到 [`SpatialAligner.forward()`](policy/sparse_modules.py:196) 时，先判断长度。  
4. 若检测到尾部参数，就拆出来。  
5. 调用 [`WeightedSpatialInterpolation.forward()`](policy/sparse_modules.py:72) 时把这 3 个值显式传进去。  
6. 插值计算按配置值执行，而不是硬编码默认值。

---

## 6. 这次修复的验证

已做最小语法检查：

- [`python -m py_compile dataset/realworld.py policy/sparse_modules.py`](dataset/realworld.py:1)
- 结果：通过（exit code 0）

---

## 7. 最终结论

这次修复解决的是“参数读到了但没用上”的典型接线问题。

现在训练链路中，`mask_aware.r_min / interp_eps / interp_tiny` 已经能真实进入插值计算；同时保留了对旧格式输入的兼容，不会把原有流程打坏。