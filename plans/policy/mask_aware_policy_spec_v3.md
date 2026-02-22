# Mask-aware RISE-2：v3 可执行 Spec（2D mask→3D 过滤 + 2D patch 降权）

> 目标：在不改变 v1/v2 计划文件的前提下，固化一份更“工程可执行”的约束文档（spec）。该 spec 用于后续切换到 [`code`](#) 模式后实施。
>
> 本 spec 只描述 **必须实现的行为**、**数据格式**、**失败处理**、**对齐规则**、**公式** 与 **接口/配置**，避免开放式描述。

---

## 0. 术语与约定

- **scene**：一个演示片段目录，内部有 `color/`、`depth/`（可能还有 `action/` 等）。
- **color frame**：RGB 图像帧，文件名通常为 timestamp（如 `1700000000.123.png`）。
- **mask frame**：mask PNG，文件名为顺序号（如 `00001.png`）。
- **mask01**：float 张量，取值 {0,1}，其中 **1 表示不可信/需要 mask 掉**，0 表示可信。
- **reliability**：float 张量，取值 [0,1]，**越大越可信**。

约束级别关键词：
- MUST：必须
- SHOULD：建议（不影响主流程）
- MAY：可选

---

## 1. mask 语义（强约束）

依据你提供的 `generate_mask.ipynb` 的关键 cell（`save_masks_for_propainter()`）逻辑：

- 输出 mask PNG 为 uint8 灰度图（PIL mode `L`）。
- `binary_mask = (mask > 0) * 255`，多来源 mask 用 `np.maximum` 合并，并可选 `cv2.dilate`。

因此本工程 **mask 语义必须固定为**：

- 白色（255 或任意 `>0`）= **需要被 mask 掉/不可信区域**
- 黑色（0）= **可信区域**

实现侧 MUST 使用如下转换（阈值严格为 `>0`，不允许改成 `>127` 等）：

- `mask01 = (mask_png > 0).float()`
  - `mask01 == 1`：不可信
  - `mask01 == 0`：可信

---

## 2. mask 文件路径与帧对齐规则（方案B，强约束）

### 2.1 mask 根路径可手动指定

Dataset MUST 支持一个可配置的 mask 根目录（建议通过 yaml config/命令行参数注入），称为 `mask_root`。

- 若未提供 `mask_root`：mask-aware 功能 MUST 处于关闭状态（等价于不使用 mask）。
- 若提供 `mask_root`：对每个 scene 必须能定位其 mask 子目录。

建议约定（可选，但需在实现中二选一并写死）：
- 约定1：`mask_dir = mask_root/<scene_name>/`（scene 名与数据集 scene 目录同名）
- 约定2：`mask_dir = <scene_dir>/mask/`（mask 与原始数据同目录）

本 spec 不强制二选一，但实现时 MUST 固化一种，并在 README/CONFIG 中同步。

### 2.2 帧对齐采用方案B：排序后一一对应

由于 mask 是基于 color 生成，且你保证 1:1 对应，因此使用 **方案B**：

对同一 scene：

1. 取 color 帧列表 `C = sorted(color/*.png)`
2. 取 mask 帧列表 `M = sorted(mask_dir/*.png)`
3. MUST 满足 `len(C) == len(M)`
   - 若不满足：MUST 直接报错并拒绝训练/评估（防止 silent mismatch）。
4. 第 `i` 个 color 帧使用第 `i` 个 mask 帧：`mask_path = M[i]`

额外强约束：
- MUST 以“排序后的 index”作为唯一映射依据；不得尝试根据文件名解析 timestamp 与序号关系。
- MUST 在日志中至少打印一次：`scene_name, len(C), len(M), first_pair, last_pair` 以便快速定位对齐问题。

---

## 3. depth_scale 是否总是 1000：结论与约束

### 3.1 代码现状（事实）

当前仓库里存在 **不一致**：

- [`dataset/data_utils.py`](dataset/data_utils.py:1) 的 `ImageProcessor.get_image_coordinates()` 内部 **硬编码** `depth_scale = 1000`（即使外部传参也会被覆盖）。
- [`dataset/projector.py`](dataset/projector.py:1) 维护了不同相机序列号的 `depth_scales`：常见为 1000，L515 为 4000。

因此：
- “depth_scale 是否总是 1000” **在当前代码里对 `get_image_coordinates()` 来说是 true（被硬编码）**
- 但在系统层面与其它模块（如 projector）来说 **并不保证总是 1000**

### 3.2 v3 spec 的强约束（必须修正/统一）

为了让 mask-aware 的 2D→3D 过滤与 `image_coords` 的几何一致，depth_scale MUST 变为 **显式一致**。

MUST 满足：

- 对同一帧，以下三件事使用 **同一个** `depth_scale`：
  1) 从 depth 反投影生成点云（2D→3D）
  2) 计算 `image_coords`（patch reference 3D coords）
  3) 使用 2D mask + depth 做精确 3D 过滤（如果启用）

实现允许两种策略（二选一，后续实现时必须固定并写入 config 文档）：

- 策略S1（推荐）：以 [`dataset/projector.py`](dataset/projector.py:1) 的 `depth_scale` 为准（按相机 serial 决定），并传递到所有用 depth 的路径；同时移除/禁止 `get_image_coordinates()` 内部硬编码。
- 策略S2：统一强制 1000，但 MUST 在数据预处理阶段把所有 depth 都转换成与 1000 兼容的单位（否则 L515 数据会出错）。

本 spec 推荐 S1。

---

## 4. mask-aware 的两条生效路径（必须都支持）

mask-aware 功能分两部分：

- A) **2D mask → 精确 3D 过滤**：在生成点云时剔除不可信区域对应的 3D 点
- B) **2D patch 降权**：在 [`policy/sparse_modules.py`](policy/sparse_modules.py:1) 的 weighted interpolation 里，对来自不可信 patch 的语义特征降低权重

这两条路径 MUST 可独立开关（便于 ablation），但默认必须都开启。

---

## 5. A 路径：2D mask → 精确 3D 点过滤（强约束）

### 5.1 输入

对每帧输入：
- `color`：H×W×3
- `depth`：H×W（uint16 或 float）
- `mask_png`：H×W（uint8，灰度）
- `intrinsics K`：fx, fy, cx, cy（与 depth 对齐后的分辨率一致）
- `depth_scale`：见 §3

### 5.2 精确过滤规则

MUST 在 **反投影生成点云之前** 使用 mask 执行过滤：

- 计算 `mask01 = (mask_png > 0)`（bool）
- 对每个像素 (u,v)：
  - 若 `mask01[v,u] == True`：该像素对应的深度点 MUST 被丢弃（不产生 3D 点）
  - 若深度无效（0/NaN）：也 MUST 丢弃
  - 仅当 `mask01[v,u] == False` 且深度有效：才反投影生成点

反投影公式（相机坐标系，单位与 `depth_scale` 一致）：

- `z = depth[v,u] / depth_scale`
- `x = (u - cx) / fx * z`
- `y = (v - cy) / fy * z`

### 5.3 与 voxel/crop 的顺序

MUST 顺序如下：

1) depth validity 过滤
2) mask 过滤
3) 反投影生成 dense 点集（可带颜色，但 RISE-2 sparse encoder 不用颜色也可）
4) workspace crop（若现有 pipeline 有 crop）
5) voxel downsample

理由：mask 过滤若放在 voxel/crop 后会变成近似过滤（不再精确）。

### 5.4 边界条件

- 若过滤后点数为 0：
  - MUST 有确定性处理：要么直接抛异常（推荐，避免训练 silently 崩），要么回退到“不过滤”模式（不推荐）。实现时二选一并固定。

---

## 6. B 路径：2D patch 降权（WeightedSpatialInterpolation 的公式约束）

### 6.1 patch 级 reliability 的计算

给定 `mask01`（1=不可信），对其进行与 dense encoder feature map 分辨率一致的池化，得到 `mask_ratio`：

- `mask_ratio = avg_pool(mask01)`
  - 输出 shape MUST 与 `image_feat` 的空间分辨率一致（即 `img_coord_size` 对齐的那个分辨率）

再得到

- `reliability = 1 - mask_ratio`

强约束：
- reliability 范围 MUST 在 [0,1]
- 对输入全黑 mask：reliability 全 1
- 对输入全白 mask：reliability 全 0

### 6.2 将 reliability 注入插值权重

现有插值（参照论文 Eq.(1) 与实现）为：

- 对每个 query seed 点 `c_g`，找 M 个邻居 `n_j`（来自 `C_s`）
- 距离 `d_j = dist(c_g, n_j)`
- base 权重 `w_j = 1 / (d_j + eps)`
- 归一化 `w_j_norm = w_j / sum(w_j)`
- 输出 `f = sum(w_j_norm * f_j)`

mask-aware MUST 改为：

- `w_j = reliability_j / (d_j + eps)`
- `w_j_norm = w_j / (sum(w_j) + tiny)`

其中：
- `reliability_j` 是邻居点 `n_j` 对应的 patch reliability（来自 §6.1）
- `eps`、`tiny` MUST 为常数（如 1e-6），避免 NaN

### 6.3 全零 reliability 的处理（强约束：r_min 下限钳制）

当某个 query 的所有 M 个邻居 `reliability_j == 0`，按 `w_j = reliability_j / (d_j + eps)` 会导致所有 `w_j=0`，归一化时出现 `0/0` → NaN。

本 v3 采用工程化的 **下限钳制** 来从根源避免全零权重：

- 设定常数 `r_min`（默认建议 `1e-3`）
- 将权重定义为：
  - `w_j = max(reliability_j, r_min) / (d_j + eps)`
  - `w_j_norm = w_j / (sum(w_j) + tiny)`

强约束：
- `r_min` MUST 为常数并可配置（建议放入 config），但必须有默认值
- 当 `reliability_j = 0` 时，仍会被强烈抑制（只保留极小权重），但不会再出现 `sum(w)=0`
- 不允许出现 NaN/Inf

备注（非强约束）：
- 若你希望更严格的屏蔽语义，可选替代策略：
  - R2：检测 `sum(reliability)==0` 时直接输出全零语义特征
  - R1：检测 `sum(reliability)==0` 时回退到原始距离权重（忽略 mask）
  这些策略更“语义纯粹”，但可能带来分布突变或更强退化；默认不采用。

---

## 7. 需要新增/传递的数据字段（接口 spec）

Dataset 输出 dict（单帧或序列）在启用 mask_root 时 MUST 额外包含：

- `mask_path`（str，可选用于 debug）
- `mask01`（torch.float32 或 torch.bool；若存 float 则 MUST 为 {0,1}）

此外，为支持 patch 降权，还需要能得到 patch 级 reliability：

- MAY 在 dataset 侧直接输出 `patch_reliability`（shape 与 `image_feat` 空间一致）
- 或者在 model 前处理阶段根据 `mask01` 计算（但 MUST 确保与 dense encoder 的 resizing/pooling 完全一致）

为减少对齐错误，本 spec 推荐 dataset 直接输出 `patch_reliability`。

---

## 8. 最小验证清单（必须能验证）

实现完成后 MUST 能通过以下检查（无需训练收敛）：

1. **对齐检查**：随机抽一个 scene，打印首尾帧映射对：`color[i] ↔ mask[i]`
2. **语义检查**：取一张 mask，统计 `mask01.mean()` 与目视一致（白多则 mean 大）
3. **3D 过滤检查**：同一帧，启用/关闭 3D 过滤，点数 MUST 显著减少（如果 mask 有白区）
4. **插值数值稳定**：在全白 mask 下不允许出现 NaN/Inf（R1/R2 任一策略都必须稳定）

---

## 9. 与现有代码的挂接点（定位用，不是实现）

后续代码落点大概率涉及：

- 点云生成：[`dataset/realworld.py`](dataset/realworld.py:1)
- `image_coords` 生成与 pooling：[`dataset/data_utils.py`](dataset/data_utils.py:1)
- 插值权重：[`policy/sparse_modules.py`](policy/sparse_modules.py:1)
- config 注入：[`configs/*.yaml`](configs/test.yaml:1)

---

## 10. 本 spec 的“完成定义”

当满足以下条件，v3/spec 实施视为完成：

- mask_root 可配置；方案B 排序对齐实现并带断言
- mask01 语义与阈值 `>0` 固化
- depth_scale 不再隐式硬编码导致不一致（至少在用到 mask-aware 的路径上保持一致）
- 3D 精确过滤 + 2D patch 降权两条路径都可开关且默认可用
- 最小验证清单全部可执行且通过
