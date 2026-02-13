# 推理阶段接入 Mask-Aware 的详细计划（仅本地 [`evaluate()`](eval.py:156)）

## 文档目标

本计划只定义**本地推理**改造方案，不涉及代码实现。目标是把训练端已经落地的 mask-aware 逻辑迁移到本地推理链路 [`evaluate()`](eval.py:156)，并保持兼容与可回退。

核心原则：

- 推理复用训练逻辑语义，不引入新语义分歧
- mask 来源由离线文件改为运行时函数接口
- 默认关闭时保持现有本地推理行为不变
- 先保证可运行与可回退，再逐步优化精度与稳定性

## 本阶段范围

- 包含：本地推理脚本 [`eval.py`](eval.py:1)
- 不包含：远程推理链路 [`eval_server.py`](eval_server.py:1)、[`WebsocketClientPolicy.infer()`](remote_eval/websocket_client_policy.py:33)

## 现状梳理（本地链路）

当前本地推理流程在 [`evaluate()`](eval.py:156)。

已具备的输入链路：

- RGB 与 Depth 获取
  - 本地测试路径：[`load_test_obs()`](eval.py:45)
  - 真实机器人路径预留在 [`SingleArmAgent.get_global_observation()`](eval_agent.py:64) 与 [`DualArmAgent.get_global_observation()`](eval_agent.py:218)
- 点云构建
  - [`create_input()`](eval.py:100) -> [`create_point_cloud()`](eval.py:64)
- 图像几何坐标与图像特征预处理
  - [`ImageProcessor.get_image_coordinates()`](dataset/data_utils.py:243)
  - [`ImageProcessor.preprocess_images()`](dataset/data_utils.py:271)
- 模型前向
  - [`policy(cloud_data, colors, image_coords, actions=None)`](eval.py:281)

与训练端差异：

- 训练端 mask 来自数据集离线对齐文件 [`RealWorldDataset.__getitem__()`](dataset/realworld.py:407)
- 本地推理端当前无 mask 获取、无 3D 过滤、无 2D reliability 传递

## 改造目标（本地）

把训练端两条机制迁移到本地推理：

- 方案A 3D 硬过滤
  - 在点云构建前，对深度图中不可信像素清零，再建点云
- 方案B 2D 软降权
  - 从 mask 计算 patch 级可靠度 `image_mask_weight`，传给 [`RISE2.forward()`](policy/policy.py:52)

并新增运行时 mask 接口：

- 输入：当前观测（至少 RGB、Depth、Proprio、可选元信息）
- 输出：`mask01` 或 `None`

## 接口定义

### 推理期 mask 提供器接口

建议定义统一接口（先空实现）：

```python
infer_mask(color, depth, proprio, meta) -> mask01 | None
```

参数语义：

- `color`：当前帧 RGB
- `depth`：当前帧深度
- `proprio`：当前本体状态（关节或 TCP + gripper）
- `meta`：可选上下文（camera intrinsics、timestamp、scene id 等）

返回语义：

- `mask01`：二维数组，`1` 表示不可信，`0` 表示可信
- `None`：表示本帧无 mask，可走回退路径

接口约束：

- mask 分辨率允许与 depth 不一致，后续统一最近邻对齐
- 输出可为 `bool`/`uint8`/`float`，推理管线统一转为 `float32` 的 0/1

### 配置扩展（本地）

沿用现有 `mask_aware` 语义并新增推理期最小配置：

- `mask_aware.enabled`
- `mask_aware.enable_3d_filter`
- `mask_aware.enable_2d_reweight`
- `mask_aware.mask_threshold`
- `mask_aware.mask_white_is_untrusted`
- `mask_aware.r_min`
- `mask_aware.interp_eps`
- `mask_aware.interp_tiny`
- `mask_aware.infer_allow_none`
- `mask_aware.infer_none_policy`

建议默认：

- `infer_none_policy: no_mask_fallback`
- `infer_mask_resize_mode: nearest`
- `infer_debug_dump: false`

## 数据流设计（本地 [`evaluate()`](eval.py:156)）

在每次观测周期内：

1. 采集 `colors`、`depths`、`proprio`
2. 调用 `infer_mask(...)` 得到 `mask01 | None`
3. 若启用 3D 过滤且 `mask01` 可用
   - 生成 `depths_for_cloud`
   - 执行 `depths_for_cloud[mask01 > 0.5] = 0`
   - 点云构建使用 `depths_for_cloud`
4. 图像坐标分支
   - 继续用原始 `depths` 生成 `image_coords`
5. 若启用 2D 重权且 `mask01` 可用
   - resize 到图像分支尺寸
   - pooling 得到 `mask_ratio`
   - `image_mask_weight = 1 - mask_ratio`
6. 调用 policy
   - 传 `image_mask_weight` 或 `None`

## 回退策略

### mask 不可用回退

触发条件：

- `infer_mask` 返回 `None`
- 接口异常
- 输出 shape 不合法且不可修复

策略：

- `no_mask_fallback`
  - 不做 3D 过滤
  - 不做 2D 降权
  - 走原始本地推理链路
- `fail_fast`
  - 立即抛错并终止当前 rollout

### 空点云回退

触发条件：

- 3D 过滤后点云为空

策略对齐训练语义：

- `warn_and_skip_filter`
  - 当前帧回退到未过滤深度重建点云
- `fail_fast`
  - 终止推理

### mask 质量保护

- 允许设置 `max_mask_ratio`
- 当全图几乎全 mask 时记录告警并回退

## 与训练语义一致性要求

必须保持以下一致：

- `mask01` 语义：1 不可信，0 可信
- 3D 过滤位置：深度图阶段
- 2D 权重定义：`image_mask_weight = 1 - mask_ratio`
- 插值权重机制：与 [`WeightedSpatialInterpolation.forward()`](policy/sparse_modules.py:72) 一致

允许差异：

- mask 来源是在线函数，不是离线文件
- 推理端可按实时性采用更保守回退策略

## 验证项（本地）

### 接口级验证

- `infer_mask` 返回 `None` 时推理可继续
- 返回合法 mask 时 shape 与 dtype 自动规范化
- 异常抛出时按配置执行回退或终止

### 功能级验证

- 开启 3D 过滤后，点云点数下降且非零
- 开启 2D 重权后，`image_mask_weight` 在 [0,1]
- policy 前向成功接收 `image_mask_weight`

### 一致性验证

- 推理配置关闭 mask-aware 时，输出与当前旧行为一致
- 本地同输入下，开启与关闭 mask-aware 的差异可解释

### 稳定性验证

- 连续多帧推理无内存泄漏
- mask 间歇缺失不会导致进程崩溃

## 分阶段执行建议（仅本地）

阶段一：最小闭环

- 仅改 [`eval.py`](eval.py:1)
- 接入空 `infer_mask` 接口与回退
- 跑通 3D 过滤与 2D 降权参数传递

阶段二：真实部署联调

- 接入真实 mask 生成函数
- 观察实时性与稳定性
- 调整回退策略

## 风险清单

- mask 与 depth 尺寸不一致导致错位
- 在线 mask 抖动导致动作不稳定
- 3D 过滤过强导致空点云频发

对应缓解：

- 强制最近邻 resize + shape 校验
- 增加时序平滑或阈值保护
- 配置化空点云回退

## 非本阶段事项

以下内容明确不在本次实现范围：

- 远程服务端改造 [`eval_server.py`](eval_server.py:1)
- websocket 协议扩展 [`remote_eval/websocket_policy_server.py`](remote_eval/websocket_policy_server.py:10)
- 远程客户端字段兼容 [`WebsocketClientPolicy.infer()`](remote_eval/websocket_client_policy.py:33)

以上内容留到后续独立阶段处理。

## 计划交付物

- 本地推理 mask 接口说明
- 本地数据流说明
- 本地回退策略说明
- 本地验证清单

本文件为“仅本地推理”版本，后续实现严格以此范围执行。