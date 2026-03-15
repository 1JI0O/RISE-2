# Dataset Eval Guide — eval_sam2_mask_dataset.py

使用已有训练数据离线测试 SAM2 mask 追踪流程，无需连接真实机器人。

---

## 用途与适用场景

| 场景 | 使用文件 |
| --- | --- |
| **真实机器人 eval** | `eval_sam2_mask_final.py` |
| **用训练数据离线测试 SAM2 / pipeline** | `eval_sam2_mask_dataset.py`（本文件） |

`eval_sam2_mask_dataset.py` 从磁盘逐帧读取 color + depth PNG，以与真实 eval 完全相同的方式送入 SAM2，输出 mask 可视化图片并打印统计。Policy 推理为**可选**——不提供 `--ckpt` 时仅测试 SAM2 追踪，提供后则跑完整推理链路（除 robot action 外均实际执行）。

---

## 数据集目录结构要求

```text
<scene_dir>/
    <camera_id>/
        color/
            <timestamp>.png    # uint8 RGB，任意分辨率
            ...
        depth/
            <timestamp>.png    # uint16 depth in mm
            ...
    lowdim/                    # 可选，本脚本不读取
    meta.json                  # 可选
```

示例：

```text
/data/haoxiang/data/airexo2/task_0012/train/scene_0001/
    cam_105422061350/
        color/1737546126606.png  1737546159940.png ...
        depth/1737546126606.png  1737546159940.png ...
```

- `color/` 和 `depth/` 下的文件名（去掉 `.png`）作为时间戳，二者必须严格一一对应。
- 脚本按时间戳升序扫描 `color/`；若任意一帧在 `depth/` 中找不到同名文件，立即报错退出。

---

## CLI 参数

### Dataset 模式专用参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `--dataset PATH` | str, **required** | 场景目录，见上文结构 |
| `--camera_id NAME` | str, **required** | 相机子目录名，如 `cam_105422061350` |
| `--max_frames N` | int, optional | 最多处理 N 帧（默认=数据集全部帧数）；帧数用尽即退出 |
| `--save_vis DIR` | str, optional | 将 mask overlay 图片保存到此目录 |

### 通用参数（dataset 模式下均为 optional）

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `--config PATH` | str, **required** | YAML 配置文件（同训练时） |
| `--ckpt PATH` | str, optional | Policy checkpoint；不提供则跳过 policy 推理 |
| `--type local\|remote` | str, default=`local` | dataset 模式下固定为 local，可忽略 |
| `--calib_airexo PATH` | str, optional | dataset 模式下不需要 |
| `--calib_rise2 PATH` | str, optional | dataset 模式下不需要 |
| `--host / --port` | str/int | 仅 remote policy 模式使用 |

---

## 典型启动命令

> `rise2` 和 `sam2` conda 环境**不兼容**，SAM2 必须以独立服务运行。
> 所有场景均需先在终端 1 启动 SAM2 服务端，再在终端 2 运行 dataset eval。

### 场景 A：纯 SAM2 追踪测试（不跑 policy）

最常用，验证 SAM2 冷启动标注与逐帧追踪。

```bash
# 终端 1（sam2 环境）
conda run -n sam2 python sam2_mask_server.py \
    --config configs/dual_teleop_dino.yaml --port 8765
# 等待打印：[sam2-server] listening on ws://0.0.0.0:8765

# 终端 2（rise2 环境）
conda activate rise2
python eval_sam2_mask_dataset.py \
    --config configs/dual_teleop_dino.yaml \
    --dataset /data/haoxiang/data/airexo2/task_0012/train/scene_0001 \
    --camera_id cam_105422061350 \
    --max_frames 100 \
    --save_vis /tmp/sam2_vis_test
```

YAML 中必须设置 `remote_port: 8765`：

```yaml
mask_aware:
  enabled: true
  sam2:
    enabled: true
    remote_port: 8765
```

启动后流程：

1. 终端 2 连接 SAM2 服务，`t=0`：标注窗口在**终端 1** 弹出，标注机械臂（`[a]`）和 gripper（`[g]`），按 `Enter` 确认
2. `t=1…N`：SAM2 自动逐帧追踪，mask overlay 图片保存到 `--save_vis` 目录
3. 帧数用尽即退出，打印 `mask_stats` 汇总

### 场景 B：完整 pipeline 测试（含 policy 推理，不执行 robot action）

在场景 A 基础上加 `--ckpt`，policy 推理、点云构建、2D mask reweighting 均会实际执行，`agent.action()` 被静默忽略。

```bash
# 终端 1（sam2 环境）—— 同场景 A
conda run -n sam2 python sam2_mask_server.py \
    --config configs/dual_teleop_dino.yaml --port 8765

# 终端 2（rise2 环境）
conda activate rise2
python eval_sam2_mask_dataset.py \
    --config configs/dual_teleop_dino.yaml \
    --dataset /data/haoxiang/data/airexo2/task_0012/train/scene_0001 \
    --camera_id cam_105422061350 \
    --ckpt logs/your_checkpoint/checkpoint.pth \
    --max_frames 60 \
    --save_vis /tmp/sam2_vis_test
```

---

## 与 eval_sam2_mask_final.py 的区别

| 功能 | eval_sam2_mask_final.py | eval_sam2_mask_dataset.py |
| --- | --- | --- |
| 数据来源 | 真实机器人相机 | 磁盘 PNG（`DatasetAgent`） |
| 机器人连接 | 必须 | 无（no-op） |
| 坐标投影 | `Projector.project_tcp_to_base_coord()` | 跳过 |
| Policy | 必须提供 ckpt | 可选，不提供时只跑 SAM2 |
| `input()` 等待 | rollout 前等待 Enter | 无（立即开始） |
| 帧数控制 | `config.deploy.max_steps` | `--max_frames`（不足时停止，默认=数据集帧数） |
| mask 可视化 | `config.deploy.vis=True` 时 | `--save_vis DIR` 时 |
| Calib 文件 | 必须提供 | 不需要 |
| 相机内参 | 从 agent 读取 | 使用 `fake_intrinsics`（硬编码） |

> **注意**：相机内参使用硬编码的 `fake_intrinsics`，与真实相机略有差异。
> 如果需要精确的点云几何（场景 C），可以在脚本顶部修改 `fake_intrinsics` 为实际相机参数。

---

## 输出说明

- **终端输出**：每步 SAM2 状态、mask 失败原因、末尾 `mask_stats` 汇总
- **`--save_vis DIR/`**：`step_000000_overlay.png`、`step_000001_overlay.png`… 红色半透明覆盖表示 mask 区域
- **`mask_stats` 含义**：

| 计数器 | 含义 |
| --- | --- |
| `sam2_propagate_fail` | SAM2 propagate 返回空 mask |
| `sam2_reset_exception` | 周期重置时发生异常 |
| `sam2_cold_start_aborted` | 用户按 ESC 放弃标注 |
| `sam2_remote_exception` | 远程服务调用失败 |
| `infer_none` | SAM2 返回 None（含以上所有原因） |
