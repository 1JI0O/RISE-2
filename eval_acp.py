"""
Evaluation.
"""
import sys
import pathlib
import os
import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import copy
import random
import wandb
import pickle
import tqdm
import numpy as np
import shutil
from collections import deque
from datetime import datetime
import cv2

import yaml
from easydict import EasyDict as edict

# 1. 算出根目录
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.absolute())
# 2. 算出 PyriteML 所在的目录
PYRITE_ML_DIR = os.path.join(ROOT_DIR, 'PyriteML')

# 将这两个都加入环境变量
sys.path.append(ROOT_DIR)
sys.path.append(PYRITE_ML_DIR)

os.chdir(ROOT_DIR)

from PyriteML.diffusion_policy.workspace.base_workspace import BaseWorkspace
from PyriteML.diffusion_policy.policy.diffusion_unet_timm_mod1_policy import (
    DiffusionUnetTimmMod1Policy,
)

import PyriteUtility.spatial_math.spatial_utilities as su

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from PyriteML.diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset

# from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator

from scipy.spatial.transform import Rotation as R

from eval_agent import SingleArmAgent

# 图像观测：看最近 2 帧，步长为 1 (间隔约 50ms)
sparse_obs_rgb_down_sample_steps = 1
sparse_obs_rgb_horizon = 2

# 低维状态（Pose）：看最近 3 帧
sparse_obs_low_dim_down_sample_steps = 1
sparse_obs_low_dim_horizon = 3

# 力矩（Wrench）：力矩通常需要更长的历史信息。
sparse_obs_wrench_down_sample_steps = 1
sparse_obs_wrench_horizon = 32
# 动作预测（Action）：预测未来 16 帧（约 0.8s 的动作轨迹）
sparse_action_down_sample_steps = 1
sparse_action_horizon = 16

# 以上这些参数可以从yaml里面读取，先实现主干逻辑

yaml_path = "/home/flexiv/data/acp/.hydra/config.yaml"
ckpt_path = "/home/flexiv/data/acp/latest.ckpt"
max_steps = 3000
eval_config_path = "/home/flexiv/git/adaptive_compliance_policy/eval/eval_config.yaml"
normalizer_path = "/home/flexiv/data/acp/sparse_normalizer.pkl"

# color_path = "/data/haoxiang/acp/flip_v3/scene_0001/cam_104122060902/color/1768287143577.png"

# yaml_path = "/data/haoxiang/logs/acp_logs/2026.01.20_04.50.05_flip_new_v3_conv_230/.hydra/config.yaml"
# ckpt_path = "/data/haoxiang/logs/acp_logs/2026.01.20_04.50.05_flip_new_v3_conv_230/checkpoints/latest.ckpt"
# max_steps = 3000
# # eval_config_path = "/home/flexiv/git/adaptive_compliance_policy/eval/eval_config.yaml"
# normalizer_path = "/data/haoxiang/logs/acp_logs/2026.01.20_04.50.05_flip_new_v3_conv_230/sparse_normalizer.pkl"


n_action_steps = 8  

# === 初始化 Buffer ===
# 使用 deque 来自动维护滑动窗口
buffer_rgb = deque(maxlen=sparse_obs_rgb_horizon)
buffer_pos = deque(maxlen=sparse_obs_low_dim_horizon)
buffer_rot = deque(maxlen=sparse_obs_low_dim_horizon)
buffer_wrench = deque(maxlen=sparse_obs_wrench_horizon)

action_queue = deque(maxlen=100)

# export PYRITE_CHECKPOINT_FOLDERS=/home/flexiv/data/acp

def reset_buffers():
    buffer_rgb.clear()
    buffer_pos.clear()
    buffer_rot.clear()
    buffer_wrench.clear()

def load_test_obs(color_path):
    # 1. 加载彩色图并转为 RGB (OpenCV 默认读入是 BGR)
    color_image = cv2.imread(color_path)
    if color_image is None:
        raise ValueError(f"无法加载图片: {color_path}")
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB).astype(np.uint8)

    return color_image

OmegaConf.register_new_resolver(
    "now", 
    lambda pattern: datetime.now().strftime(pattern), 
    replace=True
)

def evaluate():

    # cfg = OmegaConf.load(yaml_path)
    # policy = hydra.utils.instantiate(cfg.policy)

    # with open(eval_config_path, "r") as f:
    #     eval_config = edict(yaml.load(f, Loader = yaml.FullLoader))
    #     # 这个主要是agent相关的config

    # # load checkpoint
    # ckpt = torch.load(ckpt_path, map_location='cpu')
    # if "state_dicts" in ckpt:
    #     policy.load_state_dict(ckpt["state_dicts"]["policy"], strict=False)
    # else:
    #     print("abnormal ckpt load!")
    #     policy.load_state_dict(ckpt, strict=False)

	# 这个 config_path 需要指定为config.yaml的位置

    # 2. 加载并解析配置
    cfg = OmegaConf.load(yaml_path)
    OmegaConf.resolve(cfg) # 这一步必不可少，解析所有 ${task.name} 等变量

    # 3. 利用 Hydra 实例化整个 Policy 网络结构
    # 它会自动创建 TimmObsEncoderWithForce, DDIMScheduler, 以及 DiffusionUnet
    policy = hydra.utils.instantiate(cfg.policy)

        # --- 关键步骤：加载并传入 Normalizer ---
    if os.path.exists(normalizer_path):
        with open(normalizer_path, 'rb') as f:
            normalizer_data = pickle.load(f)
        
        policy.set_normalizer(normalizer_data)

    
    # 4. 加载权重
    payload = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(payload['state_dicts']['ema_model'])
    policy = policy.to(device)
    

    # set evaluation
    policy.eval()

    with open(eval_config_path, "r") as f:
        eval_config = edict(yaml.load(f, Loader = yaml.FullLoader))
        # 这个主要是agent相关的config

    # # initialize agent
    Agent = SingleArmAgent
    agent = Agent(**eval_config.deploy.agent)

    # evaluation rollout
    print("Ready for rollout. Press Enter to continue...")
    input()
    
    with torch.inference_mode():
        for t in range(max_steps):

            print(f"Step {t} ---------------------")
           

            rgb_raw, _ = agent.get_global_observation() # (H_raw, W_raw, 3), uint8

            # rgb_raw = load_test_obs(color_path)

            # 强制缩放到 224x224 
            # cv2.resize 接受的是 (Width, Height)
            rgb_resized = cv2.resize(rgb_raw, (224, 224), interpolation=cv2.INTER_AREA)

            # HWC 转 CHW 
            rgb = rgb_resized.transpose(2, 0, 1)        # (3, 224, 224)

            proprio = agent.get_proprio() # [x, y, z, rot6d, gripper]
            # get_proprio 已经 xyz_rot_transform 到六元数了，不用再次转换
            end_pos = proprio[:3]
            end_rot6d = proprio[3:9]

            wrench = agent.get_wrench()
            
            # 考虑steps
            if t % sparse_obs_rgb_down_sample_steps == 0:
                buffer_rgb.append(rgb)
            if t % sparse_obs_low_dim_down_sample_steps == 0:
                buffer_pos.append(end_pos)
                buffer_rot.append(end_rot6d)
            if t % sparse_obs_wrench_down_sample_steps == 0:
                buffer_wrench.append(wrench)

            # Padding: 如果是第一帧，把 Buffer 填满，防止长度不够报错
            if t == 0:
                while len(buffer_rgb) < sparse_obs_rgb_horizon: buffer_rgb.append(rgb)
                while len(buffer_pos) < sparse_obs_low_dim_horizon: buffer_pos.append(end_pos)
                while len(buffer_rot) < sparse_obs_low_dim_horizon: buffer_rot.append(end_rot6d)
                while len(buffer_wrench) < sparse_obs_wrench_horizon: buffer_wrench.append(wrench)

            if len(buffer_pos) < sparse_obs_low_dim_horizon:
                print(f"Step {t}: Buffer not ready, skipping prediction")
                continue  # 跳过，等待 buffer 填满

            # 动作队列为空，上一批动作全部执行完后再预测

            if len(action_queue) == 0:

                # ========================================
                # 🔥 观测相对化（和训练时一致）
                # ========================================
                base_pos = buffer_pos[-1]
                base_rot6d = buffer_rot[-1]
                base_pose9 = np.concatenate([base_pos, base_rot6d])
                base_SE3 = su.pose9_to_SE3(base_pose9)
                
                buffer_pos_relative = []
                buffer_rot_relative = []
                for pos, rot6d in zip(buffer_pos, buffer_rot):
                    pose9 = np.concatenate([pos, rot6d])
                    SE3 = su.pose9_to_SE3(pose9)
                    SE3_relative = su.SE3_inv(base_SE3) @ SE3
                    pose9_relative = su.SE3_to_pose9(SE3_relative)
                    buffer_pos_relative.append(pose9_relative[:3])
                    buffer_rot_relative.append(pose9_relative[3:9])
                
                # 构建 batch（使用相对化的观测）
                obs_batch = {
                    "sparse": {
                        "rgb_0": torch.from_numpy(np.stack(list(buffer_rgb))).unsqueeze(0).float().to(device),
                        "robot0_eef_pos": torch.from_numpy(np.stack(buffer_pos_relative)).unsqueeze(0).float().to(device),
                        "robot0_eef_rot_axis_angle": torch.from_numpy(np.stack(buffer_rot_relative)).unsqueeze(0).float().to(device),
                        "robot0_eef_wrench": torch.from_numpy(np.stack(list(buffer_wrench))).unsqueeze(0).float().to(device)
                    }
                }

                result,stiffness_unnorm,raw_pred = policy.predict_action(obs_batch)
                # print("Predicted raw action:", raw_pred)
                # time 维长度是 sparse_action_horizon

                all_pred_actions = result['sparse'].squeeze(0).cpu().numpy()
                # 9 for reference pose, 9 for virtual target, 1 for stiffness

                all_pred_stiff_raw = stiffness_unnorm.squeeze(0).cpu().numpy()

                # ========================================
                # 🔥 新增：将相对动作转换为绝对动作
                # ========================================
                # 相对初始位置

                current_SE3 = base_SE3

                # 遍历每一步动作，转换为绝对坐标
                all_pred_actions_absolute = []
                for i, relative_action in enumerate(all_pred_actions):
                    # 提取相对位姿和刚度
                    ref_pose9_rel = relative_action[0:9]
                    vt_pose9_rel = relative_action[9:18]
                    stiffness_val = relative_action[18]

                    # 转换为 SE3 矩阵
                    ref_SE3_rel = su.pose9_to_SE3(ref_pose9_rel)
                    vt_SE3_rel = su.pose9_to_SE3(vt_pose9_rel)

                    # 🔥 关键操作：相对 → 绝对
                    ref_SE3_abs = current_SE3 @ ref_SE3_rel
                    vt_SE3_abs = current_SE3 @ vt_SE3_rel

                    # 转回 pose9 格式
                    ref_pose9_abs = su.SE3_to_pose9(ref_SE3_abs)
                    vt_pose9_abs = su.SE3_to_pose9(vt_SE3_abs)

                    # 拼接成完整动作
                    absolute_action = np.concatenate([
                        ref_pose9_abs,      # 参考位姿（绝对）
                        vt_pose9_abs,       # 虚拟目标（绝对）
                        [stiffness_val]     # 刚度保持不变
                    ])
                    all_pred_actions_absolute.append(absolute_action)

                all_pred_actions_absolute = np.array(all_pred_actions_absolute)

                print(all_pred_actions)
                print("=" * 60)
                print(all_pred_actions_absolute)

                # 只执行前 n_action_steps
                # steps_to_execute = all_pred_actions[:n_action_steps]
                steps_to_execute = all_pred_actions_absolute[:n_action_steps]

                # 将动作推入队列
                for act in steps_to_execute:
                    action_queue.append(act)

            
            # 执行动作

            # 从队列中出队一个动作执行
            raw_action = action_queue.popleft() 

            # print("Raw action to execute:", raw_action)

            # Slice 1: Reference Pose 
            ref_pos = raw_action[0:3]
            ref_rot_6d = raw_action[3:9]

            # Slice 2: Virtual Target
            vt_pos = raw_action[9:12]
            vt_rot_6d = raw_action[12:18]

            # get step_action
            step_action = raw_action[9:18]

            # Slice 3: Stiffness
            stiffness_val = raw_action[18]

            # process stiffness

            # 2. 准备刚度参数
            K_MAX = 10000  # 硬
            K_MIN = 200.0   # 软
            K_ROT = 500   # 旋转刚度

            # 计算 k_low (模型输出 0~1 映射到 K_MIN~K_MAX)
            k_low = K_MIN + stiffness_val * (K_MAX - K_MIN)
            # k_low = K_MIN + stiffness_unnorm * (K_MAX - K_MIN)

            # print("stiffness raw:", stiffness_unnorm)
            

            # 3. --- 核心：计算 Force Frame ---
            # 向量方向：从 Ref 指向 VT
            diff = np.array(vt_pos) - np.array(ref_pos)
            dist = np.linalg.norm(diff)

            if dist < 1e-6:
                # 如果重合，没有特定方向，就用默认的世界坐标系（无旋转）
                # 刚度全向设为最硬
                rotation_matrix = np.eye(3)
                k_x = K_MAX 
            else:
                # --- 构建旋转矩阵 ---
                # 1. X轴：主方向
                x_axis = diff / dist
                
                # 2. Y轴：找一个辅助向量做叉乘
                temp_vec = np.array([0, 0, 1.0])
                if np.abs(np.dot(x_axis, temp_vec)) > 0.99: # 防止共线
                    temp_vec = np.array([0, 1.0, 0])
                
                y_axis = np.cross(x_axis, temp_vec)
                y_axis /= np.linalg.norm(y_axis)
                
                # 3. Z轴
                z_axis = np.cross(x_axis, y_axis)
                
                # 4. 组合成矩阵 (列向量)
                rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
                
                # 刚度：只有 X 轴是软的
                # k_x = k_low
                k_x = stiffness_val
                # 由于 policy 输出 pred_action 已经 unnorm 了，这里直接用

            if k_x > K_MAX:
                k_x = K_MAX

            force_frame = np.eye(4)
            force_frame[0:3, 0:3] = rotation_matrix
            stiffness_vector = [k_x, K_MAX, K_MAX, K_ROT, K_ROT, K_ROT]

            print(f"Step {t}:")
            print(f"Executing Action: {step_action} \n Force Frame: {force_frame}\n Stiffness: {stiffness_vector}")
            input("press Enter to continue...")

            # 接下来需要把数据（处理后）传给agent
            agent.action(step_action,force_frame,stiffness_vector,rotation_rep = "rotation_6d")

            # time.sleep(0.1) 在 action 
            # 可能有点长了，也可以把sleep放在这里
    
        agent.stop()


if __name__ == '__main__':
    reset_buffers()
    evaluate()
    # 考虑改成传参数的调用方法