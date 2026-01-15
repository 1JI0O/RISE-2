import os
import numpy as np
import shutil
import json
from tqdm import tqdm

# ================= 配置区域 (请修改这里) =================

# 1. 你的原始数据路径 (RISE-1)
SRC_ROOT = "/data/haoxiang/realdata_sampled_20251206"
SRC_TRAIN = os.path.join(SRC_ROOT, "train")
SRC_CALIB = os.path.join(SRC_ROOT, "calib")

# 2. 输出路径 (转换后的数据放这里)
DST_ROOT = "/data/haoxiang/realdata_rise2_ready"
DST_TRAIN = os.path.join(DST_ROOT, "train")
DST_CALIB = os.path.join(DST_ROOT, "calib")

# 3. 相机序列号
GLOBAL_CAM_ID = "104122063550"  # 全局相机
INHAND_CAM_ID = "043322070878"  # 手眼相机

# =======================================================

def make_symlink(src, dst):
    """创建软链接，瞬间完成，不占空间"""
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    os.symlink(src, dst)

def convert_calibration():
    """ 标定转换 """

    # intrinsics 对应 RISE1 - calib - 时间戳 - intrinsics.npy
    # camera_to_robot 对应 extrinsics.npy GLOBAL_CAM_ID

    print(f"--- 正在转换标定文件 ---")
    if not os.path.exists(DST_CALIB):
        os.makedirs(DST_CALIB)

    for ts_folder in os.listdir(SRC_CALIB):
        src_folder = os.path.join(SRC_CALIB, ts_folder)
        if not os.path.isdir(src_folder): continue

        try:
            # 读取原始矩阵
            ext_raw = np.load(os.path.join(src_folder, "extrinsics.npy"), allow_pickle=True).item()
            intr_raw = np.load(os.path.join(src_folder, "intrinsics.npy"), allow_pickle=True).item()

            # 这个 intr_raw 第四列全是 0 要转化成3x3的
            # 从 intr_raw 中提取对应 ID 的矩阵并切片
            intrinsics_processed = {
                GLOBAL_CAM_ID: intr_raw[GLOBAL_CAM_ID][:3, :3].astype(np.float32),
                INHAND_CAM_ID: intr_raw[INHAND_CAM_ID][:3, :3].astype(np.float32)
            }

            # 处理 extrinsics 读出 GLOBAL_CAM_ID 对应外参
            # extrinsics 里的格式是 {'id': [array(...)]}，所以要取 [0]
            global_ext = ext_raw[GLOBAL_CAM_ID][0].astype(np.float32)

            # 构造字典
            calib_dict = {
                'type': 'robot',
                'camera_serials': [GLOBAL_CAM_ID, INHAND_CAM_ID],
                'camera_serials_global': [GLOBAL_CAM_ID],
                'camera_serial_inhand': INHAND_CAM_ID, 
                
                'intrinsics': intrinsics_processed,
                
                'camera_to_robot': {
                    GLOBAL_CAM_ID: global_ext
                }
            }
            
            dst_path = os.path.join(DST_CALIB, f"{ts_folder}.npy")
            np.save(dst_path, calib_dict)
            print(f"  - Generated: {dst_path}")

        except Exception as e:
            print(f"  [Error] 跳过 {ts_folder}: {e}")

def convert_episodes():
    """
    [数据转换]
    合并 tcp/gripper -> lowdim, 合成meta.json , 软链接图片
    """
    print(f"\n--- 正在转换训练数据 ---")
    if not os.path.exists(DST_TRAIN):
        os.makedirs(DST_TRAIN)

    # 扫描所有 scene 文件夹
    demos = [d for d in os.listdir(SRC_TRAIN) if os.path.isdir(os.path.join(SRC_TRAIN, d))]
    # demos = demos[:3]
    # test

    for demo_name in tqdm(demos):
        src_demo = os.path.join(SRC_TRAIN, demo_name)
        dst_demo = os.path.join(DST_TRAIN, demo_name)
        
        if not os.path.exists(dst_demo): os.makedirs(dst_demo)
        
        # 读取 metadata.json
        src_metadata_json = os.path.join(src_demo, "metadata.json")
        if os.path.exists(src_metadata_json):
            with open(src_metadata_json, "r") as f:
                meta_content = json.load(f)

        # 读取 timestamp.txt
        src_timestamp_txt = os.path.join(src_demo, "timestamp.txt")
        if os.path.exists(src_timestamp_txt):
            with open(src_timestamp_txt, "r") as f:
                # .strip() 去除换行符和空格
                calib_ts = f.read().strip()
                # 按照 dataset 用法，Key 必须叫 "calib_timestamp"
                meta_content["calib_timestamp"] = calib_ts
        else:
            print(f"警告: {demo_name} 缺少 timestamp.txt，无法关联标定文件！")

        # 合成 meta.json
        dst_meta_json = os.path.join(dst_demo, "meta.json")
        with open(dst_meta_json, "w") as f:
            json.dump(meta_content, f, indent=4)

        # 链接图片 (Cam文件夹)
        for cam_id in [GLOBAL_CAM_ID, INHAND_CAM_ID]:
            src_cam = os.path.join(src_demo, f"cam_{cam_id}")
            dst_cam = os.path.join(dst_demo, f"cam_{cam_id}")
            
            if not os.path.exists(src_cam): 
                tqdm.write(f"[MISSING] 相机目录不存在: {src_cam}")
                continue
            if not os.path.exists(dst_cam): os.makedirs(dst_cam)

            # 链接 color 和 depth
            for type_dir in ["color", "depth"]:
                src_type_dir = os.path.join(src_cam, type_dir)
                dst_type_dir = os.path.join(dst_cam, type_dir)

                if os.path.exists(src_type_dir):
                    make_symlink(src_type_dir, dst_type_dir)
                else:
                    tqdm.write(f"[MISSING] 数据目录不存在: {src_type_dir}")

        # 生成 Lowdim 数据 (清洗核心)
        dst_lowdim = os.path.join(dst_demo, "lowdim")
        if not os.path.exists(dst_lowdim): os.makedirs(dst_lowdim)

        # rise1数据中，inhand和global相机都有tcp和gripper_command，时间戳还不一样
        # 按照dataset逻辑，需要都转化成一个对应的npy
        
        for cam_id in [GLOBAL_CAM_ID, INHAND_CAM_ID]:
            src_tcp_dir = os.path.join(src_demo, f"cam_{cam_id}", "tcp")
            src_grip_dir = os.path.join(src_demo, f"cam_{cam_id}", "gripper_command")

            if not os.path.exists(src_tcp_dir): 
                tqdm.write(f"[MISSING] tcp目录不存在: {src_tcp_dir}")
                continue
            
             # --- 在这里增加一个内部进度条 ---
            fnames = [f for f in os.listdir(src_tcp_dir) if f.endswith('.npy')]
            for fname in tqdm(fnames, desc=f"      Lifting lowdim (Cam {cam_id})", leave=False):

                try:
                    tcp_raw = np.load(os.path.join(src_tcp_dir, fname))  # 13维
                    grip_path = os.path.join(src_grip_dir, fname)

                    if os.path.exists(grip_path):
                        grip_raw = np.load(grip_path)
                        # 归一化示例：将 0-1000 映射到 0-0.095米
                        grip_val = float(grip_raw[0]) / 1000.0 * 0.095
                    else:
                        tqdm.write(f"[MISSING] grip文件不存在: {grip_path}")
                        grip_val = 0.0

                    # 构造 RISE-2 标准 33(robot) + 2(gripper) 字典

                    """
                    在 Dataset 中, self.robot_type 会决定 load_action 只取 robot_left 还是两个都取
                    如果是单臂, load_action 会从这个字典里抽取出 robot_left[:7] 和 gripper_left[0]
                    拼成一个长度为 8 的向量传给 getitem
                    """

                    robot_left = np.zeros(33, dtype=np.float32)
                    robot_left[:7] = tcp_raw[:7] # 前7位是[x,y,z,qx,qy,qz,qw]

                    # dataset读取逻辑 : 只读取第一个数，所以剩余的可以都填充0
                    gripper_left = np.zeros(2, dtype=np.float32)
                    gripper_left[0] = grip_val

                    rise2_dict = {
                        'robot_left': robot_left,
                        'gripper_left': gripper_left,

                        # 以下数据都用不到，为了格式正确，全部填充0
                        'airexo_left': np.zeros(8, dtype=np.float32),
                        'robot_right': np.zeros(33, dtype=np.float32),
                        'gripper_right': np.zeros(2, dtype=np.float32),
                        'airexo_right': np.zeros(8, dtype=np.float32)
                    }

                    # 都保存到 lowdim 目录
                    np.save(os.path.join(dst_lowdim, fname), rise2_dict)

                except Exception as e:
                    tqdm.write(f"  [Error] Lowdim 失败 {fname} (Cam {cam_id}): {e}")


if __name__ == "__main__":
    convert_calibration()
    convert_episodes()
    print(f"\n[Success] 数据已转换至: {DST_ROOT}")
    print("请在 Config 中设置 data_path 指向此路径，并将 robot_type 设为 single")
