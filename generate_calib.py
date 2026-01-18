import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. 常量定义
# ==========================================
INHAND_SERIAL = "043322070878"
GLOBAL_SERIALS = ["104122063550"]

# 原始矩阵 (From Constants): T_cam_to_tcp
# 物理含义: 相机坐标系看 TCP 的位置
T_CAM_TO_TCP_RAW = np.array([
    [0, -1, 0, 0],
    [1, 0, 0, 0.077],
    [0, 0, 1, 0.2665],
    [0, 0, 0, 1]
])

# 我们需要: T_tcp_to_cam (从 TCP 推导 Camera 位置)
# 所以必须求逆！
T_TCP_TO_CAM = np.linalg.inv(T_CAM_TO_TCP_RAW)

# ==========================================
# 2. 辅助函数
# ==========================================
def pose7d_to_matrix_forced_down(pose_7d):
    """
    强制重排四元数，确保 TCP 朝下。
    原始数据: [..., q0, q1, q2, q3]
    其中 q2=0.999 (index 6 in raw data).
    
    解释为 [w, x, y, z] -> y=1 -> 绕 Y 轴转 180 -> Z 轴朝下 (正确!)
    Scipy 需要 [x, y, z, w]
    """
    pose_7d = np.array(pose_7d).flatten()
    t = pose_7d[:3]
    quat_raw = pose_7d[3:] # [w, x, y, z]
    
    w, x, y, z = quat_raw[0], quat_raw[1], quat_raw[2], quat_raw[3]
    quat_scipy = [x, y, z, w]
    
    r_mat = R.from_quat(quat_scipy).as_matrix()
    
    T = np.eye(4)
    T[:3, :3] = r_mat
    T[:3, 3] = t
    return T

# ==========================================
# 3. 主程序
# ==========================================
def generate_final_calibration(base_dir, output_name="final_calibration.npy"):
    print(f"[-] 正在处理目录: {base_dir}")
    
    p_ext = os.path.join(base_dir, "extrinsics.npy")
    p_int = os.path.join(base_dir, "intrinsics.npy")
    p_tcp = os.path.join(base_dir, "tcp.npy")
    
    extrinsics_dict = np.load(p_ext, allow_pickle=True).item()
    intrinsics_dict = np.load(p_int, allow_pickle=True).item()
    tcp_vec = np.load(p_tcp, allow_pickle=True)

    # 1. 计算 TCP (强制朝下)
    T_base_tcp = pose7d_to_matrix_forced_down(tcp_vec)
    
    # Debug TCP
    print(f"[-] [TCP] Z高度: {T_base_tcp[2, 3]:.3f}m")
    # 检查 Z 轴向量
    z_axis = T_base_tcp[:3, 2]
    print(f"    [TCP] Z轴向量: {z_axis} (预期接近 -1, 即朝下)")

    # 2. 获取手眼相机观测 (Cam -> Marker)
    if INHAND_SERIAL not in extrinsics_dict:
        raise ValueError("缺少手眼相机数据")
    # 自动解包 list
    val = extrinsics_dict[INHAND_SERIAL]
    T_cam_to_marker_inhand = val[0] if isinstance(val, list) else val

    # 3. 计算标定板位置
    # 链条: Base -> TCP -> Cam -> Marker
    # T_base_marker = T_base_tcp * T_tcp_cam * T_cam_marker
    T_base_marker = T_base_tcp @ T_TCP_TO_CAM @ T_cam_to_marker_inhand
    
    print(f"[-] [标定板] 推算高度 Z: {T_base_marker[2, 3]:.4f} 米")
    
    # 验证逻辑
    h_marker = T_base_marker[2, 3]

    # 4. 组装结果
    final_data = {
        "type": "robot",
        "camera_serials": [INHAND_SERIAL] + GLOBAL_SERIALS,
        "camera_serials_global": GLOBAL_SERIALS,
        "camera_serial_inhand": INHAND_SERIAL,
        "intrinsics": {},
        "camera_to_robot": {} 
    }

    # 内参
    for s in final_data["camera_serials"]:
        if s in intrinsics_dict:
            val = intrinsics_dict[s]
            K = val[0] if isinstance(val, list) else val
            final_data["intrinsics"][s] = K[:3, :3].astype(np.float32)

    # 5. 计算全局相机
    for gid in GLOBAL_SERIALS:
        if gid not in extrinsics_dict:
            continue
            
        val = extrinsics_dict[gid]
        T_cam_to_marker_global = val[0] if isinstance(val, list) else val
        
        # 链条: Base -> Marker -> Global
        # 已知: Base -> Marker (T_base_marker)
        # 已知: Global -> Marker (T_cam_to_marker_global)
        # 求: Base -> Global
        # T_base_global = T_base_marker * inv(T_global_marker)
        T_base_global = T_base_marker @ np.linalg.inv(T_cam_to_marker_global)
        
        final_data["camera_to_robot"][gid] = T_base_global.astype(np.float32)
        
        z_height = T_base_global[2, 3]
        print(f"[+] [全局相机] {gid}")
        print(f"    高度 Z: {z_height:.4f} 米")
        print(f"    Z轴向量: {T_base_global[:3, 2]}")
        

    # 保存
    np.save(output_name, final_data, allow_pickle=True)
    print(f"[-] 文件已保存: {output_name}")

if __name__ == "__main__":
    DIR = "/data/haoxiang/realdata_sampled_20251206/calib/1765003071096"
    generate_final_calibration(DIR)