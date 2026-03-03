"""
RISE2 mask-aware 渲染定制层。

对 airexo 的 RobotRenderer 做最小化扩展：
渲染时排除 Robotiq 2F-85 gripper 部分，使 render_mask() 只覆盖机械臂连杆（link1..7），
避免夹爪区域的像素被错误地标记为"机械臂遮挡"。

robot_inhand.urdf 中 gripper 相关 link 的命名规律（均被排除）：
  *_flange
  *_robotiq_2f85_base_link
  *_{left,right}_{inner,outer}_{finger,knuckle}

保留的 link：base, *_base_link, *_link1 .. *_link7

用法（与直接使用 RobotRenderer 相同）：
    from mask.renderer import ArmOnlyRobotRenderer
    renderer = ArmOnlyRobotRenderer(
        left_joint_cfgs=left_cfg,
        right_joint_cfgs=right_cfg,
        cam_to_base=cam_to_base,
        intrinsic=intrinsic,
        urdf_file="airexo/airexo/urdf_models/robot/robot_inhand.urdf",
        width=1280, height=720,
    )
    renderer.update_joints(left_joint, right_joint)
    mask = renderer.render_mask()  # uint8 (H, W)，255 = 机械臂像素（不含 gripper）
"""

import numpy as np

from airexo.airexo.helpers.renderer import RobotRenderer
from airexo.airexo.helpers.constants import (
    O3D_RENDER_TRANSFORMATION,
    ROBOT_PREDEFINED_TRANSFORMATION,
)
from airexo.airexo.helpers import urdf_robot as _robot_helper

# link 名中包含以下任意子串，则判定为 gripper 部件并跳过
_GRIPPER_KEYWORDS = ("finger", "knuckle", "robotiq", "flange")


class ArmOnlyRobotRenderer(RobotRenderer):
    """
    RobotRenderer 子类：渲染时排除 gripper link，只保留机械臂连杆。

    __init__ 和 _update_geometry 各自过滤一次：
    - __init__：父类初始化完成后，把已加入场景的 gripper mesh 从场景和缓存中移除。
    - _update_geometry：每帧更新时跳过 gripper link，不将其重新加入场景。
    其余接口（update_joints / render_depth / render_mask）完全继承自 RobotRenderer，无需改动。
    """

    @staticmethod
    def _is_gripper(link_name: str) -> bool:
        """判断 link 是否属于 gripper 部件。"""
        return any(kw in link_name for kw in _GRIPPER_KEYWORDS)

    def __init__(self, *args, **kwargs):
        # 调用父类初始化（会把所有 link 的 mesh 加入场景）
        super().__init__(*args, **kwargs)

        # mesh_name 格式为 "{link}///{geom_param}"，取 /// 前的部分即 link 名
        gripper_names = [
            n for n in list(self.model_meshes)
            if self._is_gripper(n.split("///")[0])
        ]
        for name in gripper_names:
            self.renderer.scene.remove_geometry(name)
            del self.model_meshes[name]
            del self.last_transforms[name]

    def _update_geometry(self, joints=None):
        if joints is None:
            joints = self.last_joints
        left_joint, right_joint = joints

        # FK：复用父类同款调用，with_visuals_map=False 因为 visuals_map 已在 __init__ 缓存
        cur_transforms = _robot_helper.forward_kinematic(
            left_joint       = left_joint,
            right_joint      = right_joint,
            left_joint_cfgs  = self.left_joint_cfgs,
            right_joint_cfgs = self.right_joint_cfgs,
            is_rad           = True,
            urdf_file        = self.urdf_file,
            with_visuals_map = False,
        )

        self.renderer.scene.clear_geometry()
        for link, transform in cur_transforms.items():
            if self._is_gripper(link):  # 跳过 gripper link
                continue
            for v in self.visuals_map[link]:
                if v.geom_param is None:
                    continue
                mesh_name = "{}///{}".format(link, v.geom_param)
                if mesh_name not in self.model_meshes:
                    # 防御：gripper mesh 已在 __init__ 中从缓存删除，此处不应出现
                    continue
                tf = (
                    O3D_RENDER_TRANSFORMATION
                    @ self.cam_to_base
                    @ ROBOT_PREDEFINED_TRANSFORMATION
                    @ transform.matrix()
                    @ v.offset.matrix()
                )
                self.model_meshes[mesh_name].transform(
                    tf @ np.linalg.inv(self.last_transforms[mesh_name])
                )
                self.model_meshes[mesh_name].compute_vertex_normals()
                self.last_transforms[mesh_name] = tf
                self.renderer.scene.add_geometry(
                    mesh_name, self.model_meshes[mesh_name], self.material
                )
