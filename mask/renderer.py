"""
RISE2 Robot Renderer for mask-aware inference.

Adapted from airexo/airexo/helpers/renderer.py and urdf_robot.py.
Self-contained: does not import from the airexo submodule.

Usage:
    renderer = SeparateRiseRobotRenderer(
        cam_to_left_base  = <4x4 np.ndarray>,
        cam_to_right_base = <4x4 np.ndarray>,
        intrinsic         = <3x3 np.ndarray>,
        width=1280, height=720,
        num_robot_joints  = 7,
        urdf_left  = "airexo/airexo/urdf_models/robot/left_robot.urdf",
        urdf_right = "airexo/airexo/urdf_models/robot/right_robot.urdf",
    )
    renderer.update_joints(left_joint_rad, right_joint_rad)
    mask = renderer.render_mask()   # uint8 (H, W), 255 = arm pixel
"""

import os
import math
import numpy as np
import open3d as o3d
import kinpy as kp

# ---------------------------------------------------------------------------
# Constants (adapted from airexo/airexo/helpers/constants.py)
# ---------------------------------------------------------------------------

# Open3D offscreen renderer uses a flipped coordinate convention.
O3D_RENDER_TRANSFORMATION = np.array(
    [
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1],
    ], dtype=np.float32
)

# Unified dual-arm URDF base → physical robot base
ROBOT_PREDEFINED_TRANSFORMATION = np.array(
    [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32
)

# Left-arm URDF mounting rotation
LEFT_ROBOT_PREDEFINED_TRANSFORMATION = np.array(
    [
        [1, 0,                0,                0],
        [0, np.sqrt(2) / 2,  np.sqrt(2) / 2,  0],
        [0, -np.sqrt(2) / 2, np.sqrt(2) / 2,  0],
        [0, 0,                0,                1],
    ], dtype=np.float32
)

# Right-arm URDF mounting rotation
RIGHT_ROBOT_PREDEFINED_TRANSFORMATION = np.array(
    [
        [1, 0,               0,               0],
        [0, np.sqrt(2) / 2, -np.sqrt(2) / 2, 0],
        [0, np.sqrt(2) / 2,  np.sqrt(2) / 2, 0],
        [0, 0,               0,               1],
    ], dtype=np.float32
)

# ---------------------------------------------------------------------------
# FK helpers (adapted from airexo/airexo/helpers/urdf_robot.py)
# Robotiq 2F-85 gripper + Flexiv Rizon 4 arm
# ---------------------------------------------------------------------------

def _calc_robotiq_state(open_length):
    return 0.725 - math.asin(open_length / 0.1143)


def _convert_robotiq_gripper_joint_state(gripper_width):
    gripper_width = float(np.clip(gripper_width, 0.0, 0.085))
    state = _calc_robotiq_state(gripper_width)
    return {
        "finger_joint":              state,
        "left_outer_finger_joint":   0,
        "left_inner_knuckle_joint":  state,
        "left_inner_finger_joint":   -state,
        "right_inner_knuckle_joint": -state,
        "right_inner_finger_joint":  state,
        "right_outer_knuckle_joint": -state,
        "right_outer_finger_joint":  0,
    }


def _convert_joint_states_single(joint, num_robot_joints, is_rad=True):
    """
    Parameters
    ----------
    joint : array-like, shape (num_robot_joints + 1,)
        First num_robot_joints entries are arm joint angles;
        last entry is gripper width in meters.
    num_robot_joints : int
        Number of arm joints (7 for Flexiv Rizon 4).
    is_rad : bool
        Whether arm joints are in radians (True) or degrees (False).
    """
    joint_states = {}
    for idx in range(num_robot_joints):
        val = float(joint[idx])
        if not is_rad:
            val = val / 180.0 * math.pi
        joint_states["joint{}".format(idx + 1)] = val

    gripper_states = _convert_robotiq_gripper_joint_state(float(joint[num_robot_joints]))
    return {**joint_states, **gripper_states}


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class SeparateRiseRobotRenderer:
    """
    URDF-based renderer for a dual-arm robot using two separate URDF files
    (one per arm). Renders the arm mesh into depth / mask images.

    The kinpy model chains represent the robot's static kinematic structure
    (parsed from URDF) and are built once in __init__.  Each call to
    _update_geometry passes the current joint angles to chain.forward_kinematics(),
    which returns the per-link transforms for that configuration.  There is no
    need to rebuild the chain per frame — the chain itself is stateless with
    respect to joint configuration.

    Parameters
    ----------
    cam_to_left_base : np.ndarray (4, 4)
        Transformation from camera frame to left robot base frame.
    cam_to_right_base : np.ndarray (4, 4)
        Transformation from camera frame to right robot base frame.
    intrinsic : np.ndarray (3, 3)
        Camera intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]].
    width, height : int
        Rendered image resolution (should match the real camera).
    num_robot_joints : int
        Number of arm joints (default 7 for Flexiv Rizon 4).
    urdf_left : str
        Path to the left arm URDF file.
    urdf_right : str
        Path to the right arm URDF file.
    near_plane, far_plane : float
        Clipping planes for the perspective projection.
    """

    def __init__(
        self,
        cam_to_left_base,
        cam_to_right_base,
        intrinsic,
        width=1280,
        height=720,
        num_robot_joints=7,
        urdf_left=os.path.join(
            "airexo", "airexo", "urdf_models", "robot", "left_robot.urdf"
        ),
        urdf_right=os.path.join(
            "airexo", "airexo", "urdf_models", "robot", "right_robot.urdf"
        ),
        near_plane=0.1,
        far_plane=20.0,
    ):
        self.cam_to_left_base = np.asarray(cam_to_left_base, dtype=np.float32)
        self.cam_to_right_base = np.asarray(cam_to_right_base, dtype=np.float32)
        self.num_robot_joints = num_robot_joints
        self.urdf_left = urdf_left
        self.urdf_right = urdf_right

        # Open3D offscreen renderer + material
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = "defaultLit"

        # Build kinpy chains once from URDF (kinematic structure is static).
        # chain.forward_kinematics(joint_states) is then called each frame with
        # the current joint angles — the chain itself holds no joint state.
        self.chain_left  = kp.build_chain_from_urdf(open(urdf_left,  "rb").read())
        self.chain_right = kp.build_chain_from_urdf(open(urdf_right, "rb").read())
        self.visuals_map_left  = self.chain_left.visuals_map()
        self.visuals_map_right = self.chain_right.visuals_map()

        # Populate initial scene at zero pose
        zero_joint  = np.zeros(num_robot_joints + 1, dtype=np.float32)
        zero_states = _convert_joint_states_single(zero_joint, num_robot_joints)
        cur_transforms_left  = self.chain_left.forward_kinematics(zero_states)
        cur_transforms_right = self.chain_right.forward_kinematics(zero_states)
        self.last_joints = (zero_joint.copy(), zero_joint.copy())

        # Load meshes and add to scene at initial pose
        self.model_meshes_left = {}
        self.last_transforms_left = {}
        for link, transform in cur_transforms_left.items():
            for v in self.visuals_map_left[link]:
                if v.geom_param is None:
                    continue
                mesh_name = "left///{}///{}".format(link, v.geom_param)
                tf = (
                    O3D_RENDER_TRANSFORMATION
                    @ self.cam_to_left_base
                    @ ROBOT_PREDEFINED_TRANSFORMATION
                    @ LEFT_ROBOT_PREDEFINED_TRANSFORMATION
                    @ transform.matrix()
                    @ v.offset.matrix()
                )
                mesh = o3d.io.read_triangle_mesh(
                    os.path.join(os.path.dirname(self.urdf_left), v.geom_param)
                )
                mesh.transform(tf)
                mesh.compute_vertex_normals()
                self.model_meshes_left[mesh_name] = mesh
                self.last_transforms_left[mesh_name] = tf
                self.renderer.scene.add_geometry(mesh_name, mesh, self.material)

        self.model_meshes_right = {}
        self.last_transforms_right = {}
        for link, transform in cur_transforms_right.items():
            for v in self.visuals_map_right[link]:
                if v.geom_param is None:
                    continue
                mesh_name = "right///{}///{}".format(link, v.geom_param)
                tf = (
                    O3D_RENDER_TRANSFORMATION
                    @ self.cam_to_right_base
                    @ ROBOT_PREDEFINED_TRANSFORMATION
                    @ RIGHT_ROBOT_PREDEFINED_TRANSFORMATION
                    @ transform.matrix()
                    @ v.offset.matrix()
                )
                mesh = o3d.io.read_triangle_mesh(
                    os.path.join(os.path.dirname(self.urdf_right), v.geom_param)
                )
                mesh.transform(tf)
                mesh.compute_vertex_normals()
                self.model_meshes_right[mesh_name] = mesh
                self.last_transforms_right[mesh_name] = tf
                self.renderer.scene.add_geometry(mesh_name, mesh, self.material)

        # Camera projection
        self.renderer.scene.camera.set_projection(
            intrinsic, near_plane, far_plane, float(width), float(height)
        )

    def _update_geometry(self, joints=None):
        if joints is None:
            joints = self.last_joints
        left_joint, right_joint = joints

        # Forward kinematics with the current joint angles, using cached chains
        left_states  = _convert_joint_states_single(left_joint,  self.num_robot_joints)
        right_states = _convert_joint_states_single(right_joint, self.num_robot_joints)
        cur_transforms_left  = self.chain_left.forward_kinematics(left_states)
        cur_transforms_right = self.chain_right.forward_kinematics(right_states)

        self.renderer.scene.clear_geometry()

        for link, transform in cur_transforms_left.items():
            for v in self.visuals_map_left[link]:
                if v.geom_param is None:
                    continue
                mesh_name = "left///{}///{}".format(link, v.geom_param)
                tf = (
                    O3D_RENDER_TRANSFORMATION
                    @ self.cam_to_left_base
                    @ ROBOT_PREDEFINED_TRANSFORMATION
                    @ LEFT_ROBOT_PREDEFINED_TRANSFORMATION
                    @ transform.matrix()
                    @ v.offset.matrix()
                )
                self.model_meshes_left[mesh_name].transform(
                    tf @ np.linalg.inv(self.last_transforms_left[mesh_name])
                )
                self.model_meshes_left[mesh_name].compute_vertex_normals()
                self.last_transforms_left[mesh_name] = tf
                self.renderer.scene.add_geometry(
                    mesh_name, self.model_meshes_left[mesh_name], self.material
                )

        for link, transform in cur_transforms_right.items():
            for v in self.visuals_map_right[link]:
                if v.geom_param is None:
                    continue
                mesh_name = "right///{}///{}".format(link, v.geom_param)
                tf = (
                    O3D_RENDER_TRANSFORMATION
                    @ self.cam_to_right_base
                    @ ROBOT_PREDEFINED_TRANSFORMATION
                    @ RIGHT_ROBOT_PREDEFINED_TRANSFORMATION
                    @ transform.matrix()
                    @ v.offset.matrix()
                )
                self.model_meshes_right[mesh_name].transform(
                    tf @ np.linalg.inv(self.last_transforms_right[mesh_name])
                )
                self.model_meshes_right[mesh_name].compute_vertex_normals()
                self.last_transforms_right[mesh_name] = tf
                self.renderer.scene.add_geometry(
                    mesh_name, self.model_meshes_right[mesh_name], self.material
                )

        self.last_joints = (
            np.asarray(left_joint, dtype=np.float32),
            np.asarray(right_joint, dtype=np.float32),
        )

    def update_joints(self, left_joint, right_joint):
        """
        Update arm joint states and refresh the rendered scene.

        Parameters
        ----------
        left_joint, right_joint : array-like, shape (num_robot_joints + 1,)
            Arm joint angles in radians followed by gripper width in meters.
        """
        self._update_geometry(joints=(left_joint, right_joint))

    def render_depth(self):
        """Return depth map (float32, z in view space, inf where no geometry)."""
        return np.asarray(
            self.renderer.render_to_depth_image(z_in_view_space=True), dtype=np.float32
        )

    def render_mask(self, depth=None):
        """
        Return binary mask (uint8).
        255 = robot arm pixel, 0 = background.
        """
        if depth is None:
            depth = self.render_depth()
        mask = np.zeros(depth.shape, dtype=np.uint8)
        mask[depth < np.inf] = 255
        return mask
