"""Single-arm device-based evaluation agent for zihao/foar deployment."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from device.robot.flexiv_api import FlexivApi
from device.camera.realsense import RealSenseRGBDCamera
from flexiv_robot_interface import flexiv_rizon_interface as single

from utils.lowdim_window import LowDimWindowProvider
from utils.transformation import xyz_rot_transform


class SingleArmAgent:
    """
    Device-based single-arm evaluation agent.

    On the deployment machine, `device_zihao/` is expected to be mounted as the
    `device/` package. This implementation intentionally mirrors
    `eval_agent_zihao_foar.py` as closely as possible while exposing the current
    eval interfaces.
    """

    def __init__(
        self,
        robot_serial,
        camera_serial,
        gripper_port=None,
        gripper_key="width",
        init_joint_deg=None,
        init_joint_rad=None,
        init_tcp_pose=None,
        enable_init_tcp_pose=False,
        **kwargs,
    ):
        del gripper_port, gripper_key

        self.camera_serial = camera_serial
        self._gripper_width = 0.0

        print("Init robot, gripper, and camera.")
        self.robot = FlexivApi(serial=robot_serial, with_streaming=True)

        if init_joint_deg is None and init_joint_rad is not None:
            init_joint_deg = np.rad2deg(np.asarray(init_joint_rad, dtype=np.float32))
        if init_joint_deg is None:
            init_joint_deg = np.rad2deg(self.ready_pose_joints)
        init_joint_deg = np.asarray(init_joint_deg, dtype=np.float32)

        self.robot.send_joint_position(np.deg2rad(init_joint_deg))
        time.sleep(5.0)

        single.zero_ft_sensor(self.robot.robot)
        time.sleep(2.0)

        if enable_init_tcp_pose and init_tcp_pose is not None:
            target_tcp = np.asarray(init_tcp_pose, dtype=np.float32)
        else:
            target_tcp = self.ready_pose
        single.move_l_pose(self.robot.robot, pose=target_tcp, is_blocking=True)
        time.sleep(5.0)

        self.camera = RealSenseRGBDCamera(serial=camera_serial)
        for _ in range(30):
            self.camera.get_rgbd_image()
        print("Initialization Finished.")

        self._lowdim_provider: Optional[LowDimWindowProvider] = None
        enable_lowdim = bool(kwargs.get("enable_lowdim_provider", False))
        lowdim_hz = float(kwargs.get("lowdim_sample_hz", 100.0))
        lowdim_buf_secs = float(kwargs.get("lowdim_buffer_secs", 60.0))
        if enable_lowdim:
            self.enable_lowdim_provider(sample_hz=lowdim_hz, buffer_secs=lowdim_buf_secs)
            time.sleep(2.0)

    @property
    def intrinsics(self):
        return np.array(
            [
                [910.6730957, 0.0, 655.33905029],
                [0.0, 908.94836426, 371.05310059],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @property
    def ready_pose(self):
        return np.array(
            [
                0.5228249073028564,
                0.36680,
                0.0325,
                0.2565673291683197,
                0.6857561469078064,
                -0.6372485160827637,
                0.2404704988002777,
            ],
            dtype=np.float32,
        )

    @property
    def ready_pose_joints(self):
        return (
            np.array([39.877, -64.683, 27.603, 67.212, -43.0428, -4.469, -19.715], dtype=np.float32)
            / 180.0
            * np.pi
        )

    @property
    def ready_rot_6d(self):
        return np.array([-1, 0, 0, 0, 1, 0], dtype=np.float32)

    def get_observation(self):
        colors, depths = self.camera.get_rgbd_image()
        return colors, depths

    def get_global_observation(self):
        return self.get_observation()

    def set_tcp_pose(self, pose, rotation_rep, rotation_rep_convention=None, blocking=False):
        tcp_pose = xyz_rot_transform(
            pose,
            from_rep=rotation_rep,
            to_rep="quaternion",
            from_convention=rotation_rep_convention,
        )
        self.robot.send_tcp_pose(tcp_pose)
        if blocking:
            time.sleep(0.02)

    def get_tcp_pose(self):
        return self.robot.get_tcp_pose()

    def get_tcp_ft(self):
        return self.robot.get_force_torque_tcp()

    def get_proprio(self, rotation_rep="rotation_6d", rotation_rep_convention=None, with_joint=False):
        tcp_pose = self.get_tcp_pose()
        tcp_pose = xyz_rot_transform(
            tcp_pose,
            from_rep="quaternion",
            to_rep=rotation_rep,
            to_convention=rotation_rep_convention,
        )
        gripper_width = np.float32(self._gripper_width)

        proprio = np.concatenate([tcp_pose, [gripper_width]], axis=0)
        if with_joint:
            joint_pos = self.robot.get_joint_pos()
            proprio_joint = np.concatenate([joint_pos, [gripper_width]], axis=0)
            return proprio, proprio_joint
        return proprio

    def action(self, action, rotation_rep="rotation_6d", rotation_rep_convention=None):
        tcp_pose = xyz_rot_transform(
            action[:9],
            from_rep=rotation_rep,
            to_rep="quaternion",
            from_convention=rotation_rep_convention,
        )
        self.robot.send_tcp_pose(tcp_pose)
        time.sleep(0.02)
        self._gripper_width = 0.0

    def enable_lowdim_provider(self, sample_hz: float = 100.0, buffer_secs: float = 60.0):
        if self._lowdim_provider is not None:
            return
        self._lowdim_provider = LowDimWindowProvider(
            get_pose_fn=self.get_tcp_pose,
            get_ft_fn=self.get_tcp_ft,
            sample_hz=sample_hz,
            buffer_secs=buffer_secs,
        )

    def disable_lowdim_provider(self):
        if self._lowdim_provider is not None:
            try:
                self._lowdim_provider.stop()
            except Exception:
                pass
            self._lowdim_provider = None

    def get_ft_window(self, *, freq, length, coordinate, projector, remove_first=True):
        if self._lowdim_provider is None:
            print("LowDim provider is not enabled.")
            return None
        return self._lowdim_provider.get_ft_window(
            freq=freq,
            length=int(length),
            coordinate=coordinate,
            projector=projector,
            remove_first=bool(remove_first),
        )

    def get_tcp_window(self, *, freq, length, coordinate, projector, remove_first=True):
        if self._lowdim_provider is None:
            print("LowDim provider is not enabled.")
            return None
        return self._lowdim_provider.get_tcp_window(
            freq=freq,
            length=int(length),
            coordinate=coordinate,
            projector=projector,
            remove_first=bool(remove_first),
        )

    def stop(self):
        self.robot.stop()
        self.disable_lowdim_provider()
