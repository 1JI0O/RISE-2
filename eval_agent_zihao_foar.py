"""
Evaluation Agent.
"""

import time
import numpy as np
from typing import Optional
from easydict import EasyDict as edict
from device.robot.flexiv_api import FlexivApi
from utils.transformation import xyz_rot_transform
from utils.lowdim_window import LowDimWindowProvider
from device.camera.realsense import RealSenseRGBDCamera
from flexiv_robot_interface import flexiv_rizon_interface as single

class Agent:
    """
    Evaluation agent with Flexiv arm, Flexiv gripper and Intel RealSense RGB-D camera.

    Follow the implementation here to create your own real-world evaluation agent.
    """
    def __init__(
        self,
        robot_serial,
        camera_serial,
        **kwargs
    ): 
        self.camera_serial = camera_serial

        print("Init robot, gripper, and camera.")
        self.robot = FlexivApi(serial=robot_serial, with_streaming = True)

        self.robot.send_joint_position(self.ready_pose_joints)
        time.sleep(5.0)

        single.zero_ft_sensor(self.robot.robot)
        time.sleep(2.0)

        single.move_l_pose(self.robot.robot, pose=self.ready_pose, is_blocking=True)
        time.sleep(5.0)

        # single.zero_ft_sensor(self.robot.robot)
        # time.sleep(2.0)

        # self.robot.set_max_contact_wrench([10.0, 10.0, 10.0, 3.0, 3.0,3.0])
        # self.robot.set_max_contact_wrench([10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
        # self.robot.send_tcp_pose(self.ready_pose)
        # self.robot.set_max_contact_wrench([10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
        # self.robot.set_max_contact_wrench([30.0, 30.0, 30.0, 10.0, 10.0, 10.0])

        # time.sleep(1.5)

        self.camera = RealSenseRGBDCamera(serial = camera_serial)
        for _ in range(30): 
            self.camera.get_rgbd_image()
        print("Initialization Finished.")

        # -------- Optional FT provider --------
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
            [[910.6730957, 0.0, 655.33905029],
            [0.0, 908.94836426, 371.05310059],
            [0.0, 0.0, 1.0]]
        )
    
    @property
    def ready_pose(self):
        # flip_0323_purple_box
        # return np.array([0.4828249073028564, 0.36715561151504517, 0.06615855544805527, 0.2565673291683197, 0.6857561469078064, -0.6372485160827637, 0.2404704988002777])
    
        # flip_0326_purple_box
        return np.array([0.5228249073028564, 0.36680, 0.0325, 0.2565673291683197, 0.6857561469078064, -0.6372485160827637, 0.2404704988002777])
    
        # lowdim force proprio policy initial position
        # return np.array([0.5242411136627197, 0.13149680197238922, 0.02654320001602173, 0.25692710280418396, 0.6856518983840942, -0.6368304491043091, 0.24148906767368317])
    
    @property
    def ready_pose_joints(self):
        # flip_0312_purple_box
        # return np.array([24.855582611361513, -57.57107712857708, 21.249889889341716, 66.07185543908999, -21.601523395805618, 27.262010228117813, -43.09579574614898]) / 180. * np.pi
    
        # flip_0319_purple_box
        # return np.array([41.66233711458012, -73.22693621869601, 23.94478371555849, 44.25733088643024, -44.66060574567101, -12.657257487305937, -33.15996395489571]) / 180. * np.pi

        # flip_0326_purple_box
        return np.array([39.877, -64.683, 27.603, 67.212, -43.0428, -4.469, -19.715]) / 180. * np.pi


    @property
    def ready_rot_6d(self):
        return np.array([-1, 0, 0, 0, 1, 0])

    def get_observation(self):
        colors, depths = self.camera.get_rgbd_image()
        return colors, depths
    
    def set_tcp_pose(self, pose, rotation_rep, rotation_rep_convention = None, blocking = False):
        tcp_pose = xyz_rot_transform(
            pose,
            from_rep = rotation_rep, 
            to_rep = "quaternion",
            from_convention = rotation_rep_convention
        )
        # print("Moving to pose:", tcp_pose)
        # input("Press Enter to move to the next pose...")
        self.robot.send_tcp_pose(tcp_pose)
        if blocking:
            time.sleep(0.02)
            # time.sleep(0.05)
            
    def get_tcp_pose(self):
        tcp_pose = self.robot.get_tcp_pose()
        return tcp_pose
    
    def get_tcp_ft(self):
        return self.robot.get_force_torque_tcp()
    
    def set_gripper_width(self, width, blocking = True):
        width = int(np.clip(width / 0.095 * 1000., 0, 1000))
        while True:
            try:
                self.gripper.set_width(width)
                break
            except:
                print("set_gripper_width error")
        time1 = time.time()
        while True:
            time.sleep(0.1)
            if self.get_gripper_width()-width/1000.*0.095 < 0.005:
                break
            if time.time()-time1 > 0.5:
                break
        # if blocking:
        #     time.sleep(0.5)
    
    
    def get_gripper_width(self):
        while True:
            try:
                width = self.gripper.get_info()[0]
                break
            except:
                print("get_gripper_width error")
        return width / 1000. * 0.095
    
    def enable_lowdim_provider(self, sample_hz: float = 100.0, buffer_secs: float = 60.0):
        if self._lowdim_provider is not None:
            return
        if LowDimWindowProvider is None:
            raise ImportError("LowDimWindowProvider not found")
        self._lowdim_provider = LowDimWindowProvider(
            get_pose_fn=self.get_tcp_pose,
            get_ft_fn=self.get_tcp_ft,
            sample_hz=sample_hz,
            buffer_secs=buffer_secs
        )

    def disable_lowdim_provider(self):
        if self._lowdim_provider is not None:
            try:
                self._lowdim_provider.stop()
            except Exception:
                pass
            self._lowdim_provider = None

    def get_ft_window(self, *, freq, length, coordinate, projector, remove_first=True):
        """
        Returns a ft:(length,6), tcps:(length, 7) window only if the FT provider is enabled; returns None otherwise.
        """
        if self._lowdim_provider is None:
            print("LowDim provider is not enabled.")
            return None
        return self._lowdim_provider.get_ft_window(
            freq=freq,
            length=int(length),
            coordinate=coordinate,
            projector=projector,
            remove_first=bool(remove_first)
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
            remove_first=bool(remove_first)
        )

    def stop(self):
        self.robot.stop()
        self.disable_lowdim_provider()