"""
Evaluation Agent.
"""

import time
import numpy as np
from easyrobot.easyrobot.arm.flexiv import FlexivArm
from easyrobot.easyrobot.camera.realsense import RealSenseRGBDCamera
from utils.transformation import xyz_rot_transform
from easyrobot.easyrobot.gripper.flexiv import FlexivGripper


TELEOP_INIT_POSES = {
    "Rizon4-062703": {
        "joint_pose_deg": [
            40.439804936146,
            -77.863036355298,
            -144.776131902487,
            138.65111910746,
            86.665128557928,
            -5.775515300899,
            -30.435855721378,
        ],
        "cartesian_pose": [
            0.527064859867096,
            0.29016798734664917,
            -0.18804581463336945,
            0.013102811761200428,
            -0.05158299580216408,
            0.9984447956085205,
            -0.01659703627228737,
        ],
    },
    "Rizon4R-062046": {
        "joint_pose_deg": [
            10.981961471223,
            -65.56937888323,
            -138.21245688993,
            133.838226072864,
            89.396671614662,
            -18.523809397601,
            -49.036677178363,
        ],
        "cartesian_pose": [
            0.5785843133926392,
            -0.1378270387649536,
            -0.15745601058006287,
            0.006165927276015282,
            0.0124443881213665,
            0.9997890591621399,
            0.01513077411800623,
        ],
    },
    "Rizon4-062770": {
        "joint_pose_deg": [
            39.877,
            -64.683,
            27.603,
            67.212,
            -43.0428,
            -4.469,
            -19.715,
        ],
        "cartesian_pose": [
            0.5228249073028564,
            0.36680,
            0.0325,
            0.2565673291683197,
            0.6857561469078064,
            -0.6372485160827637,
            0.2404704988002777,
        ],
    },
}


def _recover_motion_mode(robot) -> bool:
    """Try to switch robot back to a motion-capable mode."""
    switch_mode = getattr(robot, "switch_mode", None)
    if callable(switch_mode):
        for mode in [
            "NRT_CARTESIAN_MOTION_FORCE",
            "NRT_JOINT_POSITION",
            "NRT_JOINT_IMPEDANCE",
        ]:
            try:
                switch_mode(mode)
                return True
            except Exception:
                continue

    raw_robot = getattr(robot, "robot", None)
    if raw_robot is not None:
        try:
            import flexivrdk

            for mode in [
                flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE,
                flexivrdk.Mode.NRT_JOINT_POSITION,
                flexivrdk.Mode.NRT_JOINT_IMPEDANCE,
            ]:
                try:
                    raw_robot.SwitchMode(mode)
                    return True
                except Exception:
                    continue
        except Exception:
            pass
    return False


GRIPPER_MAX_WIDTH = 0.1
GRIPPER_CLOSE_WIDTH = 0.0
GRIPPER_SWITCH_WIDTH = 0.03


def _build_raw_gripper(robot, gripper):
    # MUST prefer wrapper's internal gripper object, because it's already Enable()'d.
    raw_gripper = getattr(gripper, "gripper", None)
    if raw_gripper is not None and hasattr(raw_gripper, "Move"):
        return raw_gripper

    # Fallback: construct and enable a native gripper from robot.robot.
    raw_robot = getattr(robot, "robot", None)
    if raw_robot is not None:
        try:
            import flexivrdk

            g = flexivrdk.Gripper(raw_robot)
            for name in ["GripperFlexivModbus", "Flexiv-GN01"]:
                try:
                    g.Enable(name)
                    break
                except Exception:
                    continue
            return g
        except Exception:
            pass

    return None


def _ensure_gripper_ready(
    gripper, raw_gripper, label: str = "", do_init: bool = True
) -> None:
    g = raw_gripper if raw_gripper is not None else getattr(gripper, "gripper", None)
    if g is None:
        if label:
            print(f"[{label}] no raw gripper handle")
        return

    # Best effort: ensure enabled on common names
    for name in ["GripperFlexivModbus", "Flexiv-GN01"]:
        try:
            if hasattr(g, "Enable"):
                g.Enable(name)
                if label:
                    print(f"[{label}] gripper enabled by name={name}")
                break
        except Exception:
            continue

    if do_init and hasattr(g, "Init"):
        try:
            if label:
                print(f"[{label}] gripper init start")
            g.Init()
            time.sleep(2.0)
            if label:
                print(f"[{label}] gripper init done")
        except Exception as e:
            if label:
                print(f"[{label}] gripper init skipped: {e}")


def _read_gripper_width(gripper, default=0.0) -> float:
    try:
        states = gripper.get_states()
        if isinstance(states, dict):
            v = states.get("width", default)
        else:
            v = states["width"]
        return float(np.array(v).reshape(-1)[0])
    except Exception:
        return float(default)


def _to_gripper_width(cmd: float, mode: str = "width") -> float:
    c = float(cmd)
    if mode == "action":
        # Compatible with both [-1,1] and [0,1] policy conventions.
        if 0.0 <= c <= 1.0:
            return max(
                GRIPPER_CLOSE_WIDTH, min(GRIPPER_MAX_WIDTH, c * GRIPPER_MAX_WIDTH)
            )
        c = max(-1.0, min(1.0, c))
        return (c + 1.0) * 0.5 * GRIPPER_MAX_WIDTH
    return max(GRIPPER_CLOSE_WIDTH, min(GRIPPER_MAX_WIDTH, c))


def _move_gripper(
    gripper,
    raw_gripper,
    width: float,
    speed: float = 0.1,
    force: float = 30.0,
    label: str = "",
) -> bool:
    target = _to_gripper_width(width, mode="width")
    cmd_speed = abs(float(speed))
    cmd_force = abs(float(force))

    if label:
        print(
            f"[{label}] send gripper target width={target:.4f}, "
            f"speed={cmd_speed:.3f}, force={cmd_force:.1f}"
        )

    try:
        if raw_gripper is not None and hasattr(raw_gripper, "Move"):
            # Teleop behavior: use positive force for both open and close.
            raw_gripper.Move(target, cmd_speed, cmd_force)
            return True
    except Exception as e:
        if label:
            print(f"[{label}] raw gripper Move failed: {e}")

    try:
        gripper.set_width(target)
        return True
    except Exception as e:
        if label:
            print(f"[{label}] wrapper set_width failed: {e}")

    return False


def _init_gripper_closed(gripper, raw_gripper, label: str = "") -> float:
    width = _read_gripper_width(gripper, default=GRIPPER_MAX_WIDTH)
    for i in range(3):
        _move_gripper(
            gripper,
            raw_gripper,
            GRIPPER_CLOSE_WIDTH,
            speed=0.2,
            force=50.0,
            label=label,
        )
        time.sleep(0.35)
        width = _read_gripper_width(gripper, default=GRIPPER_CLOSE_WIDTH)
        if label:
            print(f"[{label}] feedback width after close try {i + 1}: {width:.4f}")
        if width <= 0.01:
            break
    return width


def _gripper_self_test(gripper, raw_gripper, label: str = "") -> float:
    """Open then close once for visible hardware verification."""
    _move_gripper(
        gripper, raw_gripper, 0.10, speed=0.1, force=40.0, label=f"{label}-open"
    )
    time.sleep(0.4)
    w_open = _read_gripper_width(gripper, default=0.10)
    if label:
        print(f"[{label}] self-test open feedback width={w_open:.4f}")

    _move_gripper(
        gripper, raw_gripper, 0.0, speed=0.1, force=40.0, label=f"{label}-close"
    )
    time.sleep(0.5)
    w_close = _read_gripper_width(gripper, default=0.0)
    if label:
        print(f"[{label}] self-test close feedback width={w_close:.4f}")
    return w_close


class SingleArmAgent:
    """
    Evaluation single-arm agent with Flexiv arms, Dahuan gripper and an Intel RealSense RGB-D camera.

    Follow the implementation here to create your own real-world evaluation agent.
    """

    def __init__(
        self,
        robot_serial,
        gripper_port,
        camera_serial,
        gripper_key="width",
        init_joint_deg=None,
        init_joint_rad=None,
        init_tcp_pose=None,
        enable_init_tcp_pose=False,
        gripper_self_test=True,
        max_contact_wrench=[30, 30, 30, 10, 10, 10],
        max_vel=0.5,
        max_acc=2.0,
        max_angular_vel=1.0,
        max_angular_acc=5.0,
        **kwargs,
    ):
        # initialize
        self.robot = FlexivArm(robot_serial)
        self.gripper = FlexivGripper(self.robot)
        self.camera_serial = camera_serial
        self.camera = RealSenseRGBDCamera(serial=camera_serial)
        self.intrinsics = self.camera.get_intrinsic()
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.max_angular_vel = max_angular_vel
        self.max_angular_acc = max_angular_acc
        assert gripper_key in ["width", "action"]
        self.gripper_key = gripper_key

        self.last_gripper = 0

        serial_cfg = TELEOP_INIT_POSES.get(robot_serial, {})
        if init_joint_deg is None:
            init_joint_deg = serial_cfg.get("joint_pose_deg")
        if init_tcp_pose is None:
            init_tcp_pose = serial_cfg.get("cartesian_pose")

        def _move_joint_init(robot, joint_target, pose_unit: str, name: str):
            target_arr = np.array(joint_target, dtype=np.float32)
            qdeg = (
                target_arr.tolist()
                if pose_unit == "deg"
                else np.rad2deg(target_arr).tolist()
            )
            moved = False

            try:
                from flexiv_robot_interface import flexiv_rizon_interface as single

                raw_robot = getattr(robot, "robot", None)
                if raw_robot is not None:
                    try:
                        single.move_j_deg(
                            raw_robot,
                            qdeg,
                            time_out=10,
                            joint_allow_error_angle=3,
                            stiffness_ratio=0.1,
                            compliant=False,
                            is_blocking=True,
                        )
                    except TypeError:
                        single.move_j_deg(
                            raw_robot,
                            qdeg,
                            max_vel=1,
                            max_acc=1,
                            is_lock_ext_axis=False,
                            compliant=False,
                            is_blocking=True,
                        )
                    moved = True
            except Exception as e:
                print(
                    f"[SingleArmAgent] {name} raw move_j_deg failed, fallback to wrapper: {e}"
                )

            if not moved:
                robot.send_joint_pos(
                    target_arr,
                    max_vel=[0.3] * 7,
                    max_acc=[0.5] * 7,
                    impedance=False,
                    blocking=True,
                    pose_unit=pose_unit,
                )

        # move to initial pose (prefer hard-coded per-robot init joint pose, fallback to ready tcp pose)
        if init_joint_deg is not None:
            print(
                f"[SingleArmAgent] use init joint pose for {robot_serial}: "
                f"{np.array(init_joint_deg, dtype=np.float32)}"
            )
            _move_joint_init(self.robot, init_joint_deg, "deg", "single")
        elif init_joint_rad is not None:
            print(
                f"[SingleArmAgent] use init joint(rad) pose for {robot_serial}: "
                f"{np.array(init_joint_rad, dtype=np.float32)}"
            )
            _move_joint_init(self.robot, init_joint_rad, "rad", "single")
        else:
            print(
                f"[SingleArmAgent] fallback to ready tcp pose for {robot_serial}: "
                f"{self.ready_pose}"
            )
            self.robot.send_tcp_pose(
                self.ready_pose,
                max_vel=self.max_vel,
                max_acc=self.max_acc,
                max_angular_vel=self.max_angular_vel,
                max_angular_acc=self.max_angular_acc,
            )

        if enable_init_tcp_pose and init_tcp_pose is not None:
            try:
                print(
                    f"[SingleArmAgent] additionally use init tcp pose for {robot_serial}: "
                    f"{np.array(init_tcp_pose, dtype=np.float32)}"
                )
                self.robot.send_tcp_pose(
                    np.array(init_tcp_pose, dtype=np.float32),
                    max_vel=self.max_vel,
                    max_acc=self.max_acc,
                    max_angular_vel=self.max_angular_vel,
                    max_angular_acc=self.max_angular_acc,
                )
            except Exception as e:
                print(
                    f"[SingleArmAgent] skip init tcp pose due to control-mode issue: {e}"
                )

        # align with teleop-like gripper start state (closed ~= width 0.0)
        self._raw_gripper = _build_raw_gripper(self.robot, self.gripper)
        _ensure_gripper_ready(
            self.gripper, self._raw_gripper, label="SingleArm", do_init=False
        )
        self.last_gripper = _init_gripper_closed(
            self.gripper, self._raw_gripper, label="SingleArm-init"
        )

        if gripper_self_test:
            try:
                self.last_gripper = _gripper_self_test(
                    self.gripper, self._raw_gripper, label="SingleArm-self-test"
                )
            except Exception as e:
                print(f"[SingleArmAgent] gripper self-test skipped: {e}")

        self.robot.cali_sensor()

        # gripper move may change control mode; recover before wrench setup
        _recover_motion_mode(self.robot)

        # set max contact wrench
        self.robot.robot.SetMaxContactWrench(max_contact_wrench)

    @property
    def ready_pose(self):
        return np.array([0.5, 0, 0.17, 0, 0, 1, 0], dtype=np.float32)

    # 获取相机观察到的
    def get_global_observation(self):
        _, colors, depths = self.camera.get_rgbd_images()
        return colors, depths

    # 获取机器人当前状态（本体感知），tcp 和夹爪宽度
    # 机械臂相关东西在 arm.py
    def get_proprio(
        self, rotation_rep="rotation_6d", rotation_rep_convention=None, with_joint=False
    ):
        tcp_pose = self.robot.get_tcp_pose()
        tcp_pose = xyz_rot_transform(
            tcp_pose,
            from_rep="quaternion",
            to_rep=rotation_rep,
            to_convention=rotation_rep_convention,
        )
        if with_joint:
            joint_pos = self.robot.get_joint_pos()
        gripper_width = self.gripper.get_states()["width"]

        proprio = np.concatenate([tcp_pose, [gripper_width]], axis=0)
        if with_joint:
            proprio_joint = np.concatenate([joint_pos, [gripper_width]], axis=0)
            return proprio, proprio_joint
        else:
            return proprio

    def action(self, action, rotation_rep="rotation_6d", rotation_rep_convention=None):
        tcp_pose = xyz_rot_transform(
            action[:9],
            from_rep=rotation_rep,
            to_rep="quaternion",
            from_convention=rotation_rep_convention,
        )

        self.robot.send_tcp_pose(
            tcp_pose,
            max_vel=self.max_vel,
            max_acc=self.max_acc,
            max_angular_vel=self.max_angular_vel,
            max_angular_acc=self.max_angular_acc,
        )
        time.sleep(0.1)

        gripper_action = False
        target_width = _to_gripper_width(float(action[9]), mode="width")
        if abs(target_width - self.last_gripper) >= 0.003:
            sent = _move_gripper(
                self.gripper,
                getattr(self, "_raw_gripper", None),
                target_width,
                speed=0.1,
                force=40.0,
                label="SingleArm-action",
            )
            if sent:
                self.last_gripper = _read_gripper_width(
                    self.gripper, default=target_width
                )
                gripper_action = True

        if gripper_action:
            # Gripper command may switch to primitive mode; recover for next TCP command.
            _recover_motion_mode(self.robot)
            time.sleep(0.2)

    def stop(self):
        self.robot.stop()
        self.gripper.stop()
        self.camera.stop()


class DualArmAgent:
    """
    Evaluation dual-arm agent with Flexiv arms, Robotiq grippers and an Intel RealSense RGB-D camera.

    Follow the implementation here to create your own real-world evaluation agent.
    """

    def __init__(
        self,
        left_robot_serial,
        right_robot_serial,
        left_gripper_port,
        right_gripper_port,
        camera_serial,
        left_max_contact_wrench=[30, 30, 30, 10, 10, 10],
        right_max_contact_wrench=[30, 30, 30, 10, 10, 10],
        max_vel=0.5,
        max_acc=2.0,
        max_angular_vel=1.0,
        max_angular_acc=5.0,
        gripper_key="width",
        left_init_joint_deg=None,
        right_init_joint_deg=None,
        left_init_tcp_pose=None,
        right_init_tcp_pose=None,
        enable_init_tcp_pose=False,
        gripper_self_test=True,
        **kwargs,
    ):
        # initialize
        self.left_robot = FlexivArm(left_robot_serial)
        self.right_robot = FlexivArm(right_robot_serial)
        self.left_gripper = FlexivGripper(self.left_robot)
        self.right_gripper = FlexivGripper(self.right_robot)
        self.left_raw_gripper = self.left_gripper.gripper
        self.right_raw_gripper = self.right_gripper.gripper

        # from flexiv_robot_interface import flexiv_rizon_interface as single

        self.left_gripper.close_gripper()
        self.right_gripper.close_gripper()

        # keep consistent with collector config: (width, velocity, force)
        self.gripper_switch_command = {
            "start_state": (0.0, 0.2, 50),
            "open": (0.1, 0.2, -65),
            "close": (0.0, 0.2, 45),
        }

        self.target_max_width = self.gripper_switch_command["open"][0]
        self.target_velocity = self.gripper_switch_command["open"][1]
        self.target_force = abs(self.gripper_switch_command["close"][2])

        self.camera_serial = camera_serial
        self.camera = RealSenseRGBDCamera(serial=camera_serial)
        self.intrinsics = self.camera.get_intrinsic()
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.max_angular_vel = max_angular_vel
        self.max_angular_acc = max_angular_acc

        assert gripper_key in ["width", "action"]
        self.gripper_key = gripper_key

        self.last_left_gripper = 0
        self.last_right_gripper = 0
        self._gripper_step = 0

        # move to initial pose (prefer teleop initial poses by serial, fallback to local ready poses)
        left_cfg = TELEOP_INIT_POSES.get(left_robot_serial, {})
        right_cfg = TELEOP_INIT_POSES.get(right_robot_serial, {})

        if left_init_joint_deg is None:
            left_init_joint_deg = left_cfg.get("joint_pose_deg")
        if right_init_joint_deg is None:
            right_init_joint_deg = right_cfg.get("joint_pose_deg")
        if left_init_tcp_pose is None:
            left_init_tcp_pose = left_cfg.get("cartesian_pose")
        if right_init_tcp_pose is None:
            right_init_tcp_pose = right_cfg.get("cartesian_pose")

        if left_init_joint_deg is not None:
            left_joint_target = np.array(left_init_joint_deg, dtype=np.float32)
            left_joint_pose_unit = "deg"
        else:
            left_joint_target = self.left_ready_pose
            left_joint_pose_unit = "rad"

        if right_init_joint_deg is not None:
            right_joint_target = np.array(right_init_joint_deg, dtype=np.float32)
            right_joint_pose_unit = "deg"
        else:
            right_joint_target = self.right_ready_pose
            right_joint_pose_unit = "rad"

        def _move_joint_init(robot, joint_target, pose_unit: str, name: str):
            target_arr = np.array(joint_target, dtype=np.float32)
            qdeg = (
                target_arr.tolist()
                if pose_unit == "deg"
                else np.rad2deg(target_arr).tolist()
            )
            moved = False

            # Prefer teleop-like raw robot API when available (robot.robot + single.move_j_deg)
            try:
                from flexiv_robot_interface import flexiv_rizon_interface as single

                raw_robot = getattr(robot, "robot", None)
                if raw_robot is not None:
                    try:
                        single.move_j_deg(
                            raw_robot,
                            qdeg,
                            time_out=10,
                            joint_allow_error_angle=3,
                            stiffness_ratio=0.1,
                            compliant=False,
                            is_blocking=True,
                        )
                    except TypeError:
                        single.move_j_deg(
                            raw_robot,
                            qdeg,
                            max_vel=1,
                            max_acc=1,
                            is_lock_ext_axis=False,
                            compliant=False,
                            is_blocking=True,
                        )
                    moved = True
            except Exception as e:
                print(
                    f"[DualArmAgent] {name} raw move_j_deg failed, fallback to wrapper: {e}"
                )

            if not moved:
                robot.send_joint_pos(
                    target_arr,
                    max_vel=[0.3] * 7,
                    max_acc=[0.5] * 7,
                    impedance=False,
                    blocking=True,
                    pose_unit=pose_unit,
                )

        _move_joint_init(
            self.left_robot, left_joint_target, left_joint_pose_unit, "left"
        )
        _move_joint_init(
            self.right_robot, right_joint_target, right_joint_pose_unit, "right"
        )

        # Follow teleop startup behavior: initialize by joint poses first.
        # TCP init pose is optional because some deployments cannot switch to
        # Cartesian mode at this stage.
        if enable_init_tcp_pose:
            try:
                self.left_robot.send_tcp_pose(
                    np.array(left_init_tcp_pose, dtype=np.float32),
                    max_vel=self.max_vel,
                    max_acc=self.max_acc,
                    max_angular_vel=self.max_angular_vel,
                    max_angular_acc=self.max_angular_acc,
                ) if left_init_tcp_pose is not None else None
                self.right_robot.send_tcp_pose(
                    np.array(right_init_tcp_pose, dtype=np.float32),
                    max_vel=self.max_vel,
                    max_acc=self.max_acc,
                    max_angular_vel=self.max_angular_vel,
                    max_angular_acc=self.max_angular_acc,
                ) if right_init_tcp_pose is not None else None
            except Exception as e:
                print(
                    f"[DualArmAgent] Skip init tcp pose due to control-mode issue: {e}"
                )

        # Collector-like startup command (start_state)
        start_cmd = self.gripper_switch_command["start_state"]
        self.left_raw_gripper.Move(start_cmd[0], start_cmd[1], start_cmd[2])
        self.right_raw_gripper.Move(start_cmd[0], start_cmd[1], start_cmd[2])
        self.last_left_gripper = start_cmd[0]
        self.last_right_gripper = start_cmd[0]
        time.sleep(0.7)

        if gripper_self_test:
            try:
                open_cmd = self.gripper_switch_command["open"]
                close_cmd = self.gripper_switch_command["close"]
                self.left_raw_gripper.Move(open_cmd[0], open_cmd[1], open_cmd[2])
                time.sleep(0.4)
                self.left_raw_gripper.Move(close_cmd[0], close_cmd[1], close_cmd[2])
            except Exception as e:
                print(f"[DualArm-left-self-test] failed: {e}")

            try:
                open_cmd = self.gripper_switch_command["open"]
                close_cmd = self.gripper_switch_command["close"]
                self.right_raw_gripper.Move(open_cmd[0], open_cmd[1], open_cmd[2])
                time.sleep(0.4)
                self.right_raw_gripper.Move(close_cmd[0], close_cmd[1], close_cmd[2])
            except Exception as e:
                print(f"[DualArm-right-self-test] failed: {e}")

        self.left_robot.cali_sensor()
        self.right_robot.cali_sensor()

        # Gripper operations may leave robot in NRT_PRIMITIVE_EXECUTION.
        # Recover to motion-capable mode before setting wrench and before policy control.
        if not _recover_motion_mode(self.left_robot):
            print("[DualArmAgent] failed to recover motion mode for left")
        if not _recover_motion_mode(self.right_robot):
            print("[DualArmAgent] failed to recover motion mode for right")

        # set max contact wrench
        try:
            self.left_robot.robot.SetMaxContactWrench(left_max_contact_wrench)
        except Exception as e:
            print(f"[DualArmAgent] left SetMaxContactWrench failed: {e}")
        try:
            self.right_robot.robot.SetMaxContactWrench(right_max_contact_wrench)
        except Exception as e:
            print(f"[DualArmAgent] right SetMaxContactWrench failed: {e}")

        # Ensure subsequent send_tcp_pose() starts in Cartesian motion-force mode.
        _recover_motion_mode(self.left_robot)
        _recover_motion_mode(self.right_robot)

    @property
    def left_ready_pose(self):
        return np.array(
            [
                1.5638101e00,
                -2.1990967e00,
                1.1152865e00,
                -1.8542223e00,
                -5.2635312e-01,
                2.3846159e-02,
                1.5591051e-01,
            ],
            dtype=np.float32,
        )

    @property
    def right_ready_pose(self):
        # return np.array([
        #     1.59702516e+00,
        #     -2.24452472e+00,
        #     1.23052871e+00,
        #     -1.83945274e+00,
        #     -4.81370360e-01,
        #     1.86598241e-01,
        #     2.09201425e-01
        # ], dtype=np.float32)
        return np.array(
            [
                2.164233,
                -1.7392905,
                1.7191751,
                -1.8739327,
                -0.5582644,
                0.8032095,
                0.44572696,
            ],
            dtype=np.float32,
        )

    def get_global_observation(self):
        colors, depths = self.camera.get_rgbd_images()
        return colors, depths

    def get_proprio(
        self, rotation_rep="rotation_6d", rotation_rep_convention=None, with_joint=False
    ):
        left_tcp_pose = self.left_robot.get_tcp_pose()
        right_tcp_pose = self.right_robot.get_tcp_pose()
        left_tcp_pose = xyz_rot_transform(
            left_tcp_pose,
            from_rep="quaternion",
            to_rep=rotation_rep,
            to_convention=rotation_rep_convention,
        )
        right_tcp_pose = xyz_rot_transform(
            right_tcp_pose,
            from_rep="quaternion",
            to_rep=rotation_rep,
            to_convention=rotation_rep_convention,
        )
        if with_joint:
            left_joint_pos = self.left_robot.get_joint_pos()
            right_joint_pos = self.right_robot.get_joint_pos()
        left_gripper_width = self.left_gripper.get_states()[self.gripper_key]
        right_gripper_width = self.right_gripper.get_states()[self.gripper_key]

        proprio = np.concatenate(
            [
                left_tcp_pose,
                [left_gripper_width],
                right_tcp_pose,
                [right_gripper_width],
            ],
            axis=0,
        )
        if with_joint:
            proprio_joint = np.concatenate(
                [
                    left_joint_pos,
                    [left_gripper_width],
                    right_joint_pos,
                    [right_gripper_width],
                ],
                axis=0,
            )
            return proprio, proprio_joint
        else:
            return proprio

    def action(self, action, rotation_rep="rotation_6d", rotation_rep_convention=None):
        left_tcp_pose = xyz_rot_transform(
            action[:9],
            from_rep=rotation_rep,
            to_rep="quaternion",
            from_convention=rotation_rep_convention,
        )
        right_tcp_pose = xyz_rot_transform(
            action[10:19],
            from_rep=rotation_rep,
            to_rep="quaternion",
            from_convention=rotation_rep_convention,
        )

        self.left_robot.send_tcp_pose(
            left_tcp_pose,
            max_vel=self.max_vel,
            max_acc=self.max_acc,
            max_angular_vel=self.max_angular_vel,
            max_angular_acc=self.max_angular_acc,
        )
        self.right_robot.send_tcp_pose(
            right_tcp_pose,
            max_vel=self.max_vel,
            max_acc=self.max_acc,
            max_angular_vel=self.max_angular_vel,
            max_angular_acc=self.max_angular_acc,
        )
        time.sleep(0.1)

        self._gripper_step += 1
        gripper_action = False
        left_target_width = _to_gripper_width(float(action[9]), mode=self.gripper_key)
        right_target_width = _to_gripper_width(float(action[19]), mode=self.gripper_key)

        # Collector-like gripper command: explicit (width, velocity, force)
        open_cmd = self.gripper_switch_command["open"]
        close_cmd = self.gripper_switch_command["close"]

        try:
            left_cmd = close_cmd if left_target_width < 0.03 else open_cmd
            self.left_raw_gripper.Move(left_cmd[0], left_cmd[1], left_cmd[2])
            self.last_left_gripper = left_cmd[0]
            gripper_action = True
        except Exception as e:
            print(f"[DualArm-left-action] raw Move failed: {e}")

        try:
            right_cmd = close_cmd if right_target_width < 0.03 else open_cmd
            self.right_raw_gripper.Move(right_cmd[0], right_cmd[1], right_cmd[2])
            self.last_right_gripper = right_cmd[0]
            gripper_action = True
        except Exception as e:
            print(f"[DualArm-right-action] raw Move failed: {e}")

        if self._gripper_step % 50 == 0:
            print(
                f"[DualArm-gripper-step={self._gripper_step}] "
                f"cmd=({left_target_width:.4f},{right_target_width:.4f})"
            )

        if gripper_action:
            # Keep TCP control alive after primitive gripper command.
            _recover_motion_mode(self.left_robot)
            _recover_motion_mode(self.right_robot)
            time.sleep(0.2)

    def stop(self):
        self.left_robot.stop()
        self.right_robot.stop()
        self.left_gripper.stop()
        self.right_gripper.stop()
        self.camera.stop()
