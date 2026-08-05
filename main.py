import src.mujoco_viewer as mujoco_viewer
import numpy as np
from src.IKFK import (
    TOOL_1_DEFAULT_ROT_WORLD,
    is_tool_pose_reachable,
    leg_length_to_slide_ctrl,
    stewart_ik_tool,
)
from src.utils import rot2eul, rotvec_to_rot
from src.admi_integral import IntegralAdmittance
from src.mujoco_ft_sensor import MujocoFTSensor
from src.mujoco_stewart_platform import MujocoStewartPlatform
from src.stewartt_State import ControlState, StewartStateMachine
np.set_printoptions(suppress=True, precision=9)
from src.data_logger import DataLogger

logger = DataLogger("sensor_logs/data3.xlsx")


def apply_deadband(value, deadband):
    value = np.asarray(value, dtype=float)
    deadband = np.asarray(deadband, dtype=float)
    return np.sign(value) * np.maximum(np.abs(value) - deadband, 0.0)


class Stewart(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 1.8, azimuth=45, elevation=-30)
        self.path = path
        self.model.opt.timestep = 0.001
        self.ft_sensor = MujocoFTSensor(
            self.model,
            self.data,
            filter_cutoff_hz=30.0,
        )
        self.platform = MujocoStewartPlatform(self.model, self.data)
        self.force_target_z = -10.0
        self.reaching_force_ramp_time = 1.0
        self.contact_force_ramp_time = 0.1
        self.desired_wrench = np.zeros(6, dtype=float)

        self.state_machine = StewartStateMachine(
            calibration_duration=0.1,
            contact_threshold=1.0,
            target_force_z=self.force_target_z,
            # 力的容差
            wrench_tracking_tolerance=[0.3,0.3,0.3,0.02,0.02,0.02,],
            wrench_tracking_duration=1.0)
        # 标志位
        self.contact_wrench_printed = False

        self.M=[3,3,3,1,1,1]
        self.D=[200,200,500,10,10,10]
        self.k=[0,0,0,0,0,0]
        #积分导纳
        self.admit=IntegralAdmittance(
            self.M,
            self.D,
            self.k,
            self.model.opt.timestep,
            Ki=[0.0, 0.0, 0.0, 0.6, 0.6, 0.6],
            integral_limit=[0.0, 0.0, 0.0, 0.2, 0.2, 0.2],
            integral_leak=0.9995,
        )
        
        self.force_deadband = np.array([0, 0, 0])
        self.torque_deadband = np.array([0, 0, 0])
        self.torque_gain = np.array([6.0, 6.0, 6.0])

        self.print_count = 0
        
        
    # def adaptive_damping(self, force_magnitude):
    #     """根据力的大小自适应调整阻尼系数。"""
    #     # 阻尼系数随力的大小线性增加，力越大，阻尼越大。
    #     # 这样可以在接触时快速衰减振动，同时在空闲时保持灵敏。
    #     min_damping = np.array([100, 100, 400, 1, 1, 1])
    #     max_damping = np.array([600, 600, 600, 10, 10, 10])
    #     force_threshold = 5.0  # N
    #     damping = min_damping + (max_damping - min_damping) * np.clip(force_magnitude / force_threshold, 0.0, 1.0)
    #     return damping

        
    def runBefore(self):
        #姿态是弧度
        init_platform_pose = np.array([0.0, 0.0, 0.67, 0.0, 0.0, 0.0], dtype=float)
        self.init_slide_ctrl = self.platform.set_initial_pose(
            init_platform_pose[:3],
            init_platform_pose[3:6],
        )

        self.tool_pose0 = self.ft_sensor.tool_pose_command_frame(
            TOOL_1_DEFAULT_ROT_WORLD
        )
        
        self.R_WT0 = self.ft_sensor.tool_rotation()
                
        self.admit.set_state(
            np.zeros(6, dtype=float),
            np.zeros(6, dtype=float),
        )
        self.admit_pose_tool_prev = np.zeros(6, dtype=float)

        # 世界系目标位姿逐步累加，每一步都使用当时的实时工具方向。
        self.target_tool_pos_world = self.tool_pose0[:3].copy()
        self.target_tool_rot_world = self.R_WT0.copy()
    
        # 初始设置位姿后约束力还没有稳定，不能在这里立即标定零偏。
        # 零偏会在下面的 0.1 s 保持阶段持续更新。
        self.ft_sensor.reset_bias()
        self.state_machine.reset(self.data.time)
        self.contact_wrench_printed = False
        self.desired_wrench[:] = 0.0
        self.measured_wrench = np.zeros(6, dtype=float)
        self.wrench_for_admittance = np.zeros(6, dtype=float)
        self.admit.set_force(self.desired_wrench)
    
           

    def _run_calibrating_state(self):
        """保持初始姿态，并持续更新传感器零偏。"""
        self.platform.set_leg_position_targets(self.init_slide_ctrl)
        self.ft_sensor.capture_bias()

    def _run_admittance_control(self):
        """执行一个周期的导纳、可达性检查和IK控制。"""

        ft_tool = self.ft_sensor.read_filtered_wrench()
        self.print_count += 1
        if self.print_count % 100 == 0:
            print("ft_tool_filtered:", ft_tool)

        
        # # delta_tool 是工具坐标系下的位移/转角增量
        # 状态机使用增益和死区处理前的物理六维力进行完成判断。
        self.measured_wrench = -ft_tool.copy()
        self.wrench_for_admittance = self.measured_wrench.copy()

        self.wrench_for_admittance[3:] = apply_deadband(
            self.wrench_for_admittance[3:],
            self.torque_deadband,
        ) * self.torque_gain
        
        admit_pose_tool = self.admit.update(self.wrench_for_admittance)

        # 只取本仿真步新产生的局部位姿增量，避免工具旋转时把历史轨迹一起旋转。
        pose_step_tool = admit_pose_tool - self.admit_pose_tool_prev
        self.admit_pose_tool_prev = admit_pose_tool.copy()

        # 实时工具姿态：本步位移严格沿当前工具坐标轴。
        R_WT_current = self.ft_sensor.tool_rotation()
        target_tool_pos = (self.target_tool_pos_world+ R_WT_current @ pose_step_tool[:3])
        # 本步工具系旋转增量也绕当前工具坐标轴叠加。
        R_step_world = (R_WT_current@ rotvec_to_rot(pose_step_tool[3:6])@ R_WT_current.T)
        target_tool_rot_world = R_step_world @ self.target_tool_rot_world
    
        # IK 接收相对于工具默认安装姿态的姿态命令。
        R_target_delta = TOOL_1_DEFAULT_ROT_WORLD.T @ target_tool_rot_world
        target_tool_rpy = rot2eul(R_target_delta)

        #判断是否可达
        ok, info = is_tool_pose_reachable(target_tool_pos,target_tool_rpy,return_info=True,)

        # ok和力数据都准备好后，再让状态机判断是否需要跳转。
        self.state_machine.update(
            self.data.time,
            wrench=self.measured_wrench,
            reachable=ok,
            desired_wrench=self.desired_wrench,
        )
        if not ok:
            self._report_unreachable(info)
            return
        
        leg, _ = stewart_ik_tool(target_tool_pos, target_tool_rpy)
        self.platform.set_leg_position_targets(leg_length_to_slide_ctrl(leg))
        self.target_tool_pos_world = target_tool_pos
        self.target_tool_rot_world = target_tool_rot_world

        # logger.record(time=self.data.time,ft_tool=ft_tool)

    def _ramp_desired_force_z(self, ramp_time):
        """按给定时间把工具Z轴期望力线性推进到目标值。"""
        ramp_time = float(ramp_time)
        if ramp_time <= 0.0:
            self.desired_wrench[2] = self.force_target_z
        else:
            max_step = (
                abs(self.force_target_z)
                * self.model.opt.timestep
                / ramp_time
            )
            force_error = self.force_target_z - self.desired_wrench[2]
            self.desired_wrench[2] += np.clip(force_error,-max_step,max_step)
        self.admit.set_force(self.desired_wrench)

    def _run_reaching_state(self):
        """接近阶段：慢速增加期望力，同时持续执行导纳控制。"""
        self._ramp_desired_force_z(self.reaching_force_ramp_time)
        self._run_admittance_control()

    def _run_contact_state(self):
        """接触阶段：快速把期望力提升到目标值。"""
        if not self.contact_wrench_printed:
            print("接触时六维力:", self.measured_wrench)
            self.contact_wrench_printed = True

        self._ramp_desired_force_z(self.contact_force_ramp_time)
        self._run_admittance_control()

    def _run_force_control_state(self):
        """恒力阶段：保持目标期望力并持续执行导纳控制。"""
        self.desired_wrench[2] = self.force_target_z
        self.admit.set_force(self.desired_wrench)
        self._run_admittance_control()

    def _run_finished_state(self):
        """对接完成：保持最后一次执行器控制目标。"""
        pass

    def _run_fault_state(self):
        """故障状态：默认保留最后一次位置目标。"""
        # TODO: 真机应在这里调用驱动器的安全停止或卸载接口。
        pass

    @staticmethod
    def _report_unreachable(info):
        print("target unreachable")
        print(
            f"  stroke_ok={info['stroke_ok']}, "
            f"lower_ball_ok={info['lower_ball_ok']}, "
            f"upper_ball_ok={info['upper_ball_ok']}"
        )
        print(f"  bad stroke legs: {info['bad_legs']}")
        print(f"  bad lower ball legs: {info['bad_lower_ball_legs']}")
        print(f"  bad upper ball legs: {info['bad_upper_ball_legs']}")
        print("  leg | slide_ctrl(m) | lower_ball(deg) | upper_ball(deg)")
        for index in range(6):
            print(
                f"  {index + 1:>3} | "
                f"{info['slide_ctrl'][index]:>13.3f} | "
                f"{info['lower_ball_angles_deg'][index]:>15.3f} | "
                f"{info['upper_ball_angles_deg'][index]:>15.3f}"
            )


    def runFunc(self):
        #状态机调度
        state = self.state_machine.update(self.data.time)

        if state == ControlState.CALIBRATING:
            self._run_calibrating_state()
        elif state == ControlState.FORCE_CONTROL:
            self._run_force_control_state()
        elif state == ControlState.CONTACT:
            self._run_contact_state()
        elif state == ControlState.REACHING:
            self._run_reaching_state()
        elif state == ControlState.FINISHED:
            self._run_finished_state()
        elif state == ControlState.FAULT:
            self._run_fault_state()
        else:
            raise RuntimeError(f"未处理的控制状态: {state}")

if __name__ == "__main__":
    env = Stewart("scene.xml")
    try:
        env.run_loop()
    finally:
        logger.save()
        # print(f"data saved to: {logger.path.resolve()}")
