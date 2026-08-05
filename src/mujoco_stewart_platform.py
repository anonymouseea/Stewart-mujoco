import mujoco
import numpy as np

from src.IKFK import (
    BASE_POINTS,
    SLIDE_MAX,
    SLIDE_MIN,
    leg_length_to_slide_ctrl,
    stewart_ik,
)
from src.utils import euler2quat, quat_from_z_axis


class MujocoStewartPlatform:
    """封装 Stewart 平台在 MuJoCo 中的关节状态与控制操作。"""

    def __init__(self, model, data):
        self.model = model
        self.data = data

        platform_joint_id = self._require_joint("platform_free")
        self.platform_qpos_address = model.jnt_qposadr[platform_joint_id]
        self.platform_dof_address = model.jnt_dofadr[platform_joint_id]

        self.slide_qpos_addresses = np.empty(6, dtype=int)
        self.slide_dof_addresses = np.empty(6, dtype=int)
        self.ball_qpos_addresses = np.empty(6, dtype=int)
        self.ball_dof_addresses = np.empty(6, dtype=int)

        for index in range(6):
            leg_number = index + 1
            slide_id = self._require_joint(f"leg{leg_number}_slide")
            ball_id = self._require_joint(f"leg{leg_number}_base_ball")

            self.slide_qpos_addresses[index] = model.jnt_qposadr[slide_id]
            self.slide_dof_addresses[index] = model.jnt_dofadr[slide_id]
            self.ball_qpos_addresses[index] = model.jnt_qposadr[ball_id]
            self.ball_dof_addresses[index] = model.jnt_dofadr[ball_id]

    def _require_joint(self, name):
        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        if joint_id == -1:
            raise ValueError(f"MuJoCo model中没有找到joint: {name}")
        return joint_id

    def leg_slide_positions(self):
        """返回六条腿滑动关节的当前位置。"""
        return self.data.qpos[self.slide_qpos_addresses].copy()

    def set_leg_position_targets(self, slide_targets):
        """设置六条腿的位置目标，并将对应速度执行器目标置零。"""
        slide_targets = np.asarray(slide_targets, dtype=float).reshape(6)
        self.data.ctrl[:6] = np.clip(slide_targets, SLIDE_MIN, SLIDE_MAX)
        if self.model.nu >= 12:
            self.data.ctrl[6:12] = 0.0

    def set_initial_pose(self, position, rpy):
        """设置平台、球铰及电缸的相互一致的初始状态。"""
        position = np.asarray(position, dtype=float).reshape(3)
        rpy = np.asarray(rpy, dtype=float).reshape(3)

        qpos_address = self.platform_qpos_address
        dof_address = self.platform_dof_address
        self.data.qpos[qpos_address:qpos_address + 3] = position
        self.data.qpos[qpos_address + 3:qpos_address + 7] = euler2quat(*rpy)
        self.data.qvel[dof_address:dof_address + 6] = 0.0

        lengths, top_points = stewart_ik(position, rpy)
        slide_targets = leg_length_to_slide_ctrl(lengths)
        self.set_leg_position_targets(slide_targets)

        for index in range(6):
            leg_direction = (
                top_points[:, index] - BASE_POINTS[:, index]
            ) / lengths[index]

            ball_qpos = self.ball_qpos_addresses[index]
            ball_dof = self.ball_dof_addresses[index]
            self.data.qpos[ball_qpos:ball_qpos + 4] = quat_from_z_axis(
                leg_direction
            )
            self.data.qvel[ball_dof:ball_dof + 3] = 0.0

        self.data.qpos[self.slide_qpos_addresses] = slide_targets
        self.data.qvel[self.slide_dof_addresses] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return slide_targets.copy()
