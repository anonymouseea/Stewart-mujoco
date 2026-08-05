import mujoco
import numpy as np

from src.utils import rot2eul


class MujocoFTSensor:
    """读取 MuJoCo 六维力传感器，并将其转换到工具坐标系。"""

    def __init__(
        self,
        model,
        data,
        *,
        tool_site="tool",
        sensor_site="ft_sensor_site",
        force_sensor="ft_force",
        torque_sensor="ft_torque",
        payload_body="tool_payload",
        filter_cutoff_hz=30.0,
    ):
        self.model = model
        self.data = data
        self.filter_cutoff_hz = float(filter_cutoff_hz)

        self.tool_site_id = self._require_id(
            mujoco.mjtObj.mjOBJ_SITE,
            tool_site,
            "site",
        )
        self.sensor_site_id = self._require_id(
            mujoco.mjtObj.mjOBJ_SITE,
            sensor_site,
            "site",
        )
        self.force_sensor_id = self._require_id(
            mujoco.mjtObj.mjOBJ_SENSOR,
            force_sensor,
            "sensor",
        )
        self.torque_sensor_id = self._require_id(
            mujoco.mjtObj.mjOBJ_SENSOR,
            torque_sensor,
            "sensor",
        )
        self.payload_body_id = self._require_id(
            mujoco.mjtObj.mjOBJ_BODY,
            payload_body,
            "body",
        )

        self.bias = np.zeros(6, dtype=float)
        self._filtered = np.zeros(6, dtype=float)
        self._filter_initialized = False

    def _require_id(self, object_type, name, label):
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id == -1:
            raise ValueError(f"MuJoCo model中没有找到{label}: {name}")
        return object_id

    def _read_sensor(self, sensor_id):
        address = self.model.sensor_adr[sensor_id]
        dimension = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[address:address + dimension].copy()

    def _site_position(self, site_id):
        return self.data.site_xpos[site_id].copy()

    def _site_rotation(self, site_id):
        return self.data.site_xmat[site_id].reshape(3, 3).copy()

    def tool_position(self):
        """返回工具原点在世界坐标系中的位置。"""
        return self._site_position(self.tool_site_id)

    def tool_rotation(self):
        """返回工具坐标系到世界坐标系的实时旋转矩阵 R_WT。"""
        return self._site_rotation(self.tool_site_id)

    def tool_pose_command_frame(self, default_rotation_world):
        """返回世界系工具位置和相对默认安装姿态的 RPY 命令。"""
        default_rotation_world = np.asarray(
            default_rotation_world,
            dtype=float,
        ).reshape(3, 3)
        rotation_delta = default_rotation_world.T @ self.tool_rotation()
        return np.concatenate([
            self.tool_position(),
            rot2eul(rotation_delta),
        ])

    def read_tool_wrench(self):
        """返回经过重力补偿、且参考点位于工具原点的工具系六维力。"""
        force_sensor = self._read_sensor(self.force_sensor_id)
        torque_sensor = self._read_sensor(self.torque_sensor_id)

        R_WT = self.tool_rotation()
        R_WS = self._site_rotation(self.sensor_site_id)
        p_WT = self.tool_position()
        p_WS = self._site_position(self.sensor_site_id)

        mass = self.model.body_subtreemass[self.payload_body_id]
        p_WC = self.data.subtree_com[self.payload_body_id]

        # 静止时传感器承受的支撑力与模型重力方向相反。
        force_gravity_world = -mass * self.model.opt.gravity
        force_gravity_sensor = R_WS.T @ force_gravity_world

        r_sensor_to_com_sensor = R_WS.T @ (p_WC - p_WS)
        torque_gravity_sensor = np.cross(
            r_sensor_to_com_sensor,
            force_gravity_sensor,
        )

        force_sensor -= force_gravity_sensor
        torque_sensor -= torque_gravity_sensor

        # 将传感器系六维力转换到工具系，并把力矩参考点移到工具原点。
        R_TS = R_WT.T @ R_WS
        force_tool = R_TS @ force_sensor
        torque_tool = R_TS @ torque_sensor
        r_tool_to_sensor_tool = R_WT.T @ (p_WS - p_WT)
        torque_tool += np.cross(r_tool_to_sensor_tool, force_tool)

        return np.concatenate([force_tool, torque_tool])

    def capture_bias(self):
        """把当前重力补偿后的六维力记录为零偏。"""
        self.bias = self.read_tool_wrench()
        self.reset_filter()

    def reset_bias(self):
        self.bias[:] = 0.0
        self.reset_filter()

    def reset_filter(self):
        self._filtered[:] = 0.0
        self._filter_initialized = False

    def read_filtered_wrench(self):
        """返回扣除零偏并经过一阶低通滤波的工具系六维力。"""
        wrench = self.read_tool_wrench() - self.bias

        if not self._filter_initialized:
            self._filtered[:] = wrench
            self._filter_initialized = True
            return self._filtered.copy()

        dt = self.model.opt.timestep
        tau = 1.0 / (2.0 * np.pi * self.filter_cutoff_hz)
        alpha = dt / (tau + dt)
        self._filtered += alpha * (wrench - self._filtered)
        return self._filtered.copy()
