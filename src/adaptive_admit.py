import numpy as np


def sat(x):
    return np.clip(x, -1.0, 1.0)


def clip_vec(x, limit):
    return np.clip(x, -limit, limit)


class StewartAdaptiveAdmittanceVelocity:
    """
    只有 Stewart 平台的自适应导纳控制器，速度接口版本。

    输入：
        X_meas:
            当前末端位姿 [x, y, z, roll, pitch, yaw]
            一般是工具坐标系相对于基座/世界坐标系的位姿。

        X_ref:
            粗定位目标位姿 [x, y, z, roll, pitch, yaw]
            轴位置未知时，x_ref, y_ref 只是粗定位点，不用于强位置跟踪。

        F_ext:
            末端六维力/力矩 [Fx, Fy, Fz, Mx, My, Mz]
            推荐使用工具坐标系下的力。

    输出：
        V_cmd:
            末端速度指令 [vx, vy, vz, wx, wy, wz]
            默认理解为工具坐标系下的速度。
    """

    def __init__(self, M, D, K, dt):
        self.M = np.asarray(M, dtype=float).reshape(6)
        self.D = np.asarray(D, dtype=float).reshape(6)
        self.K = np.asarray(K, dtype=float).reshape(6)
        self.dt = float(dt)

        if np.any(self.M == 0.0):
            raise ValueError("导纳控制的虚拟质量 M 不能为 0。")

        # =====================================================
        # 只有 Stewart 平台时的默认控制选择
        # =====================================================
        # Stewart 允许输出 x, y, z, roll, pitch，yaw 先关闭
        self.active_mask = np.array([1, 1, 1, 1, 1, 0], dtype=float)

        # 轴位置未知：x/y 不做位置误差跟踪，靠 Fx/Fy 找正
        # 姿态 roll/pitch 仍然可以跟踪目标姿态
        self.pose_mask = np.array([0, 0, 0, 1, 1, 0], dtype=float)

        # Fx, Fy, Fz, Mx, My 参与力控；Mz 暂时不用
        self.force_mask = np.array([1, 1, 1, 1, 1, 0], dtype=float)

        # 方向符号。某个方向越调越偏，就改对应符号为 -1
        self.axis_sign = np.array([1, 1, 1, 1, 1, 1], dtype=float)

        # 期望接触力
        # 只有 Stewart 时，z 方向也由 Stewart 控制，所以 Fz 要给一个推进力
        self.F_desire = np.array([0, 0, 20, 0, 0, 0], dtype=float)

        # =====================================================
        # 基础导纳状态
        # =====================================================
        # delta_x 不是输出位姿，而是导纳内部的虚拟位移状态
        # 速度接口最终输出的是 delta_v
        self.delta_x = np.zeros(6, dtype=float)
        self.delta_v = np.zeros(6, dtype=float)

        # =====================================================
        # 自适应状态
        # =====================================================
        self.eta = np.zeros(6, dtype=float)
        self.F_err_last = np.zeros(6, dtype=float)
        self.F_err_dot_filt = np.zeros(6, dtype=float)
        self.initialized = False

        self.p_hat = np.zeros(6, dtype=float)
        self.d_hat = np.zeros(6, dtype=float)
        self.r_hat = np.zeros(6, dtype=float)
        self.h_hat = np.zeros(6, dtype=float)

        # =====================================================
        # 基础导纳增益
        # =====================================================
        # 力误差项
        # 对应公式中的 -Kf * e_f
        self.Kf = np.array([1.0, 1.0, 1.0, 0.8, 0.8, 0.0], dtype=float)

        # 位姿误差项
        # x/y/z 不用位置误差跟踪，所以前 3 项可以为 0
        self.Kp = np.array([0.0, 0.0, 0.0, 4.0, 4.0, 0.0], dtype=float)

        # 积分项，用于消除稳态误差
        self.Ki = np.array([0.15, 0.15, 0.08, 0.08, 0.08, 0.0], dtype=float)

        # =====================================================
        # 复合误差参数
        # s = F_err_dot + Lambda_f*F_err + Lambda_p*pose_err + Lambda_i*eta
        # =====================================================
        self.Lambda_f = np.array([10, 10, 10, 8, 8, 0], dtype=float)
        self.Lambda_p = np.array([0, 0, 0, 4, 4, 0], dtype=float)
        self.Lambda_i = np.array([1.5, 1.5, 1.0, 0.8, 0.8, 0], dtype=float)

        # eta_dot = F_err + rho_pose * pose_err
        self.rho_pose = np.array([0, 0, 0, 1, 1, 0], dtype=float)

        # =====================================================
        # 自适应参数更新增益
        # =====================================================
        self.gamma_p = np.array([2e-8, 2e-8, 2e-8, 1e-8, 1e-8, 0], dtype=float)
        self.gamma_d = np.array([2e-11, 2e-11, 2e-11, 1e-11, 1e-11, 0], dtype=float)
        self.gamma_r = np.array([0, 0, 0, 2e-4, 2e-4, 0], dtype=float)
        self.gamma_h = np.array([2e-6, 2e-6, 1e-6, 1e-6, 1e-6, 0], dtype=float)

        # 泄漏项，防止参数漂移
        self.leak_p = np.ones(6) * 0.02
        self.leak_d = np.ones(6) * 0.02
        self.leak_r = np.ones(6) * 0.02
        self.leak_h = np.ones(6) * 0.02

        # =====================================================
        # 自适应参数限幅
        # 注意：这里的自适应项最终是速度补偿，不是位置补偿
        # =====================================================
        self.p_hat_limit = np.array([2e-4, 2e-4, 2e-4, 5e-4, 5e-4, 0], dtype=float)
        self.d_hat_limit = np.array([2e-6, 2e-6, 2e-6, 5e-6, 5e-6, 0], dtype=float)
        self.r_hat_limit = np.array([0, 0, 0, 0.05, 0.05, 0], dtype=float)
        self.h_hat_limit = np.array([5e-4, 5e-4, 5e-4, 1e-3, 1e-3, 0], dtype=float)

        # 快速项，直接作为速度补偿
        self.Ks = np.array([0.003, 0.003, 0.003, 0.02, 0.02, 0.0], dtype=float)

        # sat(s / phi) 的边界层
        self.phi = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 1.0], dtype=float)

        # 力误差微分滤波
        self.force_derivative_filter = 0.90

        # =====================================================
        # 限幅
        # =====================================================
        # 虚拟位移状态限幅
        self.delta_x_limit = np.array([0.030, 0.030, 0.030, 0.15, 0.15, 0.0], dtype=float)

        # 基础导纳速度限幅
        self.delta_v_limit = np.array([0.015, 0.015, 0.010, 0.20, 0.20, 0.0], dtype=float)

        # 积分项限幅
        self.eta_limit = np.array([50.0, 50.0, 80.0, 2.0, 2.0, 0.0], dtype=float)

        # 最终速度指令限幅
        # 单位：m/s 和 rad/s
        self.v_cmd_limit = np.array([0.010, 0.010, 0.06, 0.12, 0.12, 0.0], dtype=float)

        # 自适应更新时 s 限幅，避免接触瞬间参数爆炸
        self.s_adapt_limit = np.array([500, 500, 500, 100, 100, 1], dtype=float)

    def reset(self):
        self.delta_x[:] = 0.0
        self.delta_v[:] = 0.0
        self.eta[:] = 0.0
        self.F_err_last[:] = 0.0
        self.F_err_dot_filt[:] = 0.0

        self.p_hat[:] = 0.0
        self.d_hat[:] = 0.0
        self.r_hat[:] = 0.0
        self.h_hat[:] = 0.0

        self.initialized = False

    def set_force(self, F_desire):
        """
        设置期望接触力。
        例如只有 Stewart 时：
            [0, 0, 20, 0, 0, 0]
        表示 Fx=0, Fy=0, Fz=20N, Mx=0, My=0。
        """
        self.F_desire = np.asarray(F_desire, dtype=float).reshape(6)

    def set_masks(self, active_mask=None, force_mask=None, pose_mask=None):
        if active_mask is not None:
            self.active_mask = np.asarray(active_mask, dtype=float).reshape(6)
        if force_mask is not None:
            self.force_mask = np.asarray(force_mask, dtype=float).reshape(6)
        if pose_mask is not None:
            self.pose_mask = np.asarray(pose_mask, dtype=float).reshape(6)

    def set_axis_sign(self, axis_sign):
        """
        如果某个方向越调越偏，就把对应方向改成 -1。
        """
        self.axis_sign = np.asarray(axis_sign, dtype=float).reshape(6)

    def update(self, X_meas, X_ref, F_ext, contact=True, adaptive=True, V_ff=None):
        """
        自适应导纳速度接口。

        参数：
            X_meas:
                当前末端位姿 [x, y, z, roll, pitch, yaw]

            X_ref:
                参考位姿 [x, y, z, roll, pitch, yaw]
                轴位置未知时，x_ref, y_ref 不作为接触后的强跟踪目标。

            F_ext:
                当前末端六维力 [Fx, Fy, Fz, Mx, My, Mz]
                推荐为工具坐标系下的力。

            contact:
                是否接触。

            adaptive:
                是否启用自适应项。

            V_ff:
                前馈速度 [vx, vy, vz, wx, wy, wz]
                可以用于接触前慢速搜索或接触后保持小速度推进。
                没有就默认为 0。

        返回：
            V_cmd:
                末端速度指令 [vx, vy, vz, wx, wy, wz]
                默认按工具坐标系理解。

            info:
                调试信息。
        """

        X_meas = np.asarray(X_meas, dtype=float).reshape(6)
        X_ref = np.asarray(X_ref, dtype=float).reshape(6)
        F_ext = np.asarray(F_ext, dtype=float).reshape(6)

        if V_ff is None:
            V_ff = np.zeros(6, dtype=float)
        else:
            V_ff = np.asarray(V_ff, dtype=float).reshape(6)

        dt = self.dt

        # -----------------------------------------------------
        # 1. 位姿误差
        # -----------------------------------------------------
        pose_err = self.pose_mask * (X_meas - X_ref)

        # -----------------------------------------------------
        # 2. 力误差
        # -----------------------------------------------------
        if contact:
            F_err = self.force_mask * (F_ext - self.F_desire)
        else:
            F_err = np.zeros(6, dtype=float)

        # -----------------------------------------------------
        # 3. 力误差微分 + 滤波
        # -----------------------------------------------------
        if not self.initialized:
            F_err_dot_raw = np.zeros(6, dtype=float)
            self.initialized = True
        else:
            F_err_dot_raw = (F_err - self.F_err_last) / dt

        alpha = self.force_derivative_filter
        self.F_err_dot_filt = alpha * self.F_err_dot_filt + (1.0 - alpha) * F_err_dot_raw
        F_err_dot = self.F_err_dot_filt.copy()

        self.F_err_last = F_err.copy()

        # -----------------------------------------------------
        # 4. 积分误差
        # -----------------------------------------------------
        if contact:
            eta_dot = F_err + self.rho_pose * pose_err
        else:
            # 未接触时不积分力误差
            eta_dot = self.rho_pose * pose_err

        self.eta += eta_dot * dt
        self.eta = clip_vec(self.eta, self.eta_limit)

        # -----------------------------------------------------
        # 5. 复合误差
        # -----------------------------------------------------
        s = (
            F_err_dot
            + self.Lambda_f * F_err
            + self.Lambda_p * pose_err
            + self.Lambda_i * self.eta
        )

        # -----------------------------------------------------
        # 6. 基础导纳
        # M * delta_v_dot + D * delta_v + K * delta_x
        #     = -Kf*F_err - Kp*pose_err - Ki*eta
        # -----------------------------------------------------
        rhs = (
            -self.Kf * F_err
            -self.Kp * pose_err
            -self.Ki * self.eta
        )

        delta_a = (rhs - self.D * self.delta_v - self.K * self.delta_x) / self.M

        self.delta_v += delta_a * dt
        self.delta_v = clip_vec(self.delta_v, self.delta_v_limit)

        self.delta_x += self.delta_v * dt
        self.delta_x = clip_vec(self.delta_x, self.delta_x_limit)

        # -----------------------------------------------------
        # 7. 自适应律
        # -----------------------------------------------------
        if adaptive:
            s_for_adapt = clip_vec(s, self.s_adapt_limit)

            if contact:
                self.p_hat += dt * (
                    self.gamma_p * s_for_adapt * F_err
                    - self.leak_p * self.p_hat
                )

                self.d_hat += dt * (
                    self.gamma_d * s_for_adapt * F_err_dot
                    - self.leak_d * self.d_hat
                )
            else:
                # 未接触时，力相关自适应参数只泄漏
                self.p_hat += dt * (-self.leak_p * self.p_hat)
                self.d_hat += dt * (-self.leak_d * self.d_hat)

            self.r_hat += dt * (
                self.gamma_r * s_for_adapt * pose_err
                - self.leak_r * self.r_hat
            )

            self.h_hat += dt * (
                self.gamma_h * s_for_adapt * self.eta
                - self.leak_h * self.h_hat
            )

            self.p_hat = clip_vec(self.p_hat, self.p_hat_limit)
            self.d_hat = clip_vec(self.d_hat, self.d_hat_limit)
            self.r_hat = clip_vec(self.r_hat, self.r_hat_limit)
            self.h_hat = clip_vec(self.h_hat, self.h_hat_limit)

            # -------------------------------------------------
            # 8. 自适应速度补偿
            # 注意：这里输出的是速度补偿，不是位置补偿
            # -------------------------------------------------
            V_adapt = -self.axis_sign * (
                self.p_hat * F_err
                + self.d_hat * F_err_dot
                + self.r_hat * pose_err
                + self.h_hat * self.eta
                + self.Ks * sat(s / self.phi)
            )
        else:
            V_adapt = np.zeros(6, dtype=float)

        # -----------------------------------------------------
        # 9. 最终速度输出
        # -----------------------------------------------------
        V_cmd = V_ff + self.delta_v + V_adapt

        # 只允许 active_mask 指定的方向输出
        V_cmd = self.active_mask * V_cmd

        # 最终限幅
        V_cmd = clip_vec(V_cmd, self.v_cmd_limit)

        info = {
            "pose_err": pose_err.copy(),
            "F_err": F_err.copy(),
            "F_err_dot": F_err_dot.copy(),
            "eta": self.eta.copy(),
            "s": s.copy(),
            "delta_x": self.delta_x.copy(),
            "delta_v_basic": self.delta_v.copy(),
            "V_adapt": V_adapt.copy(),
            "V_cmd": V_cmd.copy(),
            "p_hat": self.p_hat.copy(),
            "d_hat": self.d_hat.copy(),
            "r_hat": self.r_hat.copy(),
            "h_hat": self.h_hat.copy(),
        }

        return V_cmd, info