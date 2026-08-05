import numpy as np


class TorqueOrientationJacobian:
    """
    力矩-姿态 Jacobian 估计器。

    目标是估计下面这个局部线性关系：
        delta_tau = J * delta_theta

    其中：
        delta_tau   是工具坐标系下的力矩变化 [Mx, My, Mz]
        delta_theta 是工具坐标系下的姿态微小变化，建议用旋转向量 [rx, ry, rz]
        J           是 3x3 的力矩-姿态 Jacobian

    得到 J 以后，可以用：
        delta_theta = -alpha * pinv(J) * tau

    直接根据当前残余力矩 tau 计算姿态修正量。
    """

    def __init__(
        self,
        alpha=0.2,
        damping=1e-4,
        max_step=np.deg2rad(0.2),
        update_gain=0.05,
        min_motion=np.deg2rad(0.02),
    ):
        # 修正步长，越大收敛越快，但太大容易振荡。
        self.alpha = float(alpha)

        # 阻尼伪逆参数，用于避免 J 接近奇异时修正量爆炸。
        self.damping = float(damping)

        # 单次输出的最大姿态修正量，单位 rad。
        self.max_step = float(max_step)

        # 在线更新 J 的速度，0 表示不更新，1 表示完全采用新估计。
        self.update_gain = float(update_gain)

        # 姿态变化太小时，delta_tau / delta_theta 容易被噪声放大，因此跳过更新。
        self.min_motion = float(min_motion)

        self.J = np.eye(3, dtype=float)
        self.prev_theta = None
        self.prev_tau = None

    def reset(self, J=None):
        """重置估计器状态，可以传入一个已有的 3x3 Jacobian。"""
        if J is None:
            self.J = np.eye(3, dtype=float)
        else:
            self.J = np.asarray(J, dtype=float).reshape(3, 3)
        self.prev_theta = None
        self.prev_tau = None

    def set_jacobian(self, J):
        """直接设置离线标定得到的 Jacobian。"""
        self.J = np.asarray(J, dtype=float).reshape(3, 3)

    def fit_from_samples(self, theta_samples, tau_samples):
        """
        用一组离线样本拟合 J。

        theta_samples: N x 3，姿态样本，建议是相对初始姿态的旋转向量，单位 rad
        tau_samples:   N x 3，对应力矩样本，单位 N*m

        拟合关系：
            tau - tau0 = J * (theta - theta0)

        返回：
            J: 3x3 Jacobian
        """
        theta_samples = np.asarray(theta_samples, dtype=float)
        tau_samples = np.asarray(tau_samples, dtype=float)

        if theta_samples.ndim != 2 or theta_samples.shape[1] != 3:
            raise ValueError("theta_samples 必须是 N x 3。")
        if tau_samples.ndim != 2 or tau_samples.shape[1] != 3:
            raise ValueError("tau_samples 必须是 N x 3。")
        if theta_samples.shape[0] != tau_samples.shape[0]:
            raise ValueError("theta_samples 和 tau_samples 的样本数量必须一致。")
        if theta_samples.shape[0] < 4:
            raise ValueError("至少需要 4 组样本，才能比较稳地拟合 3x3 Jacobian。")

        delta_theta = theta_samples - theta_samples[0]
        delta_tau = tau_samples - tau_samples[0]

        # 最小二乘：delta_tau.T = J * delta_theta.T
        self.J = delta_tau.T @ np.linalg.pinv(delta_theta.T)
        return self.J.copy()

    def update_online(self, theta, tau):
        """
        在线递推更新 J。

        theta: 当前姿态，相对初始姿态的旋转向量，单位 rad
        tau:   当前残余力矩 [Mx, My, Mz]，单位 N*m

        这个方法只负责更新 J，不返回控制量。
        """
        theta = np.asarray(theta, dtype=float).reshape(3)
        tau = np.asarray(tau, dtype=float).reshape(3)

        if self.prev_theta is None:
            self.prev_theta = theta.copy()
            self.prev_tau = tau.copy()
            return self.J.copy()

        delta_theta = theta - self.prev_theta
        delta_tau = tau - self.prev_tau

        motion_norm = np.linalg.norm(delta_theta)
        if motion_norm >= self.min_motion:
            # 单步割线估计：delta_tau ≈ J_step * delta_theta
            # 用外积得到满足当前样本的一阶近似，再与旧 J 低通融合。
            J_step = np.outer(delta_tau, delta_theta) / np.dot(delta_theta, delta_theta)
            self.J = (1.0 - self.update_gain) * self.J + self.update_gain * J_step

        self.prev_theta = theta.copy()
        self.prev_tau = tau.copy()
        return self.J.copy()

    def correction(self, tau):
        """
        根据当前残余力矩计算姿态修正量。

        tau: 当前残余力矩 [Mx, My, Mz]，单位 N*m

        返回：
            delta_theta: 工具坐标系下的旋转向量修正量，单位 rad
        """
        tau = np.asarray(tau, dtype=float).reshape(3)

        JT = self.J.T
        damped_inverse = np.linalg.inv(self.J @ JT + self.damping * np.eye(3))
        J_pinv = JT @ damped_inverse

        delta_theta = -self.alpha * (J_pinv @ tau)
        step_norm = np.linalg.norm(delta_theta)

        if step_norm > self.max_step:
            delta_theta = delta_theta / step_norm * self.max_step

        return delta_theta

    def update_and_correct(self, theta, tau):
        """
        在线更新 J，并立即计算姿态修正量。

        适合在控制循环里使用：
            estimator.update_and_correct(delta_tool[3:], ft_tool[3:])
        """
        self.update_online(theta, tau)
        return self.correction(tau)
