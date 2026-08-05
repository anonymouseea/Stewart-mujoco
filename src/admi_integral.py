import numpy as np


class IntegralAdmittance:
    def __init__(
        self,
        M,
        D,
        K,
        dt,
        Ki=None,
        integral_limit=None,
        integral_leak=0.9995,
    ) -> None:
        self.M = np.asarray(M, dtype=float).reshape(6)
        self.D = np.asarray(D, dtype=float).reshape(6)
        self.K = np.asarray(K, dtype=float).reshape(6)
        self.dt = float(dt)

        if np.any(self.M == 0.0):
            raise ValueError("导纳控制的质量 M 不能为 0，否则会除以 0。")

        # Ki 是力/力矩误差积分增益。
        # 建议位置轴先保持 0，只给姿态力矩轴积分，避免位置慢慢漂移。
        if Ki is None:
            Ki = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
        self.Ki = np.asarray(Ki, dtype=float).reshape(6)

        # 积分限幅用于防止长时间残差造成积分风up。
        if integral_limit is None:
            integral_limit = [0.0, 0.0, 0.0, 0.2, 0.2, 0.2]
        self.integral_limit = np.asarray(integral_limit, dtype=float).reshape(6)

        # 积分泄漏让历史误差缓慢衰减，避免旧误差一直残留。
        self.integral_leak = float(integral_leak)

        self.x = np.zeros(6, dtype=float)
        self.xdot = np.zeros(6, dtype=float)
        self.xdesir = np.zeros(6, dtype=float)
        self.F_desire = np.zeros(6, dtype=float)
        self.F_integral = np.zeros(6, dtype=float)

    def set_state(self, x: np.ndarray, xdesir: np.ndarray):
        self.x = np.asarray(x, dtype=float).copy()
        self.xdesir = np.asarray(xdesir, dtype=float).copy()
        self.xdot[:] = 0.0
        self.F_integral[:] = 0.0

    def set_force(self, F_desire: np.ndarray):
        self.F_desire = np.asarray(F_desire, dtype=float).copy()

    def set_stated(self, xdesir: np.ndarray):
        self.xdesir = np.asarray(xdesir, dtype=float)

    def update_martix(self, matrix):
        self.matrix = np.asarray(matrix, dtype=float).copy()

    def reset_integral(self):
        self.F_integral[:] = 0.0

    def update(self, F_ext: np.ndarray):
        F_ext = np.asarray(F_ext, dtype=float).reshape(6)
        F_err = F_ext - self.F_desire

        # 对误差积分：小的残余力矩会被慢慢累积，继续推动姿态修正。
        self.F_integral = self.integral_leak * self.F_integral + F_err * self.dt
        self.F_integral = np.clip(
            self.F_integral,
            -self.integral_limit,
            self.integral_limit,
        )

        # 当前误差负责快速响应，积分误差负责消除最后的小残差。
        F_drive = F_err + self.Ki * self.F_integral

        xddot = (
            F_drive
            - self.D * self.xdot
            - self.K * (self.x - self.xdesir)
        ) / self.M

        self.xdot += xddot * self.dt
        self.x += self.xdot * self.dt

        return self.x.copy()
