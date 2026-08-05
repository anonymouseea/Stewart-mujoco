import numpy as np


class Admittance:
    """
    6-DOF fractional-order admittance controller.

    Basic admittance:
        M * xddot + D * xdot + K * (x - xdesir) = F_ext - F_desire

    Fractional-order admittance, adapted from KDHD/KDHDc idea:
        M * xddot
        + D * xdot
        + Kc * (x - xdesir)
        + HD * D^alpha[x - xdesir]
        = F_ext - F_desire

    where alpha is usually 0.5.

    If compensate_stiffness=True:
        Kc = K - HD * W_alpha(n) / dt^alpha

    This cancels the artificial steady-state stiffness introduced by the finite-memory
    Grünwald-Letnikov fractional derivative filter, so the static relation is still
    mainly determined by K.
    """

    def __init__(
        self,
        M: float,
        D: float,
        K: float,
        dt: float,
        HD=None,
        frac_order: float = 0.5,
        frac_filter_order: int = 10,
        compensate_stiffness: bool = True,
    ) -> None:
        self.M = np.asarray(M, dtype=float).reshape(6)
        self.D = np.asarray(D, dtype=float).reshape(6)
        self.K = np.asarray(K, dtype=float).reshape(6)
        self.dt = float(dt)

        if np.any(self.M == 0.0):
            raise ValueError("导纳控制的质量 M 不能为 0，否则会除以 0。")

        if self.dt <= 0.0:
            raise ValueError("采样周期 dt 必须大于 0。")

        # 半阶阻尼系数；不传则等价于原来的普通导纳
        if HD is None:
            HD = np.zeros(6, dtype=float)
        self.HD = np.asarray(HD, dtype=float).reshape(6)

        self.frac_order = float(frac_order)
        if not (0.0 < self.frac_order < 1.0):
            raise ValueError("frac_order 建议设置在 (0, 1) 内，半阶导数取 0.5。")

        self.frac_filter_order = int(frac_filter_order)
        if self.frac_filter_order < 1:
            raise ValueError("frac_filter_order 至少为 1。")

        self.compensate_stiffness = bool(compensate_stiffness)

        # Grünwald-Letnikov 有限记忆滤波器权重
        self.frac_weights = self._gl_weights(self.frac_order, self.frac_filter_order)

        # W_alpha(n) = sum(w_j)，有限阶滤波器对常值输入会产生非零输出
        self.W_alpha = float(np.sum(self.frac_weights))
        self.frac_gain = self.W_alpha / (self.dt ** self.frac_order)

        # 误差历史：第 0 行为当前误差，第 1 行为上一拍误差……
        self.err_history = np.zeros((self.frac_filter_order + 1, 6), dtype=float)

        self.x = np.zeros(6, dtype=float)
        self.xdot = np.zeros(6, dtype=float)
        self.xdesir = np.zeros(6, dtype=float)

        # 期望力 / 力矩
        self.F_desire = np.zeros(6, dtype=float)

    @staticmethod
    def _gl_weights(alpha: float, order: int) -> np.ndarray:
        """
        Grünwald-Letnikov fractional derivative weights:
            w_0 = 1
            w_j = (1 - (alpha + 1) / j) * w_{j-1}
        """
        w = np.zeros(order + 1, dtype=float)
        w[0] = 1.0
        for j in range(1, order + 1):
            w[j] = (1.0 - (alpha + 1.0) / j) * w[j - 1]
        return w

    def reset_fractional_history(self, xerr=None):
        """
        清空/重置分数阶滤波器历史。
        建议每次切换控制模式、重新设定初始位姿时调用。
        """
        if xerr is None:
            xerr = self.x - self.xdesir
        xerr = np.asarray(xerr, dtype=float).reshape(6)
        self.err_history[:] = xerr

    def set_state(self, x: np.ndarray, xdesir: np.ndarray):
        self.x = np.asarray(x, dtype=float).reshape(6).copy()
        self.xdesir = np.asarray(xdesir, dtype=float).reshape(6).copy()
        self.xdot[:] = 0.0
        self.reset_fractional_history(self.x - self.xdesir)

    def set_force(self, F_desire: np.ndarray):
        self.F_desire = np.asarray(F_desire, dtype=float).reshape(6).copy()

    def set_stated(self, xdesir: np.ndarray):
        self.xdesir = np.asarray(xdesir, dtype=float).reshape(6).copy()
        self.reset_fractional_history(self.x - self.xdesir)

    def update_matrix(self, M=None, D=None, K=None, HD=None):
        """
        在线更新导纳参数。传 None 表示不改。
        """
        if M is not None:
            self.M = np.asarray(M, dtype=float).reshape(6)
            if np.any(self.M == 0.0):
                raise ValueError("导纳控制的质量 M 不能为 0，否则会除以 0。")
        if D is not None:
            self.D = np.asarray(D, dtype=float).reshape(6)
        if K is not None:
            self.K = np.asarray(K, dtype=float).reshape(6)
        if HD is not None:
            self.HD = np.asarray(HD, dtype=float).reshape(6)

    # 保留你原来的函数名，避免其他代码调用时报错
    def update_martix(self, matrix):
        self.matrix = np.asarray(matrix, dtype=float).copy()

    def _fractional_derivative_error(self, xerr: np.ndarray) -> np.ndarray:
        # 更新误差历史
        self.err_history[1:] = self.err_history[:-1]
        self.err_history[0] = xerr

        # D^alpha[xerr] ≈ 1/dt^alpha * sum_j w_j * xerr[k-j]
        return (self.frac_weights @ self.err_history) / (self.dt ** self.frac_order)

    def update(self, F_ext: np.ndarray):
        F_ext = np.asarray(F_ext, dtype=float).reshape(6)
        F_err = F_ext - self.F_desire

        xerr = self.x - self.xdesir
        frac_xerr = self._fractional_derivative_error(xerr)

        if self.compensate_stiffness:
            # KDHDc思想：补偿有限阶分数阶滤波器引入的“虚假稳态刚度”
            K_effective = self.K - self.HD * self.frac_gain
        else:
            # 不补偿时，半阶项会在稳态表现为额外刚度
            K_effective = self.K

        xddot = (
            F_err
            - self.D * self.xdot
            - K_effective * xerr
            - self.HD * frac_xerr
        ) / self.M

        # 半隐式欧拉：先更新速度，再更新位置
        self.xdot += xddot * self.dt
        self.x += self.xdot * self.dt

        return self.x.copy()
