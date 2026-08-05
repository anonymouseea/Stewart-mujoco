import numpy as np


class CartesianQuinticTrajectory:
    """
    Cartesian quintic polynomial trajectory.

    The trajectory is generated independently for each Cartesian dimension:
        p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5

    Parameters are numpy-like arrays, usually shape (3,) for xyz or shape (6,)
    for [x, y, z, roll, pitch, yaw].
    """

    def __init__(
        self,
        start,
        goal,
        duration,
        start_vel=None,
        goal_vel=None,
        start_acc=None,
        goal_acc=None,
    ):
        self.start = np.asarray(start, dtype=float).reshape(-1)
        self.goal = np.asarray(goal, dtype=float).reshape(-1)
        self.duration = float(duration)

        if self.duration <= 0.0:
            raise ValueError("duration must be greater than 0.")
        if self.start.shape != self.goal.shape:
            raise ValueError("start and goal must have the same shape.")

        dim = self.start.size
        self.start_vel = self._as_vector(start_vel, dim, "start_vel")
        self.goal_vel = self._as_vector(goal_vel, dim, "goal_vel")
        self.start_acc = self._as_vector(start_acc, dim, "start_acc")
        self.goal_acc = self._as_vector(goal_acc, dim, "goal_acc")
        self.coeffs = self._compute_coeffs()

    @staticmethod
    def _as_vector(value, dim, name):
        if value is None:
            return np.zeros(dim, dtype=float)

        vec = np.asarray(value, dtype=float).reshape(-1)
        if vec.size != dim:
            raise ValueError(f"{name} must have length {dim}.")
        return vec

    def _compute_coeffs(self):
        T = self.duration

        c0 = self.start
        c1 = self.start_vel
        c2 = 0.5 * self.start_acc

        rhs = np.vstack(
            (
                self.goal - (c0 + c1 * T + c2 * T**2),
                self.goal_vel - (c1 + 2.0 * c2 * T),
                self.goal_acc - (2.0 * c2),
            )
        )

        mat = np.array(
            [
                [T**3, T**4, T**5],
                [3.0 * T**2, 4.0 * T**3, 5.0 * T**4],
                [6.0 * T, 12.0 * T**2, 20.0 * T**3],
            ],
            dtype=float,
        )

        c3_to_c5 = np.linalg.solve(mat, rhs)
        return np.vstack((c0, c1, c2, c3_to_c5))

    def evaluate(self, t):
        """
        Evaluate trajectory at time t.

        Returns:
            pos, vel, acc
        """
        t = np.clip(np.asarray(t, dtype=float), 0.0, self.duration)
        powers = np.stack((np.ones_like(t), t, t**2, t**3, t**4, t**5), axis=0)
        pos = np.tensordot(powers, self.coeffs, axes=(0, 0))

        vel_powers = np.stack(
            (np.zeros_like(t), np.ones_like(t), 2.0 * t, 3.0 * t**2, 4.0 * t**3, 5.0 * t**4),
            axis=0,
        )
        vel = np.tensordot(vel_powers, self.coeffs, axes=(0, 0))

        acc_powers = np.stack(
            (np.zeros_like(t), np.zeros_like(t), 2.0 * np.ones_like(t), 6.0 * t, 12.0 * t**2, 20.0 * t**3),
            axis=0,
        )
        acc = np.tensordot(acc_powers, self.coeffs, axes=(0, 0))

        if np.ndim(t) == 0:
            return pos.reshape(-1), vel.reshape(-1), acc.reshape(-1)
        return pos, vel, acc

    def sample(self, dt=None, num=None):
        """
        Sample the whole trajectory.

        Use either dt or num:
            dt: sample interval, final point is included
            num: number of points

        Returns:
            ts, pos, vel, acc
        """
        if (dt is None) == (num is None):
            raise ValueError("Use exactly one of dt or num.")

        if dt is not None:
            dt = float(dt)
            if dt <= 0.0:
                raise ValueError("dt must be greater than 0.")
            ts = np.arange(0.0, self.duration + 0.5 * dt, dt)
            if ts[-1] > self.duration:
                ts[-1] = self.duration
            elif ts[-1] < self.duration:
                ts = np.append(ts, self.duration)
        else:
            num = int(num)
            if num < 2:
                raise ValueError("num must be at least 2.")
            ts = np.linspace(0.0, self.duration, num)

        pos, vel, acc = self.evaluate(ts)
        return ts, pos, vel, acc


def cartesian_quintic_interpolation(
    start,
    goal,
    duration,
    dt=None,
    num=None,
    start_vel=None,
    goal_vel=None,
    start_acc=None,
    goal_acc=None,
):
    """
    Generate a Cartesian quintic interpolation trajectory.

    Example:
        from src.traj import cartesian_quintic_interpolation

        ts, pos, vel, acc = cartesian_quintic_interpolation(
            start=[0.0, -0.49, 0.84],
            goal=[0.05, -0.45, 0.90],
            duration=2.0,
            dt=0.001,
        )
    """
    traj = CartesianQuinticTrajectory(
        start=start,
        goal=goal,
        duration=duration,
        start_vel=start_vel,
        goal_vel=goal_vel,
        start_acc=start_acc,
        goal_acc=goal_acc,
    )
    return traj.sample(dt=dt, num=num)


def create_cartesian_quintic_trajectory(
    start,
    goal,
    duration,
    start_vel=None,
    goal_vel=None,
    start_acc=None,
    goal_acc=None,
):
    """
    Create a trajectory object for real-time control loops.

    Example:
        from src.traj import create_cartesian_quintic_trajectory

        traj = create_cartesian_quintic_trajectory(start, goal, duration=2.0)
        pos_ref, vel_ref, acc_ref = traj.evaluate(t)
    """
    return CartesianQuinticTrajectory(
        start=start,
        goal=goal,
        duration=duration,
        start_vel=start_vel,
        goal_vel=goal_vel,
        start_acc=start_acc,
        goal_acc=goal_acc,
    )


__all__ = [
    "CartesianQuinticTrajectory",
    "cartesian_quintic_interpolation",
    "create_cartesian_quintic_trajectory",
]
