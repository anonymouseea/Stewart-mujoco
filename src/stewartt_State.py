"""Stewart 平台控制状态机。"""

from dataclasses import dataclass
from enum import Enum, auto
import math

import numpy as np


class ControlState(Enum):
    """Stewart 控制流程的基础状态；可按任务继续增加状态。"""

    CALIBRATING = auto()
    REACHING = auto()
    CONTACT = auto()
    FORCE_CONTROL = auto()
    FINISHED = auto()
    FAULT = auto()


@dataclass(frozen=True)
class StateTransition:
    previous: ControlState
    current: ControlState
    time: float
    reason: str = ""


class StewartStateMachine:
    """只管理状态与跳转，不直接依赖 MuJoCo 或真机接口。"""

    def __init__(
        self,
        calibration_duration=0.1,
        contact_threshold=1.0,
        target_force_z=-10.0,
        force_tolerance=0.01,
        wrench_tracking_tolerance=(0.5, 0.5, 0.5, 0.05, 0.05, 0.05),
        wrench_tracking_duration=0.2,
        verbose=True,
    ):
        calibration_duration = float(calibration_duration)
        if calibration_duration < 0.0:
            raise ValueError("calibration_duration不能小于0")

        contact_threshold = float(contact_threshold)
        force_tolerance = float(force_tolerance)
        wrench_tracking_tolerance = np.asarray(
            wrench_tracking_tolerance,
            dtype=float,
        ).reshape(-1)
        wrench_tracking_duration = float(wrench_tracking_duration)
        if contact_threshold < 0.0:
            raise ValueError("contact_threshold不能小于0")
        if force_tolerance < 0.0:
            raise ValueError("force_tolerance不能小于0")
        if wrench_tracking_tolerance.size != 6:
            raise ValueError("wrench_tracking_tolerance必须包含6个分量")
        if not np.all(np.isfinite(wrench_tracking_tolerance)):
            raise ValueError("wrench_tracking_tolerance必须是有限数值")
        if np.any(wrench_tracking_tolerance < 0.0):
            raise ValueError("wrench_tracking_tolerance不能小于0")
        if wrench_tracking_duration < 0.0:
            raise ValueError("wrench_tracking_duration不能小于0")

        self.calibration_duration = calibration_duration
        self.contact_threshold = contact_threshold
        self.target_force_z = float(target_force_z)
        self.force_tolerance = force_tolerance
        self.wrench_tracking_tolerance = wrench_tracking_tolerance.copy()
        self.wrench_tracking_duration = wrench_tracking_duration
        self.verbose = bool(verbose)
        self.wrench_match_start_time = None
        self.last_wrench_error = None
        self.state = ControlState.CALIBRATING
        self.enter_time = 0.0
        self.last_transition = StateTransition(
            ControlState.CALIBRATING,
            ControlState.CALIBRATING,
            0.0,
            "initialized",
        )

    @staticmethod
    def _check_time(now):
        now = float(now)
        if not math.isfinite(now):
            raise ValueError("状态机时间必须是有限数值")
        return now

    def reset(self, now=0.0):
        """从传感器标定状态重新开始。"""
        now = self._check_time(now)
        previous = self.state
        self.state = ControlState.CALIBRATING
        self.enter_time = now
        self.last_transition = StateTransition(
            previous,
            self.state,
            now,
            "reset",
        )
        self.wrench_match_start_time = None
        self.last_wrench_error = None
        return self.state

    def transition_to(self, new_state, now, reason=""):
        """切换状态；重复切换到当前状态时不重置计时。"""
        if not isinstance(new_state, ControlState):
            raise TypeError("new_state必须是ControlState")

        now = self._check_time(now)
        if new_state == self.state:
            return False

        previous = self.state
        self.state = new_state
        self.enter_time = now
        self.last_transition = StateTransition(
            previous,
            new_state,
            now,
            str(reason),
        )
        self.wrench_match_start_time = None
        if self.verbose:
            reason_text = f": {reason}" if reason else ""
            print(
                f"[state] {previous.name} -> {new_state.name} "
                f"@ {now:.3f}s{reason_text}"
            )
        return True

    def elapsed(self, now):
        """返回当前状态已经持续的时间。"""
        now = self._check_time(now)
        return max(0.0, now - self.enter_time)

    @staticmethod
    def _as_wrench(value, name):
        wrench = np.asarray(value, dtype=float).reshape(-1)
        if wrench.size != 6:
            raise ValueError(f"{name}必须包含6个分量")
        if not np.all(np.isfinite(wrench)):
            raise ValueError(f"{name}必须是有限数值")
        return wrench

    def update(
        self,
        now,
        wrench=None,
        reachable=True,
        desired_wrench=None,
    ):
        """执行基础自动跳转，并返回更新后的状态。"""
        now = self._check_time(now)

        if not reachable:
            self.fail(now, "目标不可达")
            return self.state

        if (
            self.state == ControlState.CALIBRATING
            and self.elapsed(now) >= self.calibration_duration
        ):
            self.transition_to(
                ControlState.REACHING,
                now,
                "传感器标定完成，开始接近",
            )

        elif self.state == ControlState.REACHING:
            if (
                wrench is not None
                and abs(float(wrench[2])) >= self.contact_threshold
            ):
                self.transition_to(
                    ControlState.CONTACT,
                    now,
                    "检测到接触",
                )

        elif self.state == ControlState.CONTACT:
            if desired_wrench is not None:
                force_error = abs(
                    float(desired_wrench[2]) - self.target_force_z
                )
                if force_error <= self.force_tolerance:
                    self.transition_to(
                        ControlState.FORCE_CONTROL,
                        now,
                        "目标力已建立",
                    )

        elif self.state == ControlState.FORCE_CONTROL:
            # runFunc 会先做一次无参数 update；无测量值时保持计时不变。
            if wrench is not None and desired_wrench is not None:
                measured = self._as_wrench(wrench, "wrench")
                desired = self._as_wrench(desired_wrench, "desired_wrench")
                self.last_wrench_error = np.abs(measured - desired)

                if np.all(
                    self.last_wrench_error
                    <= self.wrench_tracking_tolerance
                ):
                    if self.wrench_match_start_time is None:
                        self.wrench_match_start_time = now

                    matched_duration = now - self.wrench_match_start_time
                    if matched_duration >= self.wrench_tracking_duration:
                        self.finish(now,"六维力/力矩持续满足容差，对接完成",)
                else:
                    # 任意一个轴超出容差，都重新开始稳定计时。
                    self.wrench_match_start_time = None

        return self.state


    def finish(self, now, reason="task completed"):
        return self.transition_to(ControlState.FINISHED, now, reason)

    def fail(self, now, reason):
        return self.transition_to(ControlState.FAULT, now, reason)

    @property
    def is_terminal(self):
        return self.state in {ControlState.FINISHED, ControlState.FAULT}
