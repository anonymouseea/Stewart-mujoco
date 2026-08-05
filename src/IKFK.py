import numpy as np
from src.utils import *
#电缸零位时杆长为549mm
ZERO_SLIDE_LENGTH = 0.549

#电缸的伸长量范围
SLIDE_MIN = 0.0
SLIDE_MAX = 0.2

#球铰最大摆角
BALL_JOINT_MAX_TILT = np.deg2rad(30.0)

# 坐标来自 stewart_new.xml。
# 第 i 列表示第 i 条腿连接的下球铰点。
BASE_POINTS = np.array(
    [
        [0.19319, 0.05176, 0.0185],
        [0.14142, 0.14142, 0.0185],
        [-0.14142, 0.14142, 0.0185],
        [-0.19319, 0.05176, 0.0185],
        [0.05176, -0.19319, 0.0185],
        [-0.05176, -0.19319, 0.0185],
    ],
    dtype=float,
).T

# stewart_new.xml 中 equality 约束的连接关系：
# leg1->shangjia1，leg2->shangjia6，leg3->shangjia5，
# leg4->shangjia4，leg5->shangjia2，leg6->shangjia3。
PLATFORM_POINTS_LOCAL = np.array(
    [
        [0.15455, -0.04141, -0.0185],
        [0.04141, 0.15455, -0.0185],
        [-0.04141, 0.15455, -0.0185],
        [-0.15455, -0.04141, -0.0185],
        [0.11340, -0.11314, -0.0185],
        [-0.11314, -0.11314, -0.0185],
    ],
    dtype=float,
).T

# tool site 相对 Stewart 平台 center 的固定安装偏移。
# 该值与 stewart_new.xml 中的当前 body/site 变换保持一致。
TOOL_1_OFFSET_CENTER = np.array([0.0, -0.485, 0.160], dtype=float)
TOOL_1_ROT_CENTER = euler2rotmat(np.deg2rad(90.0), 0.0, 0.0)
TOOL_1_DEFAULT_ROT_WORLD = TOOL_1_ROT_CENTER

def tool_rot_world_from_rpy(tool_rpy_tool):
    """
    根据工具坐标系下的相对姿态命令，得到工具坐标系相对于世界坐标系的旋转矩阵。
    """
    tool_rpy_tool = np.asarray(tool_rpy_tool, dtype=float).reshape(3)
    R_tool_delta = euler2rotmat(*tool_rpy_tool)
    return TOOL_1_DEFAULT_ROT_WORLD @ R_tool_delta


def tool_local_vector_to_world(vector_tool, tool_rpy_tool):
    """
    把工具坐标系 T 下的向量转换到世界坐标系 W 下。
    可用于把工具局部坐标系下的位移命令转换成世界坐标位移。
    """
    vector_tool = np.asarray(vector_tool, dtype=float).reshape(3)
    return tool_rot_world_from_rpy(tool_rpy_tool) @ vector_tool


def stewart_ik(pos, rpy):
    """
    按 MuJoCo XML 中的实际几何计算 Stewart 平台逆解。

    pos 是 site "center" 的世界坐标，不是上铰点平面的坐标。
    rpy 是 roll/pitch/yaw，单位为弧度。
    返回六条腿的总长度，以及与六条腿对应的上铰点世界坐标。
    """
    
    #获得目标点的位姿，姿态采用欧拉角xyz转旋转矩阵
    pos = np.asarray(pos, dtype=float).reshape(3, 1)
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    
    #得到旋转矩阵
    rot = euler2rotmat(roll, pitch, yaw)
    
    #上平台六个连接点的世界坐标=平台中心坐标+旋转矩阵@平台连接点的局部坐标
    top_points = pos + rot @ PLATFORM_POINTS_LOCAL
    
    #杆长=上铰点坐标-下铰点坐标的欧几里得距离
    lengths = np.linalg.norm(top_points - BASE_POINTS, axis=0)
    
    return lengths, top_points


def platform_pose_from_tool_pose(tool_pos_world, tool_rpy_tool):
    """
    输入：末端工具在世界坐标系下的目标位置，以及工具坐标系下的相对姿态命令。
    tool_rpy_tool = [0, 0, 0] 时，工具保持 XML 中 site tool 的默认安装姿态：
    等价于世界系下 [90deg, 0, 0]。
    输出：为了让工具到达该位姿，平台 center 应该到达的位姿
    """
    tool_pos_world = np.asarray(tool_pos_world, dtype=float).reshape(3)
    tool_rpy_tool = np.asarray(tool_rpy_tool, dtype=float).reshape(3)

    R_WT = tool_rot_world_from_rpy(tool_rpy_tool)
    p_PT = TOOL_1_OFFSET_CENTER
    R_PT = TOOL_1_ROT_CENTER
    R_WP = R_WT @ R_PT.T
    p_WP = tool_pos_world - R_WP @ p_PT
    platform_rpy_world = rot2eul(R_WP)

    return p_WP, platform_rpy_world

def stewart_ik_tool(tool_pos_world, tool_rpy_tool, return_platform_pose=False):
    """
    输入末端工具在世界坐标系下的目标位置，以及工具坐标系下的相对姿态命令，
    自动换算成平台 center 位姿，
    再调用 stewart_ik 求六条腿长度。
    """
    platform_pos, platform_rpy = platform_pose_from_tool_pose(
        tool_pos_world,
        tool_rpy_tool
    )

    lengths, top_points = stewart_ik(platform_pos, platform_rpy)

    if return_platform_pose:
        return lengths, top_points, platform_pos, platform_rpy

    return lengths, top_points


def slide_ctrl_to_leg_length(ctrl, base_length=ZERO_SLIDE_LENGTH):
    """把 MuJoCo 滑动关节控制量转换为总杆长。"""
    return np.asarray(ctrl, dtype=float) + base_length


def leg_length_to_slide_ctrl(lengths, base_length=ZERO_SLIDE_LENGTH):
    """把总杆长转换为 MuJoCo 滑动关节控制量。"""
    ctrl = np.asarray(lengths, dtype=float) - base_length
    return np.clip(ctrl, SLIDE_MIN, SLIDE_MAX)



#倾斜角计算函数，输入是方向向量和轴向量，输出是方向向量相对于轴的倾斜角，单位为弧度。
def _tilt_angles(directions, axis):
    
    axis = np.asarray(axis, dtype=float).reshape(3, 1)
    
    axis = axis / np.linalg.norm(axis)
    
    directions = np.asarray(directions, dtype=float)
    
    directions = directions / np.linalg.norm(directions, axis=0)
    
    cos_angles = np.sum(directions * axis, axis=0)
    
    return np.arccos(np.clip(cos_angles, -1.0, 1.0))


def ball_joint_tilt_angles(pos, rpy):
    """
    返回球铰摆角，单位 rad。

    lower_angles: 杆方向相对世界系 +Z 的摆角。
    upper_angles: 杆方向相对平台局部 -Z 的摆角。
    绕杆自身 z 轴的扭转不参与计算。
    """
    lengths, top_points = stewart_ik(pos, rpy)
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    rot = euler2rotmat(roll, pitch, yaw)

    lower_dirs_world = (top_points - BASE_POINTS) / lengths
    upper_dirs_local = rot.T @ ((BASE_POINTS - top_points) / lengths)

    lower_angles = _tilt_angles(lower_dirs_world, [0.0, 0.0, 1.0])
    upper_angles = _tilt_angles(upper_dirs_local, [0.0, 0.0, -1.0])
    return lower_angles, upper_angles


def _pose_to_lengths(pose):
    """pose = [x, y, z, roll, pitch, yaw]，返回六条腿总长度。"""
    
    lengths, _ = stewart_ik(pose[:3], pose[3:])
    
    return lengths


def _numeric_jacobian(pose, fd_eps):
    """用中心差分计算杆长对位姿变量的雅可比矩阵。"""
    pose = np.asarray(pose, dtype=float)
    jac = np.zeros((6, 6), dtype=float)

    for i in range(6):
        step = np.zeros(6, dtype=float)
        step[i] = fd_eps
        jac[:, i] = (_pose_to_lengths(pose + step) - _pose_to_lengths(pose - step)) / (
            2.0 * fd_eps
        )

    return jac


def _normalize_rpy(rpy):
    """把欧拉角约束到 [-pi, pi]，避免迭代时角度无意义漂移。"""
    return (np.asarray(rpy, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def stewart_fk(
    lengths,
    initial_pose=None,
    max_iter=100,
    tol=1e-10,
    damping=1e-3,
    fd_eps=1e-6,
    return_info=False,
):
    """
    使用 Levenberg-Marquardt 算法求 Stewart 平台正运动学。

    lengths 是六条腿的总长度，顺序必须与 stewart_ik 的返回顺序一致。
    initial_pose 可选，格式为 [x, y, z, roll, pitch, yaw]。
    默认初值按当前模型常用工作区设置为 [0, 0, 0.65, 0, 0, 0]。

    默认返回 (pos, rpy)。
    如果 return_info=True，则返回 (pos, rpy, info)，其中 info 包含收敛状态、
    迭代次数、最终残差范数和最终阻尼系数。
    """
    target_lengths = np.asarray(lengths, dtype=float).reshape(6)

    if initial_pose is None:
        pose = np.array([0.0, 0.0, 0.65, 0.0, 0.0, 0.0], dtype=float)
    else:
        pose = np.asarray(initial_pose, dtype=float).reshape(6).copy()

    lm = float(damping)
    converged = False
    residual = _pose_to_lengths(pose) - target_lengths
    err = np.linalg.norm(residual)

    for iteration in range(1, max_iter + 1):
        if err < tol:
            converged = True
            break

        jac = _numeric_jacobian(pose, fd_eps)
        hessian = jac.T @ jac
        gradient = jac.T @ residual
        scale = np.diag(np.maximum(np.diag(hessian), 1.0))

        try:
            delta = np.linalg.solve(hessian + lm * scale, -gradient)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(hessian + lm * scale, -gradient, rcond=None)[0]

        if np.linalg.norm(delta) < tol:
            converged = True
            break

        trial_pose = pose + delta
        trial_pose[3:] = _normalize_rpy(trial_pose[3:])
        trial_residual = _pose_to_lengths(trial_pose) - target_lengths
        trial_err = np.linalg.norm(trial_residual)

        if trial_err < err:
            pose = trial_pose
            residual = trial_residual
            err = trial_err
            lm = max(lm * 0.3, 1e-12)
        else:
            lm = min(lm * 2.0, 1e12)

    converged = converged or err < tol
    pos = pose[:3].copy()
    rpy = _normalize_rpy(pose[3:])

    if return_info:
        info = {
            "converged": converged,
            "iterations": iteration if max_iter > 0 else 0,
            "residual_norm": err,
            "damping": lm,
        }
        return pos, rpy, info

    return pos, rpy





def is_pose_reachable(
    pos,
    rpy,
    slide_min=SLIDE_MIN,
    slide_max=SLIDE_MAX,
    base_length=ZERO_SLIDE_LENGTH,
    ball_joint_max_tilt=BALL_JOINT_MAX_TILT,
    check_upper_ball=True,
    return_info=False,
):
    """
    判断目标位姿是否在六个伸缩关节的行程范围内。

    pos 是平台 center 的目标位置 [x, y, z]，单位 m。
    rpy 是目标姿态 [roll, pitch, yaw]，单位 rad。
    默认伸缩关节范围来自 stewart_new.xml: 0 ~ 0.2 m。
    """
    lengths, top_points = stewart_ik(pos, rpy)
    slide_ctrl = lengths - base_length
    low_margin = slide_ctrl - slide_min
    high_margin = slide_max - slide_ctrl
    stroke_ok = bool(np.all(low_margin >= 0.0) and np.all(high_margin >= 0.0))

    lower_ball_angles, upper_ball_angles = ball_joint_tilt_angles(pos, rpy)
    lower_ball_ok = bool(np.all(lower_ball_angles <= ball_joint_max_tilt))
    upper_ball_ok = bool(np.all(upper_ball_angles <= ball_joint_max_tilt))
    ball_joint_ok = lower_ball_ok and (upper_ball_ok or not check_upper_ball)

    reachable = stroke_ok and ball_joint_ok

    if not return_info:
        return reachable

    info = {
        "reachable": reachable,
        "stroke_ok": stroke_ok,
        "ball_joint_ok": ball_joint_ok,
        "lower_ball_ok": lower_ball_ok,
        "upper_ball_ok": upper_ball_ok,
        "lengths": lengths,
        "slide_ctrl": slide_ctrl,
        "top_points": top_points,
        "low_margin": low_margin,
        "high_margin": high_margin,
        "lower_ball_angles": lower_ball_angles,
        "upper_ball_angles": upper_ball_angles,
        "lower_ball_angles_deg": np.rad2deg(lower_ball_angles),
        "upper_ball_angles_deg": np.rad2deg(upper_ball_angles),
        "bad_legs": np.where((slide_ctrl < slide_min) | (slide_ctrl > slide_max))[0] + 1,
        "bad_lower_ball_legs": np.where(lower_ball_angles > ball_joint_max_tilt)[0] + 1,
        "bad_upper_ball_legs": np.where(upper_ball_angles > ball_joint_max_tilt)[0] + 1,
    }
    return reachable, info


def is_tool_pose_reachable(tool_pos_world, tool_rpy_tool, return_info=False):
    """
    判断末端工具目标位姿是否可达。

    tool_pos_world 是工具原点在世界坐标系下的目标位置。
    tool_rpy_tool 是绕工具自身坐标系的相对姿态命令。
    """
    platform_pos, platform_rpy = platform_pose_from_tool_pose(
        tool_pos_world,
        tool_rpy_tool,
    )

    if not return_info:
        return is_pose_reachable(platform_pos, platform_rpy)

    reachable, info = is_pose_reachable(platform_pos, platform_rpy, return_info=True)
    info["platform_pos"] = platform_pos
    info["platform_rpy"] = platform_rpy
    return reachable, info
