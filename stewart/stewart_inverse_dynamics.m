function F_leg = stewart_inverse_dynamics(x, dx, ddx, R, omega, domega, ext_wrench)
% 输入参数:
%   x          - 动平台质心位置 [x; y; z] (3x1)
%   dx         - 动平台线速度 [vx; vy; vz] (3x1)
%   ddx        - 动平台线加速度 [ax; ay; az] (3x1)
%   R          - 动平台当前旋转矩阵 (3x3)
%   omega      - 动平台角速度 (3x1)
%   domega     - 动平台角加速度 (3x1)
%   ext_wrench - 外部作用在质心的力和力矩 [F_ext; M_ext] (6x1)，默认为零
%
% 输出参数:
%   F_leg      - 6根驱动杆的轴向力 (6x1)

    if nargin < 7
        ext_wrench = zeros(6,1);
    end

    %% 1. 机构几何与物理参数定义 (根据实际平台修改)
    % 动/定平台质量及惯量
    m_p = 15.0;         % 动平台总质量 (kg)
    g = [0; 0; -9.81];  % 重力加速度向量 (m/s^2)
    
    % 动平台在自身坐标系下的转动惯量矩阵 (kg·m^2)
    I_p = diag([0.5, 0.5, 0.8]); 

    % 基座铰链中心在固定坐标系下的坐标 B_i (3x6)
    % 动平台铰链中心在平台坐标系下的坐标 P_bi (3x6)
    [B, P_b] = get_stewart_geometry();

    %% 2. 运动学解算 (求每根杆的方向向量)
    P = zeros(3, 6);     % 动平台铰链在固定坐标系下的坐标
    L = zeros(3, 6);     % 杆件向量
    l = zeros(1, 6);     % 杆件长度
    s = zeros(3, 6);     % 杆件单位方向向量

    for i = 1:6
        % 将动平台铰链点转换到固定坐标系
        P(:, i) = x + R * P_b(:, i);
        % 计算杆件向量
        L(:, i) = P(:, i) - B(:, i);
        % 杆长
        l(i) = norm(L(:, i));
        % 杆件单位向量 (由基座指向动平台)
        s(:, i) = L(:, i) / l(i);
    end

    %% 3. 构建机构的逆雅可比矩阵 (Inverse Jacobian Matrix)
    % 逆雅可比矩阵 J_I 建立了动平台速度与杆件拉伸速度的关系: d_l = J_I * V_platform
    J_I = zeros(6, 6);
    for i = 1:6
        r_i = P(:, i) - x; % 质心到动平台铰链点的向量
        % 每一行: [单位方向向量^T, (r_i 叉乘 单位方向向量)^T]
        J_I(i, :) = [s(:, i)', cross(r_i, s(:, i))'];
    end

    %% 4. 动平台动力学方程计算 (牛顿-欧拉公式)
    % 将转动惯量矩阵转换到固定坐标系
    I_G = R * I_p * R';

    % 计算动平台所受的惯性力和惯性力矩
    F_inertia = m_p * (ddx - g);
    M_inertia = I_G * domega + cross(omega, I_G * omega);

    % 合成动平台的总动力学非线性扳手 (Wrench)
    % 包含：惯性项 + 外部负载
    W_platform = [F_inertia; M_inertia] + ext_wrench;

    %% 5. 求解驱动杆件的轴向力
    % 根据静力/动力学对偶原理：W_platform = J_I^T * F_leg
    % 因此：F_leg = (J_I^T) \ W_platform
    
    % 检查雅可比矩阵是否奇异
    if cond(J_I) > 1e6
        warning('机构接近奇异状态！');
    end
    
    F_leg = (J_I') \ W_platform;

end

%% ================== 辅助函数：定义平台几何结构 ==================
function [B, P_b] = get_stewart_geometry()
    % 简易定义一个六足对称并联平台的铰链点位置 (单位: 米)
    % 实际项目中请替换为真实的物理坐标
    
    r_B = 0.5;   % 基座半径
    r_P = 0.3;   % 动平台半径
    
    % 分布角度 (采用典型的小角度-大角度交错分布)
    alpha_B = deg2rad([15, 105, 135, 225, 255, 345]);
    alpha_P = deg2rad([45, 75, 165, 195, 285, 315]);
    
    B = zeros(3, 6);
    P_b = zeros(3, 6);
    
    for i = 1:6
        B(:, i) = [r_B * cos(alpha_B(i)); r_B * sin(alpha_B(i)); 0];
        P_b(:, i) = [r_P * cos(alpha_P(i)); r_P * sin(alpha_P(i)); 0];
    end
end