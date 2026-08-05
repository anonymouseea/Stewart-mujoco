clear; clc;

%% 1. 平台
% 目标位姿上平台表面中心点
pos=[0,0,650,0,0,0];
% 平台目标速度
v = [0;0;0;0;0;0];

%% 2. 平台几何参数定义

R_P = 160; % 上平台（动平台）半径 (mm)
R_B = 200; % 下平台（静平台）半径 (mm)
gamma_B = 15 * pi/180; % 下平台成对铰链偏角
gamma_P = 15 * pi/180; % 上平台成对铰链偏角

% 固定长度549mm
fix_l=549;
% 45、75、165、195、-75、-45下平台是绕60°、180°、300°
theta_B = [1/3*pi - gamma_B, 1/3*pi + gamma_B, pi - gamma_B, pi + gamma_B, -1/3*pi - gamma_B, -1/3*pi + gamma_B];
 % 15、105°、135°、225°、255°、345°上平台是绕0°、120°、240°分布
theta_P = [gamma_P, 2/3*pi - gamma_P, 2/3*pi + gamma_P, 4/3*pi - gamma_P, 4/3*pi + gamma_P, -gamma_P];

p_z=[-18.5,-18.5,-18.5,-18.5,-18.5,-18.5];
b_z=[18.5,18.5,18.5,18.5,18.5,18.5];

% 下平台铰点坐标
B_local = [R_B*cos(theta_B); R_B*sin(theta_B); b_z];  
% 上平台铰点坐标
P_local = [R_P*cos(theta_P); R_P*sin(theta_P); p_z]; 


R=eul2rot(pos(4:6));
%% 3. 平台逆运动学
T = pos(1:3)'+ R * P_local;
% 支腿列向量
L = T - B_local;
% 支腿长度
l = vecnorm(L);
% 真实控制量
control_l=l-fix_l;

disp("支腿长度");
disp(l);

disp("支腿控制量");
disp(control_l);


%% 4.雅可比矩阵计算

J = zeros(6,6);
for i = 1:6
    % 支腿单位方向
    s_i = L(:,i)/l(i);
    disp(s_i);
    % 动平台铰点位置（转到基坐标系）
    p_i = R*P_local(:,i);
    % 雅可比第i行
    J(i,:) = [s_i', cross(p_i,s_i)'];
end

disp("Jacobian:");
disp(J);

% 六根腿速度
l_dot = J*v;
Lc = 160;
J_scaled = J;
J_scaled(:,4:6) = J(:,4:6)/Lc;
aa=cond(J_scaled);

disp("归一化条件数:");
disp(aa);

disp("支腿速度:");
disp(l_dot);

%% 5.静力学
%已知雅可比矩阵，根据虚功原理推导六条支腿的轴向力与上平台的广义力之间的关系

% 假设六根支腿轴向力，单位 N
f = [10; 10; 10; 10; 10; 10];

% 平台受到的广义力/力矩
F = J' * f;

F_Nm = F;
% 转换为Nm
F_Nm(4:6) = F_Nm(4:6) / 1000;

disp("平台广义力 F = [Fx; Fy; Fz; Mx; My; Mz]");
disp(F_Nm);


%% 6. 动平台简化动力学


m = 5.68;       % 动平台质量 kg
g = 9.81;     % 重力加速度 m/s^2

% 上平台惯性张量，单位 kg*mm^2
I_body = diag([0.064498, 0.128924, 0.064498]);

% 当前姿态下，惯量转到基坐标系
I_base = R * I_body * R';

% 平台速度
v = [0; 0; 0.01];          % m/s
omega = [0; 0; 0];      % rad/s

% 平台加速度
a = [0; 0; 0.1];          % m/s^2
alpha = [0; 0; 0];      % rad/s^2

% 质量惯量矩阵
Mx = [m*eye(3), zeros(3,3);
      zeros(3,3), I_base];

% 广义加速度
V_dot = [a; alpha];

% 速度相关项
Cx = [0; 0; 0;
      cross(omega, I_base*omega)];

% 如果质心在平台中心
Gx = [0; 0; m*g; 0; 0; 0];

% 外部接触力，没有就设为0
F_ext = zeros(6,1);

% 把J转成SI形式：后三列从mm变成m
J_SI = J;
J_SI(:,4:6) = J_SI(:,4:6) / 1000;

% 平台需要的广义驱动力
W_req = Mx*V_dot + Cx + Gx - F_ext;

% 反求六根支腿轴向力
f_leg = J_SI' \ W_req;

disp("六根支腿轴向力 N:");
disp(f_leg);
%%力控
Md = diag([5, 5, 5, 0.2, 0.2, 0.2]);
Dd = diag([500, 500, 500, 5, 5, 5]);
Kd = diag([0, 0, 0, 20, 20, 20]);
dt = 0.001;
T_total = 1;
t = 0:dt:T_total;
N = length(t);
x = zeros(6,N);
dx = zeros(6,N);
ddx = zeros(6,N);

x_d = zeros(6,N);
dx_d = zeros(6,N);
ddx_d = zeros(6,N);

Fext = zeros(6,N);

[x_next, dx_next, ddx] = admiitance(x, dx, x_d, dx_d, ddx_d, Fext, Md, Dd, Kd, dt);




%% 功能函数块
function R = eul2rot(rpy)
    rx = rpy(1); ry = rpy(2); rz = rpy(3);
    Rx = [1 0 0; 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)];
    Ry = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)];
    Rz = [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1];
    R = Rz * Ry * Rx; 
end
    