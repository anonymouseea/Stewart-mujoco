% 测试脚本：验证 Stewart 逆动力学
clear; clc;

% 1. 定义某个时间点的运动状态（例如平台正在做上下正弦往复运动）
t = 0.5; 
f = 1.0; % 频率 1Hz

% 位姿
z0 = 0.6; % 初始高度
x = [0; 0; z0 + 0.05*sin(2*pi*f*t)];
R = eye(3); % 无旋转

% 速度
dx = [0; 0; 0.05*2*pi*f*cos(2*pi*f*t)];
omega = [0; 0; 0];

% 加速度
ddx = [0; 0; -0.05*(2*pi*f)^2*sin(2*pi*f*t)];
domega = [0; 0; 0];

% 外部负载（例如外部切削力或触碰力，没有则设为0）
ext_wrench = [0; 0; 0; 0; 0; 0]; 

% 2. 调用动力学函数
F_leg = stewart_inverse_dynamics(x, dx, ddx, R, omega, domega, ext_wrench);

% 3. 打印输出结果
disp('当前时刻 6 根电动缸分别需要的推力 (N):');
for i = 1:6
    fprintf('第 %d 根杆: %10.2f N\n', i, F_leg(i));
end