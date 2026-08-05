% Stewart平台参数辨识方法对比：传统OLS vs 作者提出的小波变换法
clear; clc;

%% 1. 系统真实参数设定 (你要去"猜"的未知数)
M_true = 15000;      % 真实等效质量 (对应 Mt)
B_true = 4000;       % 真实等效粘性摩擦 (对应 Bt)
Fc_true = 3000;      % 库仑摩擦力幅值 (模拟不可见的干扰)
G_true = 5000;       % 重力等效常数干扰

%% 2. 构造仿真激励信号 (恒定方向空间)
% 模拟机器人在一段时间 [0, 2s] 内单向加速并减速，速度方向不变
fs = 1000;           % 采样频率 1000Hz
t = (0:1/fs:2)';     % 时间向量

% 设计一个平滑的运动轨迹 
q = sin(pi/2 * t);               % 位置
dq = (pi/2) * cos(pi/2 * t);     % 速度 (在 t 属于 [0, 1) 时为正，满足恒定方向)
ddq = -(pi/2)^2 * sin(pi/2 * t); % 加速度

% 截取速度恒为正的时间段 (创造恒定方向空间)
idx = find(dq > 0.1); 
t_win = t(idx);
dq_win = dq(idx);
ddq_win = ddq(idx);

%% 3. 计算驱动力 (加入库仑摩擦和重力污染)
Constant_Disturbance = G_true + Fc_true; 
fa_win = M_true * ddq_win + B_true * dq_win + Constant_Disturbance;

% 加入少量高斯白噪声模拟传感器真实误差
fa_win = fa_win + 50 * randn(size(fa_win)); 

%% 4. 对比实验 A：传统最小二乘法 (受库仑摩擦常数严重污染)
Phi_A = [ddq_win, dq_win];
Y_A = fa_win;
Theta_A = pinv(Phi_A) * Y_A; % 伪逆求解

M_A = Theta_A(1);
B_A = Theta_A(2);

%% 5. 对比实验 B：小波变换辨识法 (作者的方法)
scale_a = 8; % 小波尺度

% 【修复点1】不依赖工具箱，直接用数学公式手动生成墨西哥帽(mexh)小波
t_wavelet = linspace(-5, 5, 200)'; % 小波的时域支撑区间
norm_constant = 2 / (sqrt(3) * pi^0.25); % 归一化常数
psi = norm_constant * (1 - t_wavelet.^2) .* exp(-t_wavelet.^2 / 2);

% 尺度缩放
psi_scaled = psi / sqrt(scale_a); 

% 执行卷积计算 (等效于小波变换内积)
W_ddq = conv(ddq_win, psi_scaled, 'same');
W_dq  = conv(dq_win, psi_scaled, 'same');
W_fa  = conv(fa_win, psi_scaled, 'same');

% 【修复点2】消除边界效应：截掉首尾两端由于信号突然中断产生的畸变数据
trim_len = 150; 
W_ddq_trim = W_ddq(trim_len : end-trim_len);
W_dq_trim  = W_dq(trim_len : end-trim_len);
W_fa_trim  = W_fa(trim_len : end-trim_len);

% 构建小波域方程并求解 (只用中间干净的数据)
Phi_B = [W_ddq_trim, W_dq_trim];
Y_B = W_fa_trim;
Theta_B = pinv(Phi_B) * Y_B;

M_B = Theta_B(1);
B_B = Theta_B(2);

%% 6. 结果展示
fprintf('=========== 辨识结果对比 ===========\n');
fprintf('真实质量: %.1f\t 真实粘性摩擦: %.1f\n', M_true, B_true);
fprintf('------------------------------------\n');
fprintf('传统方法 (受库仑摩擦严重干扰):\n');
fprintf('辨识质量: %.1f \t(误差: %.2f%%)\n', M_A, abs(M_A-M_true)/M_true*100);
fprintf('辨识摩擦: %.1f \t(误差: %.2f%%)\n', B_A, abs(B_A-B_true)/B_true*100);
fprintf('------------------------------------\n');
fprintf('小波变换法 (论文作者方法):\n');
fprintf('辨识质量: %.1f \t(误差: %.2f%%)\n', M_B, abs(M_B-M_true)/M_true*100);
fprintf('辨识摩擦: %.1f \t(误差: %.2f%%)\n', B_B, abs(B_B-B_true)/B_true*100);
fprintf('====================================\n');