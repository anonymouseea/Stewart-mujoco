%% test_adaptive_admittance_vs_traditional.m
% 目的：
%   只测试控制方法，不使用 Adams，不使用 Stewart 运动学。
%   比较：
%       1) 传统导纳控制
%       2) 复合误差自适应导纳控制
%
% 仿真对象：
%   简化为 2 个自由度：
%       y 方向：对接横向位置误差
%       z 方向：接触力控制方向
%
%   z 方向环境接触模型：
%       Fz = Ke * penetration + Be * penetration_dot
%
% 说明：
%   这不是完整 Stewart 平台动力学仿真，而是控制律验证仿真。
%   后续你可以把这里的 z / y 控制扩展成 6 维，再接 Stewart 逆运动学。

clear; clc; close all;

%% ===================== 仿真参数 =====================
dt = 0.001;
T  = 8.0;
t  = 0:dt:T;
N  = length(t);

% 自由度定义
IDX_Y = 1;
IDX_Z = 2;
n = 2;

% 初始位置
y0 = 0.020;     % 横向初始误差 20 mm
z0 = 0.550;     % 对接端初始 z 位置

% 环境位置，也就是锁紧机构接触面
z_env0 = 0.620;

% 名义前进轨迹
v_push = 0.040;             % m/s
z_goal = z_env0 + 0.010;    % 如果无控制，会压入 10 mm

% 期望接触力
Fd = 30;                    % N

% 环境刚度和阻尼
% 这里故意设置成未知环境：4.5 s 后刚度突变，用来测试自适应性
Ke1 = 15000;                % N/m
Ke2 = 23000;                % N/m
t_Ke_jump = 4.5;
Be  = 120;                  % N/(m/s)

% 位置内环闭环动力学参数
% 用二阶系统模拟电缸位置内环：x_dd = wn^2(x_cmd-x)-2*zeta*wn*x_dot
wn   = [35, 55];            % rad/s
zeta = [0.85, 0.75];

% 接触反力对 z 方向的等效扰动质量
mz = 8.0;                   % kg

%% ===================== 传统导纳控制参数 =====================
trad.Md = [1.0, 1.0];
trad.Bd = [22, 55];
trad.Kd = [120, 1500];

% y 方向：用位置误差修正
% z 方向：用接触力误差修正
trad.Kp = [500, 0];
trad.Kf = [0, 0.12];

% 传统导纳不加积分，不加自适应
trad.delta_limit = [0.040, 0.050];
trad.vel_limit   = [0.080, 0.080];

%% ===================== 自适应导纳控制参数 =====================
adp.Md = [1.0, 1.0];
adp.Bd = [22, 55];
adp.Kd = [120, 1500];

% 基础项
adp.Kp = [500, 0];
adp.Kf = [0, 0.12];
adp.Ki = [80, 0.0025];

% 复合误差 s = ef_dot + Lambda_f*ef + Lambda_p*ep + Lambda_i*eta
adp.Lambda_f = [0, 18];
adp.Lambda_p = [12, 0];
adp.Lambda_i = [4, 0.8];

% eta_dot = ef + rho_p * ep
adp.rho_p = [1.0, 0.0];

% 快速项
adp.Ks  = [0.0015, 0.0020];
adp.phi = [0.004, 12];

% 自适应参数初值
adp.p_hat = zeros(1,n);
adp.d_hat = zeros(1,n);
adp.r_hat = zeros(1,n);
adp.h_hat = zeros(1,n);

% 自适应增益
% 注意：z 方向力误差单位是 N，ef_dot 可能很大，所以 gamma 不能太大
adp.gamma_p = [0, 2e-9];
adp.gamma_d = [0, 2e-12];
adp.gamma_r = [2e-2, 0];
adp.gamma_h = [5e-3, 5e-8];

% 泄漏项，防止参数漂移
adp.leak_p = [0.02, 0.02];
adp.leak_d = [0.02, 0.02];
adp.leak_r = [0.02, 0.02];
adp.leak_h = [0.02, 0.02];

% 自适应参数限幅
adp.p_limit = [0, 1.2e-4];
adp.d_limit = [0, 2.0e-5];
adp.r_limit = [0.6, 0];
adp.h_limit = [0.2, 0.002];

% 状态限幅
adp.delta_limit = [0.040, 0.050];
adp.vel_limit   = [0.080, 0.080];
adp.eta_limit   = [0.050, 300];

% s 限幅，仅用于参数更新，避免接触瞬间 ef_dot 太大导致参数爆炸
adp.s_adapt_limit = [2, 800];

% 力误差微分低通滤波
adp.force_diff_alpha = 0.90;

%% ===================== 运行两组仿真 =====================
res_trad = run_simulation("traditional", t, dt, ...
    y0, z0, z_env0, z_goal, v_push, Fd, ...
    Ke1, Ke2, t_Ke_jump, Be, wn, zeta, mz, trad, adp);

res_adp = run_simulation("adaptive", t, dt, ...
    y0, z0, z_env0, z_goal, v_push, Fd, ...
    Ke1, Ke2, t_Ke_jump, Be, wn, zeta, mz, trad, adp);

%% ===================== 性能指标 =====================
metric_trad = calc_metrics(res_trad, Fd);
metric_adp  = calc_metrics(res_adp, Fd);

fprintf('\n================ 仿真结果对比 ================\n');
fprintf('传统导纳：\n');
fprintf('  最大接触力峰值       = %.2f N\n', metric_trad.peak_force);
fprintf('  稳态接触力平均误差   = %.2f N\n', metric_trad.force_ss_error);
fprintf('  最终横向位置误差     = %.3f mm\n', metric_trad.final_y_error_mm);
fprintf('  最大 z 向速度        = %.3f m/s\n', metric_trad.max_abs_vz);

fprintf('\n自适应导纳：\n');
fprintf('  最大接触力峰值       = %.2f N\n', metric_adp.peak_force);
fprintf('  稳态接触力平均误差   = %.2f N\n', metric_adp.force_ss_error);
fprintf('  最终横向位置误差     = %.3f mm\n', metric_adp.final_y_error_mm);
fprintf('  最大 z 向速度        = %.3f m/s\n', metric_adp.max_abs_vz);

fprintf('\n改善比例：\n');
fprintf('  接触力峰值降低       = %.1f %%\n', ...
    100 * (metric_trad.peak_force - metric_adp.peak_force) / metric_trad.peak_force);
fprintf('  稳态力误差降低       = %.1f %%\n', ...
    100 * (metric_trad.force_ss_error - metric_adp.force_ss_error) / metric_trad.force_ss_error);
fprintf('  横向位置误差降低     = %.1f %%\n', ...
    100 * (metric_trad.final_y_error_mm - metric_adp.final_y_error_mm) / metric_trad.final_y_error_mm);

%% ===================== 画图 =====================
figure('Name','Contact force');
plot(t, res_trad.Fz, 'LineWidth', 1.4); hold on;
plot(t, res_adp.Fz, 'LineWidth', 1.4);
yline(Fd, '--', 'LineWidth', 1.2);
xline(t_Ke_jump, ':', 'Ke jump', 'LineWidth', 1.2);
grid on;
xlabel('Time / s');
ylabel('Contact force F_z / N');
legend('Traditional admittance','Adaptive admittance','Desired force','Location','best');
title('接触力对比');

figure('Name','Z displacement');
plot(t, res_trad.z, 'LineWidth', 1.4); hold on;
plot(t, res_adp.z, 'LineWidth', 1.4);
plot(t, res_trad.z_ff, 'k--', 'LineWidth', 1.1);
yline(z_env0, ':', 'Environment position', 'LineWidth', 1.2);
grid on;
xlabel('Time / s');
ylabel('z position / m');
legend('Traditional','Adaptive','Feedforward','Environment','Location','best');
title('z 向位移对比');

figure('Name','Z velocity');
plot(t, res_trad.vz, 'LineWidth', 1.4); hold on;
plot(t, res_adp.vz, 'LineWidth', 1.4);
grid on;
xlabel('Time / s');
ylabel('z velocity / m/s');
legend('Traditional','Adaptive','Location','best');
title('z 向速度对比');

figure('Name','Lateral position error');
plot(t, res_trad.y * 1000, 'LineWidth', 1.4); hold on;
plot(t, res_adp.y * 1000, 'LineWidth', 1.4);
grid on;
xlabel('Time / s');
ylabel('y error / mm');
legend('Traditional','Adaptive','Location','best');
title('横向对接位置误差对比');

figure('Name','Command compensation');
plot(t, res_trad.delta_z * 1000, 'LineWidth', 1.4); hold on;
plot(t, res_adp.delta_z * 1000, 'LineWidth', 1.4);
grid on;
xlabel('Time / s');
ylabel('\Delta z / mm');
legend('Traditional basic compensation','Adaptive total compensation','Location','best');
title('z 向补偿量对比');

figure('Name','Adaptive parameters');
subplot(2,2,1);
plot(t, res_adp.p_hat_z, 'LineWidth', 1.2); grid on;
xlabel('Time / s'); ylabel('p\_hat\_z');

subplot(2,2,2);
plot(t, res_adp.d_hat_z, 'LineWidth', 1.2); grid on;
xlabel('Time / s'); ylabel('d\_hat\_z');

subplot(2,2,3);
plot(t, res_adp.r_hat_y, 'LineWidth', 1.2); grid on;
xlabel('Time / s'); ylabel('r\_hat\_y');

subplot(2,2,4);
plot(t, res_adp.h_hat_y, 'LineWidth', 1.2); grid on;
xlabel('Time / s'); ylabel('h\_hat\_y');

%% ============================================================
%%                     本脚本的局部函数
%% ============================================================

function res = run_simulation(mode, t, dt, ...
    y0, z0, z_env0, z_goal, v_push, Fd, ...
    Ke1, Ke2, t_Ke_jump, Be, wn, zeta, mz, trad, adp)

    N = length(t);
    n = 2;

    IDX_Y = 1;
    IDX_Z = 2;

    % 实际位置和速度
    x = [y0, z0];
    v = [0, 0];

    % 导纳基础补偿状态
    delta = zeros(1,n);
    delta_dot = zeros(1,n);

    % 自适应控制状态
    eta = zeros(1,n);
    ef_last = zeros(1,n);
    ef_dot_filt = zeros(1,n);

    p_hat = adp.p_hat;
    d_hat = adp.d_hat;
    r_hat = adp.r_hat;
    h_hat = adp.h_hat;

    % 数据记录
    res.t = t;
    res.y = zeros(1,N);
    res.z = zeros(1,N);
    res.vy = zeros(1,N);
    res.vz = zeros(1,N);
    res.Fz = zeros(1,N);
    res.z_ff = zeros(1,N);
    res.delta_z = zeros(1,N);
    res.p_hat_z = zeros(1,N);
    res.d_hat_z = zeros(1,N);
    res.r_hat_y = zeros(1,N);
    res.h_hat_y = zeros(1,N);

    for k = 1:N
        tk = t(k);

        % 名义前进轨迹
        z_ff = min(z0 + v_push * tk, z_goal);
        base_cmd = [y0, z_ff];

        % 环境刚度突变
        if tk < t_Ke_jump
            Ke = Ke1;
        else
            Ke = Ke2;
        end

        % 接触力模型
        penetration = x(IDX_Z) - z_env0;
        if penetration > 0
            Fz = Ke * penetration + Be * max(v(IDX_Z), 0);
            Fz = max(Fz, 0);
        else
            Fz = 0;
        end

        contact = Fz > 0.5 || x(IDX_Z) > z_env0;

        % 误差定义
        % y 方向：位置误差，目标是 y = 0
        ep = [x(IDX_Y), 0];

        % z 方向：力误差，未接触前不引入力误差
        if contact
            ef = [0, Fz - Fd];
        else
            ef = [0, 0];
        end

        switch mode
            case "traditional"
                % 传统导纳：
                % Md * delta_ddot + Bd * delta_dot + Kd * delta
                %     = -Kp*ep - Kf*ef
                rhs = -trad.Kp .* ep - trad.Kf .* ef;

                delta_ddot = (rhs - trad.Bd .* delta_dot - trad.Kd .* delta) ./ trad.Md;

                delta_dot = delta_dot + delta_ddot * dt;
                delta_dot = vec_clip(delta_dot, trad.vel_limit);

                delta = delta + delta_dot * dt;
                delta = vec_clip(delta, trad.delta_limit);

                x_cmd = base_cmd + delta;

                % 用于画图
                total_delta = delta;

            case "adaptive"
                % 力误差微分
                ef_dot_raw = (ef - ef_last) / dt;
                ef_last = ef;

                alpha = adp.force_diff_alpha;
                ef_dot_filt = alpha * ef_dot_filt + (1 - alpha) * ef_dot_raw;
                ef_dot = ef_dot_filt;

                % 积分误差
                % y 方向积分位置误差；z 方向积分力误差
                eta_dot = ef + adp.rho_p .* ep;

                % 未接触前不积累 z 向力误差，避免提前发散
                if ~contact
                    eta_dot(IDX_Z) = 0;
                end

                eta = eta + eta_dot * dt;
                eta = vec_clip(eta, adp.eta_limit);

                % 复合误差
                s = ef_dot + adp.Lambda_f .* ef + ...
                    adp.Lambda_p .* ep + adp.Lambda_i .* eta;

                % 基础导纳
                rhs = -adp.Kp .* ep - adp.Kf .* ef - adp.Ki .* eta;

                delta_ddot = (rhs - adp.Bd .* delta_dot - adp.Kd .* delta) ./ adp.Md;

                delta_dot = delta_dot + delta_ddot * dt;
                delta_dot = vec_clip(delta_dot, adp.vel_limit);

                delta = delta + delta_dot * dt;
                delta = vec_clip(delta, adp.delta_limit);

                % 参数更新用限幅后的 s
                s_adapt = vec_clip(s, adp.s_adapt_limit);

                if contact
                    p_hat = p_hat + dt * (adp.gamma_p .* s_adapt .* ef ...
                        - adp.leak_p .* p_hat);

                    d_hat = d_hat + dt * (adp.gamma_d .* s_adapt .* ef_dot ...
                        - adp.leak_d .* d_hat);
                else
                    % 未接触前不更新力相关参数
                    p_hat = p_hat + dt * (-adp.leak_p .* p_hat);
                    d_hat = d_hat + dt * (-adp.leak_d .* d_hat);
                end

                % 位置误差相关参数可以一直更新
                r_hat = r_hat + dt * (adp.gamma_r .* s_adapt .* ep ...
                    - adp.leak_r .* r_hat);

                h_hat = h_hat + dt * (adp.gamma_h .* s_adapt .* eta ...
                    - adp.leak_h .* h_hat);

                % 参数投影限幅
                p_hat = vec_clip(p_hat, adp.p_limit);
                d_hat = vec_clip(d_hat, adp.d_limit);
                r_hat = vec_clip(r_hat, adp.r_limit);
                h_hat = vec_clip(h_hat, adp.h_limit);

                % 自适应补偿项
                adaptive_term = p_hat .* ef + ...
                                d_hat .* ef_dot + ...
                                r_hat .* ep + ...
                                h_hat .* eta + ...
                                adp.Ks .* sat_vec(s ./ adp.phi);

                delta_adaptive = -adaptive_term;

                % 最终命令
                x_cmd = base_cmd + delta + delta_adaptive;

                % 限制总补偿，防止发散
                total_delta = x_cmd - base_cmd;
                total_delta = vec_clip(total_delta, [0.050, 0.060]);
                x_cmd = base_cmd + total_delta;

            otherwise
                error("Unknown mode");
        end

        % 位置内环闭环动力学
        acc = wn.^2 .* (x_cmd - x) - 2 .* zeta .* wn .* v;

        % 接触反力对 z 方向的影响
        % Fz 会把对接端往回顶
        acc(IDX_Z) = acc(IDX_Z) - Fz / mz;

        % 积分实际系统
        v = v + acc * dt;
        x = x + v * dt;

        % 记录数据
        res.y(k) = x(IDX_Y);
        res.z(k) = x(IDX_Z);
        res.vy(k) = v(IDX_Y);
        res.vz(k) = v(IDX_Z);
        res.Fz(k) = Fz;
        res.z_ff(k) = z_ff;
        res.delta_z(k) = total_delta(IDX_Z);

        if mode == "adaptive"
            res.p_hat_z(k) = p_hat(IDX_Z);
            res.d_hat_z(k) = d_hat(IDX_Z);
            res.r_hat_y(k) = r_hat(IDX_Y);
            res.h_hat_y(k) = h_hat(IDX_Y);
        end
    end
end

function y = vec_clip(x, limit)
    y = min(max(x, -limit), limit);
end

function y = sat_vec(x)
    y = min(max(x, -1), 1);
end

function m = calc_metrics(res, Fd)
    N = length(res.t);
    last_idx = round(0.85 * N):N;

    m.peak_force = max(res.Fz);
    m.force_ss_error = mean(abs(res.Fz(last_idx) - Fd));
    m.final_y_error_mm = abs(res.y(end)) * 1000;
    m.max_abs_vz = max(abs(res.vz));
end