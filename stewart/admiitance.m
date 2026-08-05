function [x_next, dx_next, ddx] = admiitance(x, dx, x_d, dx_d, ddx_d, Fext, Md, Dd, Kd, dt)
%ADMITTANCE_UPDATE 六维导纳/阻抗模型单步更新
%
% 阻抗/导纳模型：
% Md*(ddx - ddx_d) + Dd*(dx - dx_d) + Kd*(x - x_d) = Fext
%
% 整理：
% ddx = ddx_d + inv(Md)*(Fext - Dd*(dx-dx_d) - Kd*(x-x_d))
%
% 输入：
% x      : 当前实际位姿          6×1 [m; m; m; rad; rad; rad]
% dx     : 当前实际速度          6×1
% x_d    : 当前期望位姿          6×1
% dx_d   : 当前期望速度          6×1
% ddx_d  : 当前期望加速度        6×1
% Fext   : 当前外力/外力矩       6×1 [N; N; N; N*m; N*m; N*m]
% Md     : 虚拟质量矩阵          6×6
% Dd     : 虚拟阻尼矩阵          6×6
% Kd     : 虚拟刚度矩阵          6×6
% dt     : 采样时间              s
%
% 输出：
% x_next  : 下一步位姿
% dx_next : 下一步速度
% ddx     : 当前加速度

    ddx = ddx_d + Md \ (Fext - Dd*(dx - dx_d) - Kd*(x - x_d));

    dx_next = dx + ddx * dt;

    x_next = x + dx_next * dt;

end