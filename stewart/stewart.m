function stewart_gui()
    % Stewart 平台交互式 GUI (正逆运动学双向控制)
    clear; clc; close all;

    %% 1. 平台几何参数定义
    R_B = 200; % 下平台（静平台）半径 (mm)
    R_P = 160; % 上平台（动平台）半径 (mm)
    gamma_B = 15 * pi/180; % 下平台成对铰链偏角
    gamma_P = 15 * pi/180; % 上平台成对铰链偏角
    % 15、30、165、195、-75、-45
    theta_B = [1/3*pi - gama_B, 1/3*pi + gamma_B, pi - gamma_B, pi + gamma_B, -1/3*pi - gamma_B, -1/3*pi + gamma_B];

    theta_P = [gamma_P, 2/3*pi - gamma_P, 2/3*pi + gamma_P, 4/3*pi - gamma_P, 4/3*pi + gamma_P, -gamma_P];
    
    B = [R_B*cos(theta_B); R_B*sin(theta_B); zeros(1,6)];  

    P_local = [R_P*cos(theta_P); R_P*sin(theta_P); zeros(1,6)]; 
    
    %% 2. 初始化状态变量
    current_pos = [0; 0; 650];
    current_rpy = [0; 0; 0];
    [current_L, current_T] = calc_IK(current_pos, current_rpy, P_local, B);

    %% 3. 创建 UI 界面
    fig = uifigure('Name', 'Stewart Platform Interactive GUI', 'Position', [100, 100, 1100, 650]);
    
    % 创建 3D 绘图区
    ax = uiaxes(fig, 'Position', [10, 50, 600, 550]);
    view(ax, 3); grid(ax, 'on'); axis(ax, 'equal');
    xlabel(ax, 'X (mm)'); ylabel(ax, 'Y (mm)'); zlabel(ax, 'Z (mm)');
    % 固定视角范围避免抖动
    xlim(ax, [-300 300]); ylim(ax, [-300 300]); zlim(ax, [0 800]);

    % 创建控制面板
    pnl_pose = uipanel(fig, 'Title', '目标位姿输入 (触发逆解)', 'Position', [630, 420, 440, 180]);
    pnl_legs = uipanel(fig, 'Title', '电缸长度调节 (触发正解)', 'Position', [630, 50, 440, 350]);

    % --- 3.1 位姿输入控件 ---
    labels = {'X (mm)', 'Y (mm)', 'Z (mm)', 'Roll (deg)', 'Pitch (deg)', 'Yaw (deg)'};
    pose_edit = cell(1,6);
    for i = 1:6
        row = floor((i-1)/3); col = mod(i-1,3);
        uilabel(pnl_pose, 'Position', [20+col*140, 110-row*60, 70, 22], 'Text', labels{i});
        pose_edit{i} = uieditfield(pnl_pose, 'numeric', 'Position', [90+col*140, 110-row*60, 50, 22]);
        
        % 初始化数值
        if i <= 3
            pose_edit{i}.Value = current_pos(i);
        else
            pose_edit{i}.Value = current_rpy(i-3) * 180/pi; % 弧度转角度
        end
        % 绑定回调事件
        pose_edit{i}.ValueChangedFcn = @(src, event) update_from_pose();
    end

    % --- 3.2 缸长滑块控件 ---
    leg_sliders = cell(1,6);
    leg_labels = cell(1,6);
    for i = 1:6
        y_pos = 290 - (i-1)*50;
        uilabel(pnl_legs, 'Position', [20, y_pos, 50, 22], 'Text', sprintf('缸 %d', i));
        
        % 拖动条限制在 100 到 350 mm 之间
        leg_sliders{i} = uislider(pnl_legs, 'Position', [70, y_pos+10, 280, 3]);
        leg_sliders{i}.Limits = [565, 765]; 
        leg_sliders{i}.Value = current_L(i);
        
        % 数值显示标签
        leg_labels{i} = uilabel(pnl_legs, 'Position', [365, y_pos, 60, 22], 'Text', num2str(current_L(i), '%.3f'));

        % 绑定回调：拖动结束触发正解，拖动过程中仅更新标签数字
        leg_sliders{i}.ValueChangedFcn = @(src, event) update_from_legs();
        leg_sliders{i}.ValueChangingFcn = @(src, event) update_slider_label(i, event.Value);
    end

    % 首次绘制
    draw_platform(ax, B, current_T, current_pos, current_rpy);


    %% ================= 内部回调函数 =================
    
    % 【逆解回调】：读取输入框位姿 -> 算缸长 -> 更新滑块与 3D 图
    function update_from_pose()
        % 读取输入框 (转换为弧度)
        pos = [pose_edit{1}.Value; pose_edit{2}.Value; pose_edit{3}.Value];
        rpy = [pose_edit{4}.Value; pose_edit{5}.Value; pose_edit{6}.Value] * pi/180;
        
        [L, T] = calc_IK(pos, rpy, P_local, B);
        
        % 更新全局状态
        current_pos = pos; current_rpy = rpy; 
        current_L = L; current_T = T;
        
        % 更新滑块和文本
        for k = 1:6
            leg_sliders{k}.Value = min(max(L(k), leg_sliders{k}.Limits(1)), leg_sliders{k}.Limits(2));
            leg_labels{k}.Text = num2str(L(k), '%.3f');
        end
        
        draw_platform(ax, B, T, pos, rpy);
    end

    % 【正解回调】：读取滑块缸长 -> 算位姿 -> 更新输入框与 3D 图
    function update_from_legs()
        L_target = zeros(1,6);
        for k = 1:6
            L_target(k) = leg_sliders{k}.Value;
        end
        
        % 使用当前位姿作为初始猜测，提高 fsolve 求解速度
        guess = [current_pos; current_rpy];
        [pos, rpy] = calc_FK(L_target, guess, P_local, B);
        
        % 再做一次逆解取得动平台的精准三维坐标
        [~, T] = calc_IK(pos, rpy, P_local, B);
        
        % 更新全局状态
        current_pos = pos; current_rpy = rpy; 
        current_T = T;
        
        % 更新上方输入框 (转换为角度)
        for k = 1:3
            pose_edit{k}.Value = pos(k);
            pose_edit{k+3}.Value = round(rpy(k) * 180/pi, 2);
        end
        
        draw_platform(ax, B, T, pos, rpy);
    end

    % 滑块实时拖动仅更新数字标签
    function update_slider_label(idx, val)
        leg_labels{idx}.Text = num2str(val, '%.1f');
    end

end

%% ================= 核心算法与绘图子函数 =================

function [L, T] = calc_IK(pos, rpy, P_local, B)
    R = eul2rot(rpy);
    T = repmat(pos, 1, 6) + R * P_local;
    L = sqrt(sum((T - B).^2, 1));
end

function [pos, rpy] = calc_FK(L_target, guess, P_local, B)
    options = optimoptions('fsolve', 'Display', 'off', 'Algorithm', 'levenberg-marquardt');
    sol = fsolve(@(x) obj_func(x, L_target, P_local, B), guess, options);
    pos = sol(1:3);
    rpy = sol(4:6);
end

function err = obj_func(x, L_target, P_local, B)
    pos = x(1:3); rpy = x(4:6);
    [L_calc, ~] = calc_IK(pos, rpy, P_local, B);
    err = L_calc - L_target; 
end

function R = eul2rot(rpy)
    rx = rpy(1); ry = rpy(2); rz = rpy(3);
    Rx = [1 0 0; 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)];
    Ry = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)];
    Rz = [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1];
    R = Rz * Ry * Rx; 
end

function draw_platform(ax, B, T, pos, rpy)
    cla(ax); % 清除重绘
    hold(ax, 'on');
    
    B_poly = [B, B(:,1)];
    T_poly = [T, T(:,1)];
    
    % 绘制基座与动平台
    fill3(ax, B_poly(1,:), B_poly(2,:), B_poly(3,:), [0.8 0.8 0.8], 'FaceAlpha', 0.5);
    plot3(ax, B_poly(1,:), B_poly(2,:), B_poly(3,:), 'k-', 'LineWidth', 2);
    
    fill3(ax, T_poly(1,:), T_poly(2,:), T_poly(3,:), [0.4 0.6 0.9], 'FaceAlpha', 0.6);
    plot3(ax, T_poly(1,:), T_poly(2,:), T_poly(3,:), 'b-', 'LineWidth', 2);
    
    % 绘制缸体
    for i = 1:6
        plot3(ax, [B(1,i), T(1,i)], [B(2,i), T(2,i)], [B(3,i), T(3,i)], 'r-', 'LineWidth', 2);
        plot3(ax, B(1,i), B(2,i), B(3,i), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 5);
        plot3(ax, T(1,i), T(2,i), T(3,i), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 5);

        leg_len = norm(T(:,i) - B(:,i));
        label_pos = (B(:,i) + T(:,i)) / 2;
        offset_dir = label_pos - pos;
        if norm(offset_dir(1:2)) < 1e-6
            offset_dir = [cos(2*pi*(i-1)/6); sin(2*pi*(i-1)/6); 0];
        else
            offset_dir = [offset_dir(1:2); 0] / norm(offset_dir(1:2));
        end
        label_pos = label_pos + 18 * offset_dir;

        text(ax, label_pos(1), label_pos(2), label_pos(3), ...
            sprintf('L%d=%.1f mm', i, leg_len), ...
            'FontSize', 10, ...
            'Color', [0.45 0 0], ...
            'BackgroundColor', 'w', ...
            'EdgeColor', [0.85 0.85 0.85], ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle');
    end
    
    % 绘制坐标系
    axis_len = 80;
    quiver3(ax, 0, 0, 0, axis_len, 0, 0, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    quiver3(ax, 0, 0, 0, 0, axis_len, 0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    quiver3(ax, 0, 0, 0, 0, 0, axis_len, 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    
    R = eul2rot(rpy);
    X_P = R(:, 1) * axis_len; Y_P = R(:, 2) * axis_len; Z_P = R(:, 3) * axis_len;
    quiver3(ax, pos(1), pos(2), pos(3), X_P(1), X_P(2), X_P(3), 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    quiver3(ax, pos(1), pos(2), pos(3), Y_P(1), Y_P(2), Y_P(3), 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    quiver3(ax, pos(1), pos(2), pos(3), Z_P(1), Z_P(2), Z_P(3), 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    
    hold(ax, 'off');
end
