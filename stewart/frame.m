%% 3. 坐标系与机构朝向动画显示
figure('Color', 'w', 'Name', 'Stewart 平台坐标系朝向动画');
hold on; grid on; axis equal;
view(3); % 三维视角
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
xlim([-250 250]); ylim([-250 250]); zlim([0 220]);

% --- A. 绘制静平台(基座)坐标轴 ---
% 红色=X轴, 绿色=Y轴, 蓝色=Z轴
quiver3(0,0,0, 60,0,0, 'r', 'LineWidth', 2.5, 'MaxHeadSize', 0.5); text(70,0,0, '+X_B', 'Color', 'r', 'FontWeight', 'bold');
quiver3(0,0,0, 0,60,0, 'g', 'LineWidth', 2.5, 'MaxHeadSize', 0.5); text(0,70,0, '+Y_B', 'Color', 'g', 'FontWeight', 'bold');
quiver3(0,0,0, 0,0,60, 'b', 'LineWidth', 2.5, 'MaxHeadSize', 0.5); text(0,0,70, '+Z_B', 'Color', 'b', 'FontWeight', 'bold');

% 绘制静平台铰链圆环及球铰点 (蓝色)
plot3([B(1,:) B(1,1)], [B(2,:) B(2,1)], [B(3,:) B(3,1)], 'b--', 'LineWidth', 1);
scatter3(B(1,:), B(2,:), B(3,:), 50, 'b', 'filled');
for i = 1:6
    text(B(1,i), B(2,i), B(3,i)-10, ['B' num2str(i)], 'Color', 'b');
end

% --- B. 计算并绘制动平台 ---
[~, T] = calc_IK(target_pos, target_rpy, P_local, B);

% 绘制动平台铰链圆环及球铰点 (品红色)
plot3([T(1,:) T(1,1)], [T(2,:) T(2,1)], [T(3,:) T(3,1)], 'm-', 'LineWidth', 1.5);
scatter3(T(1,:), T(2,:), T(3,:), 50, 'm', 'filled');
for i = 1:6
    text(T(1,i), T(2,i), T(3,i)+10, ['P' num2str(i)], 'Color', 'm');
end

% --- C. 绘制动平台随动坐标轴 ---
R = eul2rot(target_rpy);
% 动平台中心点
ox = target_pos(1); oy = target_pos(2); oz = target_pos(3); 
quiver3(ox, oy, oz, R(1,1)*50, R(2,1)*50, R(3,1)*50, 'r', 'LineWidth', 2.5); text(ox+R(1,1)*60, oy+R(2,1)*60, oz+R(3,1)*60, '+X_P', 'Color', 'r');
quiver3(ox, oy, oz, R(1,2)*50, R(2,2)*50, R(3,2)*50, 'g', 'LineWidth', 2.5); text(ox+R(1,2)*60, oy+R(2,2)*60, oz+R(3,2)*60, '+Y_P', 'Color', 'g');
quiver3(ox, oy, oz, R(1,3)*50, R(2,3)*50, R(3,3)*50, 'b', 'LineWidth', 2.5); text(ox+R(1,3)*60, oy+R(2,3)*60, oz+R(3,3)*60, '+Z_P', 'Color', 'b');

% --- D. 绘制 6 根驱动连杆 ---
h_legs = zeros(1, 6);
for i = 1:6
    h_legs(i) = plot3([B(1,i) T(1,i)], [B(2,i) T(2,i)], [B(3,i) T(3,i)], 'k', 'LineWidth', 2);
end

% --- E. 循环旋转视角动画 ---
title('Stewart 平台空间方位 (360° 旋转观察中...)');
for az = 0:2:360
    view(az, 25); % 改变水平视角，仰角固定为 25 度
    drawnow;
    pause(0.02);  % 控制动画帧率
end
title('动画播放完毕（可手动旋转图窗观察）');