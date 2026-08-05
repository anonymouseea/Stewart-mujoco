function workspace_analysis()
%WORKSPACE_ANALYSIS Reachable workspace analysis for stewart.m geometry.
% Length is the total actuator length: fixed length 565 mm + stroke 0..200 mm.
% Ball-joint limit is modeled as a +/-30 deg cone about base +Z and platform -Z.

    clear; clc; close all;

    %% Geometry and limits
    R_B = 200;              % base platform radius (mm)
    R_P = 160;              % moving platform radius (mm)
    gamma_B = deg2rad(15);  % paired base joint offset
    gamma_P = deg2rad(15);  % paired platform joint offset

    L_min = 565;            % mm
    L_max = 565 + 200;      % mm
    joint_limit = deg2rad(30);

    theta_B = [pi/3 - gamma_B, pi/3 + gamma_B, ...
               pi   - gamma_B, pi   + gamma_B, ...
              -pi/3 - gamma_B, -pi/3 + gamma_B];
    theta_P = [gamma_P, ...
               2*pi/3 - gamma_P, 2*pi/3 + gamma_P, ...
               4*pi/3 - gamma_P, 4*pi/3 + gamma_P, ...
              -gamma_P];

    B = [R_B*cos(theta_B); R_B*sin(theta_B); zeros(1,6)];
    P_local = [R_P*cos(theta_P); R_P*sin(theta_P); zeros(1,6)];

    %% Translational workspace at zero attitude
    xy_step = 5;   % mm, decrease for finer results
    z_step = 10;   % mm, only used for plotting the point cloud
    x_values = -350:xy_step:350;
    y_values = -350:xy_step:350;

    xyz = [];
    volume_mm3 = 0;
    z_low_map = nan(numel(y_values), numel(x_values));
    z_high_map = nan(numel(y_values), numel(x_values));

    for ix = 1:numel(x_values)
        x = x_values(ix);
        for iy = 1:numel(y_values)
            y = y_values(iy);
            [ok, z_low, z_high] = zero_attitude_z_range([x; y], B, P_local, ...
                L_min, L_max, joint_limit);

            if ok
                z_low_map(iy, ix) = z_low;
                z_high_map(iy, ix) = z_high;
                volume_mm3 = volume_mm3 + (z_high - z_low) * xy_step^2;
                z_values = z_low:z_step:z_high;
                xyz = [xyz; [x*ones(numel(z_values),1), ...
                             y*ones(numel(z_values),1), ...
                             z_values(:)]]; %#ok<AGROW>
            end
        end
    end

    center_ok = zero_attitude_z_range([0; 0], B, P_local, ...
        L_min, L_max, joint_limit);
    [~, z0_low, z0_high] = zero_attitude_z_range([0; 0], B, P_local, ...
        L_min, L_max, joint_limit);

    fprintf('Zero-attitude workspace with length %.0f..%.0f mm and ball joint +/-%.0f deg\n', ...
        L_min, L_max, rad2deg(joint_limit));
    fprintf('Center height z range: %.1f .. %.1f mm\n', z0_low, z0_high);
    fprintf('Sampled X range: %.1f .. %.1f mm\n', min(xyz(:,1)), max(xyz(:,1)));
    fprintf('Sampled Y range: %.1f .. %.1f mm\n', min(xyz(:,2)), max(xyz(:,2)));
    fprintf('Sampled Z range: %.1f .. %.1f mm\n', min(xyz(:,3)), max(xyz(:,3)));
    fprintf('Approx. translational volume: %.3f L\n', volume_mm3 / 1e6);
    fprintf('Approx. sampled point count: %d\n', size(xyz, 1));
    fprintf('Center range check flag: %d\n', center_ok);

    figure('Name', 'Zero-attitude reachable workspace');
    scatter3(xyz(:,1), xyz(:,2), xyz(:,3), 4, xyz(:,3), 'filled');
    axis equal; grid on; view(3);
    xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
    title('Reachable center positions, RPY = [0 0 0]');
    colorbar;

    figure('Name', 'Workspace height bounds');
    subplot(1,2,1);
    imagesc(x_values, y_values, z_low_map);
    axis equal tight; set(gca, 'YDir', 'normal'); colorbar;
    xlabel('X (mm)'); ylabel('Y (mm)'); title('Minimum reachable Z (mm)');

    subplot(1,2,2);
    imagesc(x_values, y_values, z_high_map);
    axis equal tight; set(gca, 'YDir', 'normal'); colorbar;
    xlabel('X (mm)'); ylabel('Y (mm)'); title('Maximum reachable Z (mm)');

    %% Optional attitude scan at center position
    z_home = (z0_low + z0_high) / 2;
    angle_values = -35:1:35;
    yaw_values = -45:2:45;
    feasible_rpy = [];
    for roll = angle_values
        for pitch = angle_values
            for yaw = yaw_values
                pos = [0; 0; z_home];
                rpy = deg2rad([roll; pitch; yaw]);
                if is_pose_reachable(pos, rpy, B, P_local, L_min, L_max, joint_limit)
                    feasible_rpy = [feasible_rpy; roll, pitch, yaw]; %#ok<AGROW>
                end
            end
        end
    end

    if ~isempty(feasible_rpy)
        fprintf('\nAttitude scan at center z = %.1f mm:\n', z_home);
        fprintf('Roll range:  %.1f .. %.1f deg\n', min(feasible_rpy(:,1)), max(feasible_rpy(:,1)));
        fprintf('Pitch range: %.1f .. %.1f deg\n', min(feasible_rpy(:,2)), max(feasible_rpy(:,2)));
        fprintf('Yaw range:   %.1f .. %.1f deg\n', min(feasible_rpy(:,3)), max(feasible_rpy(:,3)));

        figure('Name', 'Reachable attitudes at center');
        scatter3(feasible_rpy(:,1), feasible_rpy(:,2), feasible_rpy(:,3), ...
            8, feasible_rpy(:,3), 'filled');
        grid on; axis tight; view(3);
        xlabel('Roll (deg)'); ylabel('Pitch (deg)'); zlabel('Yaw (deg)');
        title(sprintf('Reachable RPY at X=0, Y=0, Z=%.1f mm', z_home));
        colorbar;
    end
end

function [ok, z_low, z_high] = zero_attitude_z_range(xy, B, P_local, L_min, L_max, joint_limit)
    x = xy(1);
    y = xy(2);
    z2_low = 0;
    z2_high = inf;
    z_joint_low = 0;

    for i = 1:6
        dxy = [x; y] + P_local(1:2,i) - B(1:2,i);
        d2 = dxy.' * dxy;
        d = sqrt(d2);

        z2_low = max(z2_low, L_min^2 - d2);
        z2_high = min(z2_high, L_max^2 - d2);
        z_joint_low = max(z_joint_low, d / tan(joint_limit));
    end

    z_low = max(sqrt(max(0, z2_low)), z_joint_low);
    z_high = sqrt(z2_high);
    ok = isreal(z_high) && isfinite(z_high) && z_low <= z_high;
end

function ok = is_pose_reachable(pos, rpy, B, P_local, L_min, L_max, joint_limit)
    R = eul2rot_local(rpy);
    T = repmat(pos, 1, 6) + R * P_local;
    v = T - B;
    L = sqrt(sum(v.^2, 1));
    u = v ./ L;

    base_ok = all(acos(max(min(u(3,:), 1), -1)) <= joint_limit);
    platform_ok = true;
    platform_z = R(:,3);
    for i = 1:6
        platform_angle = acos(max(min(dot(u(:,i), platform_z), 1), -1));
        platform_ok = platform_ok && platform_angle <= joint_limit;
    end

    ok = all(L >= L_min & L <= L_max) && base_ok && platform_ok;
end

function R = eul2rot_local(rpy)
    rx = rpy(1); ry = rpy(2); rz = rpy(3);
    Rx = [1 0 0; 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)];
    Ry = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)];
    Rz = [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1];
    R = Rz * Ry * Rx;
end
