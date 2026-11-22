clear; close all; clc;

fprintf('========================================\n');
fprintf('DeepONet Kernel Comparison Visualization (2 Systems)\n');
fprintf('========================================\n\n');

fprintf('[1/2] Loading data...\n');

sys_files = {'data/kernel_comparison_lam1.mat', 'data/kernel_comparison_lam2.mat'};
sys_labels = {'1', '2'};
all_data = cell(1, length(sys_files));

for s = 1:length(sys_files)
    try
        data = load(sys_files{s});
    catch
        error(['Data file %s not found!\nPlease regenerate the corresponding .mat file from the original project'], sys_files{s});
    end

    if isfield(data, 'a_func_str')
        fprintf('  System %s: a(x) = %s\n', sys_labels{s}, data.a_func_str);
    end
    if isfield(data, 'lam_func_str')
        fprintf('  System %s: lambda(x) = %s\n', sys_labels{s}, data.lam_func_str);
    end
    if isfield(data, 'mu')
        fprintf('  System %s: mu = %.1f\n', sys_labels{s}, data.mu);
    end
    fprintf('  System %s: matrix size %d x %d\n', sys_labels{s}, size(data.k_true,1), size(data.k_true,2));

    all_data{s} = data;
    plot_system_comparison(data, sys_labels{s}, s);
end

fprintf('\n[Combined] Generating 2x3 combined comparison...\n');
plot_combined_comparison(all_data, sys_labels);

fprintf('\nDone!\n');

function plot_system_comparison(data, sys_label, fig_index)
    k_true = data.k_true;
    k_pred = data.k_hat;
    N = size(k_true, 1);

    spatial = linspace(-0.5, 0.5, N);
    [meshx, meshy] = ndgrid(spatial, spatial);
    mask = abs(meshy) <= abs(meshx);

    k_true_masked = k_true;
    k_pred_masked = k_pred;

    if isfield(data, 'diff')
        diff_data = data.diff;
    else
        diff_data = k_pred - k_true;
    end
    diff_masked = diff_data;

    k_true_masked(~mask) = NaN;
    k_pred_masked(~mask) = NaN;
    diff_masked(~mask) = NaN;

    z_min = min(k_true(:));
    z_max = max(k_true(:));
    z_range = z_max - z_min;
    z_limit = [z_min - 0.1*z_range, z_max + 0.1*z_range];

    fprintf('\n[2/2] Generating 3D comparison plot for System %s...\n', sys_label);

    fig = figure('Position', [100 + (fig_index-1)*60, 100 + (fig_index-1)*60, 1200, 350], 'Color', 'white');
    subplot_width = 0.20;
    subplot_height = 0.55;
    bottom_margin = 0.25;
    left_margin = 0.08;
    gap = 0.13;

    subplot('Position', [left_margin, bottom_margin, subplot_width, subplot_height]);
    surf(meshx, meshy, k_true_masked, 'EdgeColor', 'none', 'FaceColor', 'interp');
    xlabel({'y', '(a)'}, 'FontSize', 20, 'FontWeight', 'bold', 'Interpreter', 'tex', 'FontName', 'Cambria Math');
    ylabel('$\xi$', 'FontSize', 18, 'Interpreter', 'latex');
    zlabel(sprintf('$k_{%s}$', sys_label), 'FontSize', 18, 'Interpreter', 'latex');
    title(sprintf('$k_{%s}$', sys_label), 'FontSize', 18, 'Interpreter', 'latex');
    view(10, 15);
    colormap parula;
    shading interp; axis tight;
    zlim(z_limit);

    subplot('Position', [left_margin + subplot_width + gap, bottom_margin, subplot_width, subplot_height]);
    surf(meshx, meshy, k_pred_masked, 'EdgeColor', 'none', 'FaceColor', 'interp');
    xlabel({'y', '(b)'}, 'FontSize', 20, 'FontWeight', 'bold', 'Interpreter', 'tex', 'FontName', 'Cambria Math');
    ylabel('$\xi$', 'FontSize', 18, 'Interpreter', 'latex');
    zlabel(sprintf('$\\hat{k}_{%s}$', sys_label), 'FontSize', 18, 'Interpreter', 'latex');
    title(sprintf('$\\hat{k}_{%s}$', sys_label), 'FontSize', 18, 'Interpreter', 'latex');
    view(10, 15);
    colormap parula;
    shading interp; axis tight;
    zlim(z_limit);

    subplot('Position', [left_margin + 2*(subplot_width + gap), bottom_margin, subplot_width, subplot_height]);
    surf(meshx, meshy, diff_masked, 'EdgeColor', 'none', 'FaceColor', 'interp');
    xlabel({'y', '(c)'}, 'FontSize', 20, 'FontWeight', 'bold', 'Interpreter', 'tex', 'FontName', 'Cambria Math');
    ylabel('$\xi$', 'FontSize', 18, 'Interpreter', 'latex');
    zlabel(sprintf('$k_{%s} - \\hat{k}_{%s}$', sys_label, sys_label), 'FontSize', 18, 'Interpreter', 'latex');
    title(sprintf('kernel error $k_{%s} - \\hat{k}_{%s}$', sys_label, sys_label), 'FontSize', 18, 'Interpreter', 'latex');
    view(10, 15);
    colormap parula;
    shading interp; axis tight;

    valid_diff = diff_data(mask);
    fprintf('\n--- System %s Difference Statistics ---\n', sys_label);
    fprintf('Max diff: %.6f\n', max(valid_diff));
    fprintf('Min diff: %.6f\n', min(valid_diff));
    fprintf('Mean diff: %.6f\n', mean(valid_diff));
    fprintf('Std diff: %.6f\n', std(valid_diff));
    if isfield(data, 'relative_l2_error')
        fprintf('Relative L2 error: %.4f%%\n', data.relative_l2_error * 100);
    end

    if ~exist('figs', 'dir')
        mkdir('figs');
    end
    png_name = sprintf('figs/kernel_comparison_3d_sys%s.png', sys_label);
    exportgraphics(fig, png_name, 'Resolution', 300);
    fprintf('Figure saved to: %s\n', png_name);
end

function plot_combined_comparison(all_data, sys_labels)
    nsys = length(all_data);

    kern_cache = cell(nsys, 1);
    for s = 1:nsys
        data = all_data{s};
        lbl = sys_labels{s};

        [meshx, meshy, k_true_m, k_pred_m, diff_m, z_limit] = prepare_kernel_masks(data);

        kern_cache{s} = struct('meshx', meshx, 'meshy', meshy, ...
            'true', k_true_m, 'pred', k_pred_m, 'diff', diff_m, ...
            'zlim', z_limit, 'lbl', lbl);
    end

    panel_dir = 'figs/plot_kernel';
    if ~exist(panel_dir, 'dir'); mkdir(panel_dir); end
    panelPos = [100, 100, 460, 400];
    save_names = {'kernel_comparison_3d_combined_sys1_true','kernel_comparison_3d_combined_sys1_pred','kernel_comparison_3d_combined_sys1_err', ...
                  'kernel_comparison_3d_combined_sys2_true','kernel_comparison_3d_combined_sys2_pred','kernel_comparison_3d_combined_sys2_err'};
    idx = 0;
    for s = 1:nsys
        c = kern_cache{s};
        cols = {c.true, c.pred, c.diff};
        for col = 1:3
            idx = idx + 1;
            save_combined_panel(c.meshx, c.meshy, cols{col}, c.lbl, col, c.zlim, col ~= 3, ...
                fullfile(panel_dir, save_names{idx}), panelPos);
        end
    end
end

function save_combined_panel(meshx, meshy, Z, sys_label, col, z_limit, use_zlim, outpath, panelPos)

    fig = figure('Color', 'w', 'Position', panelPos);
    surf(meshx, meshy, Z, 'EdgeColor', 'none', 'FaceColor', 'interp');
    xlabel('y', 'FontSize', 23, 'Interpreter', 'tex', 'FontName', 'Cambria Math');
    ylabel('$\xi$', 'FontSize', 23, 'Interpreter', 'latex');
    if col == 1
        zlabel(sprintf('$k_{%s}$', sys_label), 'FontSize', 23, 'Interpreter', 'latex');
    elseif col == 2
        zlabel(sprintf('$\\hat{k}_{%s}$', sys_label), 'FontSize', 23, 'Interpreter', 'latex');
    else
        zlabel(sprintf('$k_{%s} - \\hat{k}_{%s}$', sys_label, sys_label), 'FontSize', 20, 'Interpreter', 'latex');
    end
    view(10, 15); colormap parula; shading interp; axis tight;
    if use_zlim, zlim(z_limit); end
    exportgraphics(fig, [outpath '.png'], 'Resolution', 300);
    fprintf('Panel saved: %s.png\n', outpath);
    close(fig);
end

function [meshx, meshy, k_true_m, k_pred_m, diff_m, z_limit] = prepare_kernel_masks(data)
    k_true = data.k_true;
    k_pred = data.k_hat;
    N = size(k_true, 1);
    spatial = linspace(-0.5, 0.5, N);
    [meshx, meshy] = ndgrid(spatial, spatial);
    mask = abs(meshy) <= abs(meshx);

    if isfield(data, 'diff')
        diff_data = data.diff;
    else
        diff_data = k_pred - k_true;
    end

    k_true_m = k_true;  k_true_m(~mask) = NaN;
    k_pred_m = k_pred;  k_pred_m(~mask) = NaN;
    diff_m   = diff_data;  diff_m(~mask) = NaN;

    z_min = min(k_true(:));  z_max = max(k_true(:));
    z_range = z_max - z_min;
    z_limit = [z_min - 0.1*z_range, z_max + 0.1*z_range];
end
