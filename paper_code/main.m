CONTROL = 'ctrl';
PLOT_COMPARISON = true;

L = 1; M = 100;
T = 2; N = 2000;

a1 = @(x) 0.4 + 1.5*(x-0.5).^2;
lam1 = @(x) 0.05 + 0.10*x;
ic1 = @(x) 0.2 + 6*x;

a2 = @(x) 0.4 + 1.5*(x-0.5).^2;
lam2 = @(x) 0.05 + 0.10*x;
ic2 = @(x) 0.125 + 0.0125*x.^2;

alpha1 = 1e-3;
alpha2 = 0.08;
eps_r  = 1e-3;

pde1 = create_pde('Ka',a1,'Va',lam1, 'La',0, 'ic',ic1,'bc0',Robin(-1,1,0),'bcl',Neumann(0));
pde2 = create_pde('Ka',a2,'Va',lam2, 'La',0, 'ic',ic2,'bc0',Robin(-1,1,0),'bcl',Neumann(0));
sol1 = create_pde_sol(pde1, 'L',L, 'M',M, 'T',T, 'N',N);
sol2 = create_pde_sol(pde2, 'L',L, 'M',M, 'T',T, 'N',N);

method_tag = 'deeponet';
KERNEL_MAT = fullfile('data', 'kernels_deeponet.mat');
KERNEL_CMD = 'Please regenerate the DeepONet kernel from the original project';

fprintf('\n========== Kernel source: %s ==========\n', upper(method_tag));

if ~exist(KERNEL_MAT, 'file')
    error('Kernel file %s not found. %s', KERNEL_MAT, KERNEL_CMD);
end

kernel_data = load(KERNEL_MAT);

if isfield(kernel_data, 'method')
    stored_method = lower(char(kernel_data.method));
    fprintf('Kernel file: %s (method = %s)\n', KERNEL_MAT, stored_method);
    if ~strcmp(stored_method, method_tag)
        warning('Kernel file %s was generated with method = %s, requested %s.', ...
            KERNEL_MAT, stored_method, method_tag);
    end
end

need_fields = {'k_half', 'dk_half', 'k_neg_half', 'dk_neg_half', ...
               'k_half_2', 'dk_half_2', 'k_neg_half_2', 'dk_neg_half_2'};
missing_fields = {};
for ii = 1:numel(need_fields)
    if ~isfield(kernel_data, need_fields{ii})
        missing_fields{end+1} = need_fields{ii};
    end
end
if ~isempty(missing_fields)
    error('Kernel file %s is missing fields: %s', KERNEL_MAT, strjoin(missing_fields, ', '));
end

k_half        = double(kernel_data.k_half(:));
dk_half       = double(kernel_data.dk_half(:));
k_neg_half    = double(kernel_data.k_neg_half(:));
dk_neg_half   = double(kernel_data.dk_neg_half(:));
k_half_2      = double(kernel_data.k_half_2(:));
dk_half_2     = double(kernel_data.dk_half_2(:));
k_neg_half_2  = double(kernel_data.k_neg_half_2(:));
dk_neg_half_2 = double(kernel_data.dk_neg_half_2(:));

fprintf('  U1: k_half range [%.4f, %.4f]\n', min(k_half), max(k_half));
fprintf('  U2: k_half_2 range [%.4f, %.4f]\n', min(k_half_2), max(k_half_2));

y = linspace(-0.5, 0.5, M+1)';
Delta1 = zeros(M+1, 1);
for i = 1:M+1
    xi = y(i);
    x_integral = linspace(-0.5, xi, 100)';
    integrand = arrayfun(@(x) lam1(x+0.5)./a1(x+0.5), x_integral);
    Delta1(i) = exp(-0.5 * trapz(x_integral, integrand));
end

Delta2 = zeros(M+1, 1);
for i = 1:M+1
    xi = y(i);
    x_integral = linspace(-0.5, xi, 100)';
    integrand = arrayfun(@(x) lam2(x+0.5)./a2(x+0.5), x_integral);
    Delta2(i) = exp(-0.5 * trapz(x_integral, integrand));
end

Delta1_inv_half = 1.0 / Delta1(end);
Delta2_inv_half = 1.0 / Delta2(end);

V1_check = 1 - 0.5 * lam1(0) / a1(0);
V2_check = 0.5 * lam1(1) / a1(1);
V3_check = 1 - 0.5 * lam2(0) / a2(0);
V4_check = 0.5 * lam2(1) / a2(1);

u1c = zeros(1, N+1);
u2c = zeros(1, N+1);
u3c = zeros(1, N+1);
u4c = zeros(1, N+1);

xi_grid = linspace(-0.5, 0.5, M+1)';

for n = 1:N
    if strcmp(CONTROL, 'ctrl')
        U1_at_xi = sol1.u(:,n);

        integrand_U1c = (dk_neg_half - V1_check * k_neg_half) .* Delta1 .* U1_at_xi;
        U1c_val = -k_neg_half(1) * sol1.u(1,n) + trapz(xi_grid, integrand_U1c);

        integrand_U2c = (dk_half + V2_check * k_half) .* Delta1 .* U1_at_xi;
        U2c_val = -k_half(end) * sol1.u(end,n) - Delta1_inv_half * trapz(xi_grid, integrand_U2c);

        u1c(n+1) = U1c_val;
        u2c(n+1) = U2c_val;

        U2_at_xi = sol2.u(:,n);

        integrand_U3c = (dk_neg_half_2 - V3_check * k_neg_half_2) .* Delta2 .* U2_at_xi;
        U3c_val = -k_neg_half_2(1) * sol2.u(1,n) + trapz(xi_grid, integrand_U3c);

        integrand_U4c = (dk_half_2 + V4_check * k_half_2) .* Delta2 .* U2_at_xi;
        U4c_val = -k_half_2(end) * sol2.u(end,n) - Delta2_inv_half * trapz(xi_grid, integrand_U4c);

        u3c(n+1) = U3c_val;
        u4c(n+1) = U4c_val;

        sol1.bc0 = Robin(-1, 1, u1c(n+1));
        sol1.bcl = Neumann(u2c(n+1));
        sol2.bc0 = Robin(-1, 1, u3c(n+1));
        sol2.bcl = Neumann(u4c(n+1));
    else
        sol1.bc0 = Robin(-1, 1, 0);
        sol1.bcl = Neumann(0);
        sol2.bc0 = Robin(-1, 1, 0);
        sol2.bcl = Neumann(0);
    end

    sol1.f = -alpha1*(sol1.u(:,n) + eps_r*sol2.u(:,n));
    sol2.f =  alpha2*(sol1.u(:,n) + eps_r*sol2.u(:,n));

    sol1 = solve_pde_step(sol1);
    sol2 = solve_pde_step(sol2);
end

L2_norm1 = zeros(1, N+1);
L2_norm2 = zeros(1, N+1);
x_col = linspace(0, L, M+1)';

for n = 1:N+1
    L2_norm1(n) = sqrt(trapz(x_col, sol1.u(:,n).^2));
    L2_norm2(n) = sqrt(trapz(x_col, sol2.u(:,n).^2));
end

if strcmp(CONTROL, 'ctrl')
    fprintf('=== Closed-loop System Results ===\n');
else
    fprintf('=== Open-loop System Results ===\n');
end
fprintf('Kernel method: %s\n', method_tag);
fprintf('Initial L2 norm - U1: %.6f, U2: %.6f\n', L2_norm1(1), L2_norm2(1));
fprintf('Final L2 norm - U1: %.6f, U2: %.6f\n', L2_norm1(end), L2_norm2(end));
fprintf('Convergence ratio - U1: %.6f, U2: %.6f\n', ...
    L2_norm1(end)/L2_norm1(1), L2_norm2(end)/L2_norm2(1));
if strcmp(CONTROL, 'ctrl')
    fprintf('Final control signals - U1c: %.6f, U2c: %.6f\n', u1c(end), u2c(end));
    fprintf('                        U3c: %.6f, U4c: %.6f\n', u3c(end), u4c(end));
end

if PLOT_COMPARISON
    panel_dir = 'figs/main';
    if ~exist(panel_dir, 'dir')
        mkdir(panel_dir);
    end
    panelPos = [100, 100, 600, 500];

    fig_x1 = figure('Color', 'w', 'Position', panelPos);
    surf(sol1.t, sol1.x, sol1.u, 'EdgeColor', 'none', 'FaceColor', 'interp');
    xlim([0 T]);
    xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 28);
    ylabel('z', 'Interpreter', 'tex', 'FontSize', 28, 'FontName', 'Cambria Math');
    zlabel('x_1(z,t)', 'Interpreter', 'tex', 'FontSize', 28, 'FontName', 'Cambria Math');
    exportgraphics(fig_x1, fullfile(panel_dir, sprintf('state_evolution_%s_%s_x1.png', CONTROL, method_tag)), 'Resolution', 300);
    close(fig_x1);

    fig_x2 = figure('Color', 'w', 'Position', panelPos);
    surf(sol2.t, sol2.x, sol2.u, 'EdgeColor', 'none', 'FaceColor', 'interp');
    xlim([0 T]);
    xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 28);
    ylabel('z', 'Interpreter', 'tex', 'FontSize', 28, 'FontName', 'Cambria Math');
    zlabel('x_2(z,t)', 'Interpreter', 'tex', 'FontSize', 28, 'FontName', 'Cambria Math');
    exportgraphics(fig_x2, fullfile(panel_dir, sprintf('state_evolution_%s_%s_x2.png', CONTROL, method_tag)), 'Resolution', 300);
    close(fig_x2);
end
