

L = 1; M = 100;
T = 2; N = 2000;
dt = T / N;
t = linspace(0, T, N+1);

fprintf('\n========== Setting Common System Parameters ==========\n');

a1_etm = @(x) 0.4 + 1.5*(x-0.5).^2;
lam1_etm = @(x) 0.05 + 0.1*x;
ic1_etm = @(x) 0.2 + 6*x;

a2_etm = @(x) 0.4 + 1.5*(x-0.5).^2;
lam2_etm = @(x) 0.05 + 0.1*x;
ic2_etm = @(x) 0.125 + 0.0125*x.^2;

alpha1 = 1e-3;
alpha2 = 0.08;
eps_r  = 1e-3;

sigma11 = 10;
sigma21 = 0.3;
sigma31 = 0.1;

sigma12 = 10;
sigma22 = 0.001;
sigma32 = 0.001;

beta1 = 4.5;
beta2 = 10;
eta10 = 1e-5;
eta20 = 1e-5;
fprintf('Common sigma parameters: σ11=%.2f, σ21=%.2f, σ31=%.2f, σ12=%.2f, σ22=%.2f, σ32=%.2f\n', ...
    sigma11, sigma21, sigma31, sigma12, sigma22, sigma32);
fprintf('Dynamic ETM parameters: β1=%.2f, β2=%.2f\n\n', beta1, beta2);

y = linspace(-0.5, 0.5, M+1)';
Delta1_etm = zeros(M+1, 1);
for i = 1:M+1
    xi = y(i);
    x_integral = linspace(-0.5, xi, 100)';
    integrand = arrayfun(@(x) lam1_etm(x+0.5)./a1_etm(x+0.5), x_integral);
    integral_val = trapz(x_integral, integrand);
    Delta1_etm(i) = exp(-0.5 * integral_val);
end

Delta2_etm = zeros(M+1, 1);
for i = 1:M+1
    xi = y(i);
    x_integral = linspace(-0.5, xi, 100)';
    integrand = arrayfun(@(x) lam2_etm(x+0.5)./a2_etm(x+0.5), x_integral);
    integral_val = trapz(x_integral, integrand);
    Delta2_etm(i) = exp(-0.5 * integral_val);
end

Delta1_inv_half_etm = 1.0 / Delta1_etm(end);
Delta2_inv_half_etm = 1.0 / Delta2_etm(end);

V1_check_etm = 1 - 0.5 * lam1_etm(0) / a1_etm(0);
V2_check_etm = 0.5 * lam1_etm(1) / a1_etm(1);
V3_check_etm = 1 - 0.5 * lam2_etm(0) / a2_etm(0);
V4_check_etm = 0.5 * lam2_etm(1) / a2_etm(1);

fprintf('\n========== Loading DeepONet Kernels ==========\n');

KERNEL_MAT = 'data/kernels_deeponet.mat';
if ~exist(KERNEL_MAT, 'file')
    error(['File %s not found!\n' ...
           'Please regenerate data/kernels_deeponet.mat from the original project'], KERNEL_MAT);
end

kernel_data = load(KERNEL_MAT);

if isfield(kernel_data, 'method')
    method_str_etm = char(kernel_data.method);
    fprintf('  Kernel file: %s (method = %s)\n', KERNEL_MAT, method_str_etm);
    if ~strcmpi(method_str_etm, 'deeponet')
        warning(['%s records method = %s, not a DeepONet-predicted kernel!\n' ...
                 '  To use the network kernel, regenerate data/kernels_deeponet.mat from the original project'], ...
                 KERNEL_MAT, method_str_etm);
    end
else
    warning('%s has no method field; cannot confirm it is a DeepONet kernel.', KERNEL_MAT);
end

if isfield(kernel_data, 'k_full')
    k1_full_etm = double(kernel_data.k_full);
    fprintf('DeepONet U1 k_full (full kernel matrix) loaded, size: %d x %d\n', size(k1_full_etm,1), size(k1_full_etm,2));
else
    error('data/kernels_deeponet.mat is missing the U1 system k_full full kernel matrix');
end

if isfield(kernel_data, 'k_half') && isfield(kernel_data, 'dk_half')
    k1_half_etm = double(kernel_data.k_half(:));
    dk1_half_etm = double(kernel_data.dk_half(:));
    fprintf('DeepONet U1 k_half, dk_half loaded\n');
else
    error('data/kernels_deeponet.mat is missing the U1 system k_half or dk_half');
end

if isfield(kernel_data, 'k_neg_half') && isfield(kernel_data, 'dk_neg_half')
    k1_neg_half_etm = double(kernel_data.k_neg_half(:));
    dk1_neg_half_etm = double(kernel_data.dk_neg_half(:));
    fprintf('DeepONet U1 k_neg_half, dk_neg_half loaded\n');
else
    error('data/kernels_deeponet.mat is missing the U1 system k_neg_half or dk_neg_half');
end

if isfield(kernel_data, 'k_full_2')
    k2_full_etm = double(kernel_data.k_full_2);
    fprintf('DeepONet U2 k_full_2 (full kernel matrix) loaded, size: %d x %d\n', size(k2_full_etm,1), size(k2_full_etm,2));
else
    error('data/kernels_deeponet.mat is missing the U2 system k_full_2 full kernel matrix');
end

if isfield(kernel_data, 'k_half_2') && isfield(kernel_data, 'dk_half_2')
    k2_half_etm = double(kernel_data.k_half_2(:));
    dk2_half_etm = double(kernel_data.dk_half_2(:));
    fprintf('DeepONet U2 k_half_2, dk_half_2 loaded\n');
else
    error('data/kernels_deeponet.mat is missing the U2 system k_half_2 or dk_half_2');
end

if isfield(kernel_data, 'k_neg_half_2') && isfield(kernel_data, 'dk_neg_half_2')
    k2_neg_half_etm = double(kernel_data.k_neg_half_2(:));
    dk2_neg_half_etm = double(kernel_data.dk_neg_half_2(:));
    fprintf('DeepONet U2 k_neg_half_2, dk_neg_half_2 loaded\n');
else
    error('data/kernels_deeponet.mat is missing the U2 system k_neg_half_2 or dk_neg_half_2');
end

fprintf('  U1 full kernel: k1_full range [%.4f, %.4f]\n', min(k1_full_etm(:)), max(k1_full_etm(:)));
fprintf('  U1 boundary kernel: k1_half(0.5,xi) range [%.4f, %.4f]\n', min(k1_half_etm), max(k1_half_etm));
fprintf('  U1 boundary kernel: k1_neg_half(-0.5,xi) range [%.4f, %.4f]\n', min(k1_neg_half_etm), max(k1_neg_half_etm));
fprintf('  U2 full kernel: k2_full range [%.4f, %.4f]\n', min(k2_full_etm(:)), max(k2_full_etm(:)));
fprintf('  U2 boundary kernel: k2_half(0.5,xi) range [%.4f, %.4f]\n', min(k2_half_etm), max(k2_half_etm));
fprintf('  U2 boundary kernel: k2_neg_half(-0.5,xi) range [%.4f, %.4f]\n', min(k2_neg_half_etm), max(k2_neg_half_etm));
fprintf('  Method: solveIntegralFD computation\n\n');

fprintf('\n========== Running Dynamic ETM Simulation ==========\n');

pde1_detm = create_pde('Ka',a1_etm,'Va',lam1_etm, 'La',0, 'ic',ic1_etm,'bc0',Robin(-1,1,0),'bcl',Neumann(0));
pde2_detm = create_pde('Ka',a2_etm,'Va',lam2_etm, 'La',0, 'ic',ic2_etm,'bc0',Robin(-1,1,0),'bcl',Neumann(0));
sol1_detm = create_pde_sol(pde1_detm, 'L',L, 'M',M, 'T',T, 'N',N);
sol2_detm = create_pde_sol(pde2_detm, 'L',L, 'M',M, 'T',T, 'N',N);

k1_neg_half_detm = k1_neg_half_etm;
dk1_neg_half_detm = dk1_neg_half_etm;
k1_half_detm = k1_half_etm;
dk1_half_detm = dk1_half_etm;
k2_neg_half_detm = k2_neg_half_etm;
dk2_neg_half_detm = dk2_neg_half_etm;
k2_half_detm = k2_half_etm;
dk2_half_detm = dk2_half_etm;
Delta1_detm = Delta1_etm;
Delta1_inv_half_detm = Delta1_inv_half_etm;
Delta2_detm = Delta2_etm;
Delta2_inv_half_detm = Delta2_inv_half_etm;
V1_check_detm = V1_check_etm;
V2_check_detm = V2_check_etm;
V3_check_detm = V3_check_etm;
V4_check_detm = V4_check_etm;

u1c_detm = zeros(1, N+1);
u2c_detm = zeros(1, N+1);
u3c_detm = zeros(1, N+1);
u4c_detm = zeros(1, N+1);

eta1_detm = zeros(1, N+1);
eta2_detm = zeros(1, N+1);
eta1_detm(1) = eta10;
eta2_detm(1) = eta20;

trigger_times_detm_1 = [];
trigger_indices_detm_1 = [];
last_trigger_idx_detm_1 = 1;
trigger_times_detm_2 = [];
trigger_indices_detm_2 = [];
last_trigger_idx_detm_2 = 1;

sol1_last_trigger_detm_1 = sol1_detm.u(:, 1);
sol2_last_trigger_detm_1 = sol2_detm.u(:, 1);
sol1_last_trigger_detm_2 = sol1_detm.u(:, 1);
sol2_last_trigger_detm_2 = sol2_detm.u(:, 1);

dx_half = 1.0 / M;
xi_grid = linspace(-0.5, 0.5, M+1)';
x_grid = linspace(0, L, M+1)';

fprintf('Running DETM simulation...\n');
for n = 1:N

    U1_at_xi = sol1_detm.u(:,n);
    U2_at_xi = sol2_detm.u(:,n);

    w1_detm = zeros(M+1, 1);
    for i = 1:M+1
        y_val = xi_grid(i);
        base1 = Delta1_detm(i) * U1_at_xi(i);
        y_abs = abs(y_val);
        if y_abs > 0
            xi_min = -y_abs;
            xi_max = y_abs;
            idx_min = find(xi_grid >= xi_min, 1);
            idx_max = find(xi_grid <= xi_max, 1, 'last');
            if ~isempty(idx_min) && ~isempty(idx_max) && idx_min <= idx_max
                xi_integral = xi_grid(idx_min:idx_max);
                k_vals = k1_full_etm(i, idx_min:idx_max)';
                U_vals = U1_at_xi(idx_min:idx_max);
                Delta_vals = Delta1_detm(idx_min:idx_max);
                integrand = k_vals .* Delta_vals .* U_vals;
                integral_val = trapz(xi_integral, integrand);
            else
                integral_val = 0;
            end
        else
            integral_val = 0;
        end
        w1_detm(i) = base1 + integral_val;
    end

    w2_detm = zeros(M+1, 1);
    for i = 1:M+1
        y_val = xi_grid(i);
        base2 = Delta2_detm(i) * U2_at_xi(i);
        y_abs = abs(y_val);
        if y_abs > 0
            xi_min = -y_abs;
            xi_max = y_abs;
            idx_min = find(xi_grid >= xi_min, 1);
            idx_max = find(xi_grid <= xi_max, 1, 'last');
            if ~isempty(idx_min) && ~isempty(idx_max) && idx_min <= idx_max
                xi_integral = xi_grid(idx_min:idx_max);
                k_vals = k2_full_etm(i, idx_min:idx_max)';
                U_vals = U2_at_xi(idx_min:idx_max);
                Delta_vals = Delta2_detm(idx_min:idx_max);
                integrand = k_vals .* Delta_vals .* U_vals;
                integral_val = trapz(xi_integral, integrand);
            else
                integral_val = 0;
            end
        else
            integral_val = 0;
        end
        w2_detm(i) = base2 + integral_val;
    end

    norm_w1_detm = sqrt(trapz(w1_detm.^2) * dx_half);
    norm_w2_detm = sqrt(trapz(w2_detm.^2) * dx_half);
    norm_W_detm = sqrt(norm_w1_detm^2 + norm_w2_detm^2);
    w1_half_detm = w1_detm(end);
    w2_half_detm = w2_detm(end);
    w1_neg_half_detm = w1_detm(1);
    w2_neg_half_detm = w2_detm(1);
    norm_w_half_detm = sqrt(w1_half_detm^2 + w2_half_detm^2);
    norm_w_neg_half_detm = sqrt(w1_neg_half_detm^2 + w2_neg_half_detm^2);

    if n > 1
        U1_diff = sol1_last_trigger_detm_1 - sol1_detm.u(:,n);

        boundary_term_d11 = -k1_neg_half_detm(1) * (sol1_last_trigger_detm_1(1) - sol1_detm.u(1,n));
        integrand_d11 = (dk1_neg_half_detm(:) - V1_check_detm * k1_neg_half_detm) .* Delta1_detm .* U1_diff;
        integral_d11 = trapz(xi_grid, integrand_d11);
        d_11 = boundary_term_d11 + integral_d11;

        boundary_term_d12 = -k1_half_detm(end) * Delta1_detm(end) * (sol1_last_trigger_detm_1(end) - sol1_detm.u(end,n));
        integrand_d12 = (dk1_half_detm(:) + V2_check_detm * k1_half_detm) .* Delta1_detm .* U1_diff;
        integral_d12 = trapz(xi_grid, integrand_d12);
        d_12 = boundary_term_d12 - integral_d12;

        norm_d1_detm = sqrt(d_11^2 + d_12^2);
    else
        d_11 = 0;
        d_12 = 0;
        norm_d1_detm = 0;
    end

    trigger_condition_detm_1 = (eta1_detm(n) + sigma21 * norm_W_detm^2 + sigma31 * norm_w_half_detm^2 <= sigma11 * norm_d1_detm^2);
    should_trigger_detm_1 = (n == 1) || trigger_condition_detm_1;

    if n > 1
        U2_diff = sol2_last_trigger_detm_2 - sol2_detm.u(:,n);

        boundary_term_d21 = -k2_neg_half_detm(1) * (sol2_last_trigger_detm_2(1) - sol2_detm.u(1,n));
        integrand_d21 = (dk2_neg_half_detm(:) - V3_check_detm * k2_neg_half_detm) .* Delta2_detm .* U2_diff;
        integral_d21 = trapz(xi_grid, integrand_d21);
        d_21 = boundary_term_d21 + integral_d21;

        boundary_term_d22 = -k2_half_detm(end) * Delta2_detm(end) * (sol2_last_trigger_detm_2(end) - sol2_detm.u(end,n));
        integrand_d22 = (dk2_half_detm(:) + V4_check_detm * k2_half_detm) .* Delta2_detm .* U2_diff;
        integral_d22 = trapz(xi_grid, integrand_d22);
        d_22 = boundary_term_d22 - integral_d22;

        norm_d2_detm = sqrt(d_21^2 + d_22^2);
    else
        d_21 = 0;
        d_22 = 0;
        norm_d2_detm = 0;
    end

    trigger_condition_detm_2 = (eta2_detm(n) + sigma22 * norm_W_detm^2 + sigma32 * norm_w_neg_half_detm^2 <= sigma12 * norm_d2_detm^2);
    should_trigger_detm_2 = (n == 1) || trigger_condition_detm_2;

    if should_trigger_detm_1

        if n > 1
            Psi1 = -sigma11*norm_d1_detm^2 + sigma21*norm_W_detm^2 + sigma31*norm_w_half_detm^2;
            deta1_detm = -beta1*eta1_detm(n) + Psi1;
            eta1_detm(n+1) = eta1_detm(n) + deta1_detm*dt;
        else
            eta1_detm(n+1) = eta1_detm(n);
        end

        trigger_times_detm_1 = [trigger_times_detm_1, t(n)];
        trigger_indices_detm_1 = [trigger_indices_detm_1, n];
        last_trigger_idx_detm_1 = n;
        sol1_last_trigger_detm_1 = sol1_detm.u(:,n);
        sol2_last_trigger_detm_1 = sol2_detm.u(:,n);

        U1_at_xi_ctrl = sol1_detm.u(:,n);
        integrand_U1c = (dk1_neg_half_detm(:) - V1_check_detm * k1_neg_half_detm) .* Delta1_detm .* U1_at_xi_ctrl;
        integral_U1c = trapz(xi_grid, integrand_U1c);
        U1c_val = -k1_neg_half_detm(1) * sol1_detm.u(1,n) + integral_U1c;
        U1c_val = U1c_val(1);

        integrand_U2c = (dk1_half_detm(:) + V2_check_detm * k1_half_detm) .* Delta1_detm .* U1_at_xi_ctrl;
        integral_U2c = trapz(xi_grid, integrand_U2c);
        U2c_val = -k1_half_detm(end) * sol1_detm.u(end,n) - Delta1_inv_half_detm * integral_U2c;
        U2c_val = U2c_val(1);

        u1c_detm(n+1) = U1c_val;
        u2c_detm(n+1) = U2c_val;
    else

        if n > 1
            Psi1 = -sigma11*norm_d1_detm^2 + sigma21*norm_W_detm^2 + sigma31*norm_w_half_detm^2;
            deta1_detm = -beta1*eta1_detm(n) + Psi1;
            eta1_detm(n+1) = eta1_detm(n) + deta1_detm*dt;
        else
            eta1_detm(n+1) = eta1_detm(n);
        end

        u1c_detm(n+1) = u1c_detm(last_trigger_idx_detm_1 + 1);
        u2c_detm(n+1) = u2c_detm(last_trigger_idx_detm_1 + 1);
    end

    if should_trigger_detm_2

        if n > 1
            Psi2 = -sigma12*norm_d2_detm^2 + sigma22*norm_W_detm^2 + sigma32*norm_w_neg_half_detm^2;
            deta2_detm = -beta2*eta2_detm(n) + Psi2;
            eta2_detm(n+1) = eta2_detm(n) + deta2_detm*dt;
        else
            eta2_detm(n+1) = eta2_detm(n);
        end

        trigger_times_detm_2 = [trigger_times_detm_2, t(n)];
        trigger_indices_detm_2 = [trigger_indices_detm_2, n];
        last_trigger_idx_detm_2 = n;
        sol1_last_trigger_detm_2 = sol1_detm.u(:,n);
        sol2_last_trigger_detm_2 = sol2_detm.u(:,n);

        U2_at_xi_ctrl = sol2_detm.u(:,n);
        integrand_U3c = (dk2_neg_half_detm(:) - V3_check_detm * k2_neg_half_detm) .* Delta2_detm .* U2_at_xi_ctrl;
        integral_U3c = trapz(xi_grid, integrand_U3c);
        U3c_val = -k2_neg_half_detm(1) * sol2_detm.u(1,n) + integral_U3c;
        U3c_val = U3c_val(1);

        integrand_U4c = (dk2_half_detm(:) + V4_check_detm * k2_half_detm) .* Delta2_detm .* U2_at_xi_ctrl;
        integral_U4c = trapz(xi_grid, integrand_U4c);
        U4c_val = -k2_half_detm(end) * sol2_detm.u(end,n) - Delta2_inv_half_detm * integral_U4c;
        U4c_val = U4c_val(1);

        u3c_detm(n+1) = U3c_val;
        u4c_detm(n+1) = U4c_val;
    else

        if n > 1
            Psi2 = -sigma12*norm_d2_detm^2 + sigma22*norm_W_detm^2 + sigma32*norm_w_neg_half_detm^2;
            deta2_detm = -beta2*eta2_detm(n) + Psi2;
            eta2_detm(n+1) = eta2_detm(n) + deta2_detm*dt;
        else
            eta2_detm(n+1) = eta2_detm(n);
        end

        u3c_detm(n+1) = u3c_detm(last_trigger_idx_detm_2 + 1);
        u4c_detm(n+1) = u4c_detm(last_trigger_idx_detm_2 + 1);
    end

    sol1_detm.bc0 = Robin(-1, 1, u1c_detm(n+1));
    sol1_detm.bcl = Neumann(u2c_detm(n+1));
    sol2_detm.bc0 = Robin(-1, 1, u3c_detm(n+1));
    sol2_detm.bcl = Neumann(u4c_detm(n+1));

    sol1_detm.f = -alpha1*(sol1_detm.u(:,n) + eps_r*sol2_detm.u(:,n));
    sol2_detm.f =  alpha2*(sol1_detm.u(:,n) + eps_r*sol2_detm.u(:,n));

    sol1_detm = solve_pde_step(sol1_detm);
    sol2_detm = solve_pde_step(sol2_detm);
end

fprintf('DETM simulation completed!\n');
fprintf('  Total triggers - Left boundary: %d, Right boundary: %d\n', ...
    length(trigger_times_detm_1), length(trigger_times_detm_2));

fprintf('\n========== Running Static ETM Simulation ==========\n');

pde1_setm = create_pde('Ka',a1_etm,'Va',lam1_etm, 'La',0, 'ic',ic1_etm,'bc0',Robin(-1,1,0),'bcl',Neumann(0));
pde2_setm = create_pde('Ka',a2_etm,'Va',lam2_etm, 'La',0, 'ic',ic2_etm,'bc0',Robin(-1,1,0),'bcl',Neumann(0));
sol1_setm = create_pde_sol(pde1_setm, 'L',L, 'M',M, 'T',T, 'N',N);
sol2_setm = create_pde_sol(pde2_setm, 'L',L, 'M',M, 'T',T, 'N',N);

k1_neg_half_setm = k1_neg_half_etm;
dk1_neg_half_setm = dk1_neg_half_etm;
k1_half_setm = k1_half_etm;
dk1_half_setm = dk1_half_etm;
k2_neg_half_setm = k2_neg_half_etm;
dk2_neg_half_setm = dk2_neg_half_etm;
k2_half_setm = k2_half_etm;
dk2_half_setm = dk2_half_etm;
Delta1_setm = Delta1_etm;
Delta1_inv_half_setm = Delta1_inv_half_etm;
Delta2_setm = Delta2_etm;
Delta2_inv_half_setm = Delta2_inv_half_etm;
V1_check_setm = V1_check_etm;
V2_check_setm = V2_check_etm;
V3_check_setm = V3_check_etm;
V4_check_setm = V4_check_etm;

u1c_setm = zeros(1, N+1);
u2c_setm = zeros(1, N+1);
u3c_setm = zeros(1, N+1);
u4c_setm = zeros(1, N+1);

trigger_times_setm_1 = [];
trigger_indices_setm_1 = [];
last_trigger_idx_setm_1 = 1;
trigger_times_setm_2 = [];
trigger_indices_setm_2 = [];
last_trigger_idx_setm_2 = 1;

sol1_last_trigger_setm_1 = sol1_setm.u(:, 1);
sol2_last_trigger_setm_1 = sol2_setm.u(:, 1);
sol1_last_trigger_setm_2 = sol1_setm.u(:, 1);
sol2_last_trigger_setm_2 = sol2_setm.u(:, 1);

fprintf('Running SETM simulation...\n');
for n = 1:N

    U1_at_xi = sol1_setm.u(:,n);
    U2_at_xi = sol2_setm.u(:,n);

    w1_setm = zeros(M+1, 1);
    for i = 1:M+1
        y_val = xi_grid(i);
        base1 = Delta1_setm(i) * U1_at_xi(i);
        y_abs = abs(y_val);
        if y_abs > 0
            xi_min = -y_abs;
            xi_max = y_abs;
            idx_min = find(xi_grid >= xi_min, 1);
            idx_max = find(xi_grid <= xi_max, 1, 'last');
            if ~isempty(idx_min) && ~isempty(idx_max) && idx_min <= idx_max
                xi_integral = xi_grid(idx_min:idx_max);
                k_vals = k1_full_etm(i, idx_min:idx_max)';
                U_vals = U1_at_xi(idx_min:idx_max);
                Delta_vals = Delta1_setm(idx_min:idx_max);
                integrand = k_vals .* Delta_vals .* U_vals;
                integral_val = trapz(xi_integral, integrand);
            else
                integral_val = 0;
            end
        else
            integral_val = 0;
        end
        w1_setm(i) = base1 + integral_val;
    end

    w2_setm = zeros(M+1, 1);
    for i = 1:M+1
        y_val = xi_grid(i);
        base2 = Delta2_setm(i) * U2_at_xi(i);
        y_abs = abs(y_val);
        if y_abs > 0
            xi_min = -y_abs;
            xi_max = y_abs;
            idx_min = find(xi_grid >= xi_min, 1);
            idx_max = find(xi_grid <= xi_max, 1, 'last');
            if ~isempty(idx_min) && ~isempty(idx_max) && idx_min <= idx_max
                xi_integral = xi_grid(idx_min:idx_max);
                k_vals = k2_full_etm(i, idx_min:idx_max)';
                U_vals = U2_at_xi(idx_min:idx_max);
                Delta_vals = Delta2_setm(idx_min:idx_max);
                integrand = k_vals .* Delta_vals .* U_vals;
                integral_val = trapz(xi_integral, integrand);
            else
                integral_val = 0;
            end
        else
            integral_val = 0;
        end
        w2_setm(i) = base2 + integral_val;
    end

    norm_w1_setm = sqrt(trapz(w1_setm.^2) * dx_half);
    norm_w2_setm = sqrt(trapz(w2_setm.^2) * dx_half);
    norm_W_setm = sqrt(norm_w1_setm^2 + norm_w2_setm^2);
    w1_half_setm = w1_setm(end);
    w2_half_setm = w2_setm(end);
    w1_neg_half_setm = w1_setm(1);
    w2_neg_half_setm = w2_setm(1);
    norm_w_half_setm = sqrt(w1_half_setm^2 + w2_half_setm^2);
    norm_w_neg_half_setm = sqrt(w1_neg_half_setm^2 + w2_neg_half_setm^2);

    if n > 1
        U1_diff = sol1_last_trigger_setm_1 - sol1_setm.u(:,n);

        boundary_term_d11 = -k1_neg_half_setm(1) * (sol1_last_trigger_setm_1(1) - sol1_setm.u(1,n));
        integrand_d11 = (dk1_neg_half_setm(:) - V1_check_setm * k1_neg_half_setm) .* Delta1_setm .* U1_diff;
        integral_d11 = trapz(xi_grid, integrand_d11);
        d_11 = boundary_term_d11 + integral_d11;

        boundary_term_d12 = -k1_half_setm(end) * Delta1_setm(end) * (sol1_last_trigger_setm_1(end) - sol1_setm.u(end,n));
        integrand_d12 = (dk1_half_setm(:) + V2_check_setm * k1_half_setm) .* Delta1_setm .* U1_diff;
        integral_d12 = trapz(xi_grid, integrand_d12);
        d_12 = boundary_term_d12 - integral_d12;

        norm_d1_setm = sqrt(d_11^2 + d_12^2);
    else
        d_11 = 0;
        d_12 = 0;
        norm_d1_setm = 0;
    end

    threshold_1 = sigma21 * norm_W_setm^2 + sigma31 * norm_w_half_setm^2;
    trigger_condition_setm_1 = (sigma11 * norm_d1_setm^2 >= threshold_1);
    should_trigger_setm_1 = (n == 1) || trigger_condition_setm_1;

    if n > 1
        U2_diff = sol2_last_trigger_setm_2 - sol2_setm.u(:,n);

        boundary_term_d21 = -k2_neg_half_setm(1) * (sol2_last_trigger_setm_2(1) - sol2_setm.u(1,n));
        integrand_d21 = (dk2_neg_half_setm(:) - V3_check_setm * k2_neg_half_setm) .* Delta2_setm .* U2_diff;
        integral_d21 = trapz(xi_grid, integrand_d21);
        d_21 = boundary_term_d21 + integral_d21;

        boundary_term_d22 = -k2_half_setm(end) * Delta2_setm(end) * (sol2_last_trigger_setm_2(end) - sol2_setm.u(end,n));
        integrand_d22 = (dk2_half_setm(:) + V4_check_setm * k2_half_setm) .* Delta2_setm .* U2_diff;
        integral_d22 = trapz(xi_grid, integrand_d22);
        d_22 = boundary_term_d22 - integral_d22;

        norm_d2_setm = sqrt(d_21^2 + d_22^2);
    else
        d_21 = 0;
        d_22 = 0;
        norm_d2_setm = 0;
    end

    threshold_2 = sigma22 * norm_W_setm^2 + sigma32 * norm_w_neg_half_setm^2;
    trigger_condition_setm_2 = (sigma12 * norm_d2_setm^2 >= threshold_2);
    should_trigger_setm_2 = (n == 1) || trigger_condition_setm_2;

    if should_trigger_setm_1
        trigger_times_setm_1 = [trigger_times_setm_1, t(n)];
        trigger_indices_setm_1 = [trigger_indices_setm_1, n];
        last_trigger_idx_setm_1 = n;
        sol1_last_trigger_setm_1 = sol1_setm.u(:,n);
        sol2_last_trigger_setm_1 = sol2_setm.u(:,n);

        U1_at_xi_ctrl = sol1_setm.u(:,n);
        integrand_U1c = (dk1_neg_half_setm(:) - V1_check_setm * k1_neg_half_setm) .* Delta1_setm .* U1_at_xi_ctrl;
        integral_U1c = trapz(xi_grid, integrand_U1c);
        U1c_val = -k1_neg_half_setm(1) * sol1_setm.u(1,n) + integral_U1c;
        U1c_val = U1c_val(1);

        integrand_U2c = (dk1_half_setm(:) + V2_check_setm * k1_half_setm) .* Delta1_setm .* U1_at_xi_ctrl;
        integral_U2c = trapz(xi_grid, integrand_U2c);
        U2c_val = -k1_half_setm(end) * sol1_setm.u(end,n) - Delta1_inv_half_setm * integral_U2c;
        U2c_val = U2c_val(1);
        u1c_setm(n+1) = U1c_val;
        u2c_setm(n+1) = U2c_val;
    else
        u1c_setm(n+1) = u1c_setm(last_trigger_idx_setm_1 + 1);
        u2c_setm(n+1) = u2c_setm(last_trigger_idx_setm_1 + 1);
    end

    if should_trigger_setm_2
        trigger_times_setm_2 = [trigger_times_setm_2, t(n)];
        trigger_indices_setm_2 = [trigger_indices_setm_2, n];
        last_trigger_idx_setm_2 = n;
        sol1_last_trigger_setm_2 = sol1_setm.u(:,n);
        sol2_last_trigger_setm_2 = sol2_setm.u(:,n);

        U2_at_xi_ctrl = sol2_setm.u(:,n);
        integrand_U3c = (dk2_neg_half_setm(:) - V3_check_setm * k2_neg_half_setm) .* Delta2_setm .* U2_at_xi_ctrl;
        integral_U3c = trapz(xi_grid, integrand_U3c);
        U3c_val = -k2_neg_half_setm(1) * sol2_setm.u(1,n) + integral_U3c;
        U3c_val = U3c_val(1);
        integrand_U4c = (dk2_half_setm(:) + V4_check_setm * k2_half_setm) .* Delta2_setm .* U2_at_xi_ctrl;
        integral_U4c = trapz(xi_grid, integrand_U4c);
        U4c_val = -k2_half_setm(end) * sol2_setm.u(end,n) - Delta2_inv_half_setm * integral_U4c;
        U4c_val = U4c_val(1);
        u3c_setm(n+1) = U3c_val;
        u4c_setm(n+1) = U4c_val;
    else
        u3c_setm(n+1) = u3c_setm(last_trigger_idx_setm_2 + 1);
        u4c_setm(n+1) = u4c_setm(last_trigger_idx_setm_2 + 1);
    end

    sol1_setm.bc0 = Robin(-1, 1, u1c_setm(n+1));
    sol1_setm.bcl = Neumann(u2c_setm(n+1));
    sol2_setm.bc0 = Robin(-1, 1, u3c_setm(n+1));
    sol2_setm.bcl = Neumann(u4c_setm(n+1));

    sol1_setm.f = -alpha1*(sol1_setm.u(:,n) + eps_r*sol2_setm.u(:,n));
    sol2_setm.f =  alpha2*(sol1_setm.u(:,n) + eps_r*sol2_setm.u(:,n));

    sol1_setm = solve_pde_step(sol1_setm);
    sol2_setm = solve_pde_step(sol2_setm);
end

fprintf('SETM simulation completed!\n');
fprintf('  Total triggers - Left boundary: %d, Right boundary: %d\n', ...
    length(trigger_times_setm_1), length(trigger_times_setm_2));

fprintf('\n========== Creating Comparison Plots ==========\n');

if ~isempty(trigger_times_detm_1)
    ETCtimes_detm_1 = length(trigger_indices_detm_1);
    trigger_intervals_detm_1 = zeros(1, ETCtimes_detm_1);
    for i = 2:ETCtimes_detm_1
        trigger_intervals_detm_1(i-1) = t(trigger_indices_detm_1(i)) - t(trigger_indices_detm_1(i-1));
        if (i == ETCtimes_detm_1)
            if trigger_indices_detm_1(end) <= N
                trigger_intervals_detm_1(i) = t(N) - t(trigger_indices_detm_1(end));
            else
                trigger_intervals_detm_1(i) = t(N) - t(trigger_indices_detm_1(end-1));
            end
        end
    end
end

if ~isempty(trigger_times_detm_2)
    ETCtimes_detm_2 = length(trigger_indices_detm_2);
    trigger_intervals_detm_2 = zeros(1, ETCtimes_detm_2);
    for i = 2:ETCtimes_detm_2
        trigger_intervals_detm_2(i-1) = t(trigger_indices_detm_2(i)) - t(trigger_indices_detm_2(i-1));
        if (i == ETCtimes_detm_2)
            if trigger_indices_detm_2(end) <= N
                trigger_intervals_detm_2(i) = t(N) - t(trigger_indices_detm_2(end));
            else
                trigger_intervals_detm_2(i) = t(N) - t(trigger_indices_detm_2(end-1));
            end
        end
    end
end

if ~isempty(trigger_times_setm_1)
    ETCtimes_setm_1 = length(trigger_indices_setm_1);
    trigger_intervals_setm_1 = zeros(1, ETCtimes_setm_1);
    for i = 2:ETCtimes_setm_1
        trigger_intervals_setm_1(i-1) = t(trigger_indices_setm_1(i)) - t(trigger_indices_setm_1(i-1));
        if (i == ETCtimes_setm_1)
            if trigger_indices_setm_1(end) <= N
                trigger_intervals_setm_1(i) = t(N) - t(trigger_indices_setm_1(end));
            else
                trigger_intervals_setm_1(i) = t(N) - t(trigger_indices_setm_1(end-1));
            end
        end
    end
end

if ~isempty(trigger_times_setm_2)
    ETCtimes_setm_2 = length(trigger_indices_setm_2);
    trigger_intervals_setm_2 = zeros(1, ETCtimes_setm_2);
    for i = 2:ETCtimes_setm_2
        trigger_intervals_setm_2(i-1) = t(trigger_indices_setm_2(i)) - t(trigger_indices_setm_2(i-1));
        if (i == ETCtimes_setm_2)
            if trigger_indices_setm_2(end) <= N
                trigger_intervals_setm_2(i) = t(N) - t(trigger_indices_setm_2(end));
            else
                trigger_intervals_setm_2(i) = t(N) - t(trigger_indices_setm_2(end-1));
            end
        end
    end
end

total_triggers_detm = length(trigger_times_detm_1) + length(trigger_times_detm_2);
total_triggers_setm = length(trigger_times_setm_1) + length(trigger_times_setm_2);

if ~exist('figs', 'dir')
    mkdir('figs');
end
trigger_stats_file = fopen('figs/trigger_counts.txt', 'w');
fprintf(trigger_stats_file, 'ETM Trigger Counts\n');
fprintf(trigger_stats_file, '==================\n\n');
fprintf(trigger_stats_file, 'Dynamic ETM:\n');
fprintf(trigger_stats_file, '  U1: %d\n', length(trigger_times_detm_1));
fprintf(trigger_stats_file, '  U2: %d\n', length(trigger_times_detm_2));
fprintf(trigger_stats_file, '  Total: %d\n\n', total_triggers_detm);
fprintf(trigger_stats_file, 'Static ETM:\n');
fprintf(trigger_stats_file, '  U1: %d\n', length(trigger_times_setm_1));
fprintf(trigger_stats_file, '  U2: %d\n', length(trigger_times_setm_2));
fprintf(trigger_stats_file, '  Total: %d\n', total_triggers_setm);
fclose(trigger_stats_file);
fprintf('Trigger counts saved to: figs/trigger_counts.txt\n');

panel_dir = 'figs/compare_etm';
if ~exist(panel_dir, 'dir'); mkdir(panel_dir); end
panelPos = [100, 100, 560, 460];

save_trigger_panel(trigger_times_detm_1, trigger_intervals_detm_1, ETCtimes_detm_1, 'b', ...
    fullfile(panel_dir, 'etm_trigger_intervals_detm_x1'), panelPos, [0 T]);
save_trigger_panel(trigger_times_detm_2, trigger_intervals_detm_2, ETCtimes_detm_2, 'r', ...
    fullfile(panel_dir, 'etm_trigger_intervals_detm_x2'), panelPos, [0 T]);
save_trigger_panel(trigger_times_setm_1, trigger_intervals_setm_1, ETCtimes_setm_1, 'b', ...
    fullfile(panel_dir, 'etm_trigger_intervals_setm_x1'), panelPos, [0 T]);
save_trigger_panel(trigger_times_setm_2, trigger_intervals_setm_2, ETCtimes_setm_2, 'r', ...
    fullfile(panel_dir, 'etm_trigger_intervals_setm_x2'), panelPos, [0 T]);

fprintf('\n========== Creating System State Evolution Plots ==========\n');

x_grid = linspace(0, L, M+1);

if ~exist('figs/compare_etm', 'dir'); mkdir('figs/compare_etm'); end
panelPos = [100, 100, 600, 500];
viewAz = -135; viewEl = 35;

fig_s1 = figure('Color', 'w', 'Position', panelPos);
surf(t, x_grid, sol1_detm.u, 'EdgeColor', 'none');
xlim([0 T]); ax = gca; ax.FontSize = 14;
xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 28);
ylabel('z', 'Interpreter', 'tex', 'FontSize', 28, 'FontName', 'Cambria Math');
zlabel('x_1(z,t)', 'Interpreter', 'tex', 'FontSize', 28, 'FontName', 'Cambria Math');
view(viewAz, viewEl);
exportgraphics(fig_s1, fullfile('figs/compare_etm', 'system_state_detm_x1.png'), 'Resolution', 300);
close(fig_s1);

fig_s2 = figure('Color', 'w', 'Position', panelPos);
surf(t, x_grid, sol2_detm.u, 'EdgeColor', 'none');
xlim([0 T]); ax = gca; ax.FontSize = 14;
xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 28);
ylabel('z', 'Interpreter', 'tex', 'FontSize', 28, 'FontName', 'Cambria Math');
zlabel('x_2(z,t)', 'Interpreter', 'tex', 'FontSize', 28, 'FontName', 'Cambria Math');
view(viewAz, viewEl);
exportgraphics(fig_s2, fullfile('figs/compare_etm', 'system_state_detm_x2.png'), 'Resolution', 300);
close(fig_s2);

fig_s3 = figure('Color', 'w', 'Position', panelPos);
surf(t, x_grid, sol1_setm.u, 'EdgeColor', 'none');
xlim([0 T]); ax = gca; ax.FontSize = 14;
xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 28);
ylabel('z', 'Interpreter', 'tex', 'FontSize', 28, 'FontName', 'Cambria Math');
zlabel('x_1(z,t)', 'Interpreter', 'tex', 'FontSize', 28, 'FontName', 'Cambria Math');
view(viewAz, viewEl);
exportgraphics(fig_s3, fullfile('figs/compare_etm', 'system_state_setm_x1.png'), 'Resolution', 300);
close(fig_s3);

fig_s4 = figure('Color', 'w', 'Position', panelPos);
surf(t, x_grid, sol2_setm.u, 'EdgeColor', 'none');
xlim([0 T]); ax = gca; ax.FontSize = 14;
xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 28);
ylabel('z', 'Interpreter', 'tex', 'FontSize', 28, 'FontName', 'Cambria Math');
zlabel('x_2(z,t)', 'Interpreter', 'tex', 'FontSize', 28, 'FontName', 'Cambria Math');
view(viewAz, viewEl);
exportgraphics(fig_s4, fullfile('figs/compare_etm', 'system_state_setm_x2.png'), 'Resolution', 300);
close(fig_s4);

fprintf('\n========== Comparison Statistics ==========\n');
fprintf('\n--- Dynamic ETM (DETM) ---\n');
fprintf('U₁ system:\n');
fprintf('  Total triggers: %d\n', length(trigger_times_detm_1));
fprintf('  Update rate: %.2f%%\n', 100 * length(trigger_times_detm_1) / N);
fprintf('U₂ system:\n');
fprintf('  Total triggers: %d\n', length(trigger_times_detm_2));
fprintf('  Update rate: %.2f%%\n', 100 * length(trigger_times_detm_2) / N);
fprintf('Total: %d triggers\n', length(trigger_times_detm_1) + length(trigger_times_detm_2));
fprintf('Overall update rate: %.2f%%\n', 100 * (length(trigger_times_detm_1) + length(trigger_times_detm_2)) / (2 * N));

fprintf('\n--- Static ETM (SETM) ---\n');
fprintf('U₁ system:\n');
fprintf('  Total triggers: %d\n', length(trigger_times_setm_1));
fprintf('  Update rate: %.2f%%\n', 100 * length(trigger_times_setm_1) / N);
fprintf('U₂ system:\n');
fprintf('  Total triggers: %d\n', length(trigger_times_setm_2));
fprintf('  Update rate: %.2f%%\n', 100 * length(trigger_times_setm_2) / N);
fprintf('Total: %d triggers\n', length(trigger_times_setm_1) + length(trigger_times_setm_2));
fprintf('Overall update rate: %.2f%%\n', 100 * (length(trigger_times_setm_1) + length(trigger_times_setm_2)) / (2 * N));

fprintf('\n--- Comparison ---\n');
diff_u1 = length(trigger_times_setm_1) - length(trigger_times_detm_1);
diff_u2 = length(trigger_times_setm_2) - length(trigger_times_detm_2);
diff_total = diff_u1 + diff_u2;
fprintf('U₁ system: SETM has %d more triggers than DETM (%.1f%% more)\n', ...
    diff_u1, 100 * diff_u1 / length(trigger_times_detm_1));
fprintf('U₂ system: SETM has %d more triggers than DETM (%.1f%% more)\n', ...
    diff_u2, 100 * diff_u2 / length(trigger_times_detm_2));
fprintf('Total: SETM has %d more triggers than DETM (%.1f%% more)\n', ...
    diff_total, 100 * diff_total / (length(trigger_times_detm_1) + length(trigger_times_detm_2)));
fprintf('==========================================\n\n');

fprintf('Comparison plot created successfully!\n');

all_trigger_indices_detm = sort([trigger_indices_detm_1, trigger_indices_detm_2]);
all_trigger_indices_setm = sort([trigger_indices_setm_1, trigger_indices_setm_2]);

plot_TriggerIntervalDistribution(t, all_trigger_indices_detm, all_trigger_indices_setm, './figs');

function plot_TriggerIntervalDistribution(t, trigger_indices_detm, trigger_indices_setm, fig_dir)

    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end

    ETCtimes_DETM = length(trigger_indices_detm);
    trigger_intervals_DETM = zeros(1, ETCtimes_DETM);
    for i = 2:ETCtimes_DETM
        trigger_intervals_DETM(i-1) = t(trigger_indices_detm(i)) - t(trigger_indices_detm(i-1));
    end
    trigger_intervals_DETM(trigger_intervals_DETM == 0) = [];
    sorted_DETM = sort(trigger_intervals_DETM);

    ETCtimes_SETM = length(trigger_indices_setm);
    trigger_intervals_SETM = zeros(1, ETCtimes_SETM);
    for i = 2:ETCtimes_SETM
        trigger_intervals_SETM(i-1) = t(trigger_indices_setm(i)) - t(trigger_indices_setm(i-1));
    end
    trigger_intervals_SETM(trigger_intervals_SETM == 0) = [];
    sorted_SETM = sort(trigger_intervals_SETM);

    fig_dist = figure('Position', [100, 100, 1200, 500]);

    percentiles_DETM = linspace(0, 100, length(sorted_DETM));
    percentiles_SETM = linspace(0, 100, length(sorted_SETM));

    plot(percentiles_DETM, sorted_DETM, 'b-', 'LineWidth', 1.5, 'DisplayName', 'DETM (Dynamic)');
    hold on;

    plot(percentiles_SETM, sorted_SETM, 'r--', 'LineWidth', 1.5, 'DisplayName', 'SETM (Static)');

    ax = gca;
    ax.FontSize = 12;

    xlabel('Percentile', 'Interpreter', 'latex', 'FontSize', 20);
    ylabel('Trigger Interval (s)', 'Interpreter', 'latex', 'FontSize', 20);
    title('DETM vs SETM: Trigger Interval Distribution (Sorted)','Interpreter', 'latex', 'FontSize', 28);

    legend('Location', 'northwest', 'Interpreter', 'latex', 'FontSize', 12);

    grid on;

    set(gca, 'YScale', 'log');

    hold off;

    exportgraphics(fig_dist, fullfile(fig_dir, 'fig6_interval_distribution.png'), 'Resolution', 300);
    fprintf('Figure 6 saved: Trigger Interval Distribution (Sorted)\n');
end

function save_trigger_panel(times, intervals, etc_count, color, outpath, panelPos, xlim_vals)

    fig = figure('Color', 'w', 'Position', panelPos);
    hold on;
    if ~isempty(times)
        scatter(times, intervals, 15, color);
        xlim(xlim_vals);
        for i = 1:etc_count
            line([times(i), times(i)], [0, intervals(i)]);
        end
    end
    ax = gca; box on; ax.LineWidth = 1; ax.FontSize = 11;
    xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 28);
    hold off;
    exportgraphics(fig, [outpath '.png'], 'Resolution', 300);
    fprintf('Panel saved: %s.png\n', outpath);
    close(fig);
end
