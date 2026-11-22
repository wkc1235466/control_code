% Copyright (c) 2018 by Bo Zhuang <sdzhuangbo@hotmail.com>
% Jiangnan University, All Rights Reserved.
function solution = create_pde_sol(pde, varargin )
    
    p = inputParser;
    p.StructExpand = false;
    p.addRequired('pde', @isstruct);
    p.addParameter('T', 1.0, @isnumeric);
    p.addParameter('N', 100, @isnumeric);
    p.addParameter('L', 1.0, @isnumeric);
    p.addParameter('M', 20,  @isnumeric);
    
    parse(p, pde, varargin{:});
    
    pde = p.Results.pde;
    T = p.Results.T;
    N = p.Results.N;
    L = p.Results.L;
    M = p.Results.M;
    
    x = linspace(0, L, M + 1)'; % x as column vector
    t = linspace(0, T, N + 1);
    dx = L / M;
    dt = T / N;
    R = dt / dx^2;
    
    u = zeros(M + 1, N + 1);
    u(:,1) = vecfx(pde.ic, x);
    l2 = zeros(1, N+1);
    l2(1) = sqrt(trapz(u(:,1).^2) * dx);
    
    Ka = vecfx(pde.Ka, x);
    Va = vecfx(pde.Va, x);
    La = vecfx(pde.La, x);
    
    rKAddVh2 = R * (Ka + Va * dx / 2);
    rKSubVh2 = R * (Ka - Va * dx / 2);
    
    MD = sparse(1:M+1, 1:M+1, 1 + 2*R*Ka - dt*La);
    ML = sparse(2:M+1, 1:M, - rKAddVh2(2:M+1), M+1, M+1);
    MU = sparse(1:M, 2:M+1, - rKSubVh2(1:M), M+1, M+1);
    A = MD + ML + MU;
    
    solution = struct('u', u, 'x', x, 't', t, 'n', 1, ...
        'T', T, 'N', N, 'L', L, 'M', M, ...
        'dx', dx, 'h', dx, 'dt', dt, 'tau', dt, 'R', R, ...
        'Ka', Ka, 'Va', Va, 'La', La, 'time_varying', 0, ...
        'f', pde.f, 'bc0', pde.bc0, 'bcl', pde.bcl, 'l2', l2, ...
        'rKAddVh2', rKAddVh2, 'rKSubVh2', rKSubVh2, ...
        'A', A, 'pde', pde);
    
end
