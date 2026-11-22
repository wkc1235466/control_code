% Copyright (c) 2018 by Bo Zhuang <sdzhuangbo@hotmail.com>
% Jiangnan University, All Rights Reserved.
function solution = solve_pde_step(solution)

    N = solution.N;
    n = solution.n;
    if N + 1 == n
        warning('Solve over');
        return;
    end
    
    M = solution.M;
    x = solution.x;
    t = solution.t;
    dx = solution.dx;
    dt = solution.dt;
    tau = dt;
    h = dx;
    
    Ka = vecfx(solution.Ka, x);
    Va = vecfx(solution.Va, x);
    La = vecfx(solution.La, x);

    if ~( isequal(La, solution.La) && ...
            isequal(Va, solution.Va) && ...
                isequal(Ka, solution.Ka) )
        rebuildA();  % rebuild matrix A
    end

    [lb, add2A11, add2A12, add2b1] = apply_bc0(solution.bc0);
    [rb, add2Amm, add2Am2, add2bm] = apply_bcl(solution.bcl);

    MA = rb - lb + 1;
    idx = lb:rb;
    
    A = solution.A(idx, idx);
    A(1,1) = A(1,1) + add2A11;            % A(1,1)
    A(1,2) = A(1,2) + add2A12;            % A(1,2)
    A(MA, MA) = A(MA, MA) + add2Amm;      % A(MA,MA)
    A(MA, MA-1) = A(MA, MA-1) + add2Am2;  % A(MA,MA-1)
    
    fn1 = vecfxt(solution.f, x, t(n+1));
    b = solution.u(idx, n) + tau * fn1(idx);
    b(1) = b(1) + add2b1;      % b(1)
    b(MA) = b(MA) + add2bm;    % b(MA)
    
    solution.u(idx, n+1) = A \ b;

    solution.l2(n+1) = sqrt(trapz(solution.u(:,n+1).^2) * dx);
    
    solution.n = n + 1;

    return

    function rebuildA()
        if 0 == solution.time_varying
            disp('Time-varying system... auto rebuild it.');
            solution.time_varying = 1;
        end

        R = solution.R;  % R = dt / dx^2
        rKAddVh2 = R * (Ka + Va * dx / 2);
        rKSubVh2 = R * (Ka - Va * dx / 2);
        MD = sparse(1:M+1, 1:M+1, 1 + 2*R*Ka - dt*La);
        ML = sparse(2:M+1, 1:M, - rKAddVh2(2:M+1), M+1, M+1);
        MU = sparse(1:M, 2:M+1, - rKSubVh2(1:M), M+1, M+1);
        A = MD + ML + MU;

        solution.Ka = Ka;
        solution.Va = Va;
        solution.aL = La;
        solution.rKAddVh2 = rKAddVh2;
        solution.rKSubVh2 = rKSubVh2;
        solution.A = A;
    end

    function  [lb, add2A11, add2A12, add2b1] = apply_bc0(bc0)
        rKAddVh2 = solution.rKAddVh2;
        alpha_n = valft(bc0.bc, t(n+1));

        if 'D' == bc0.type % Dirichlet left
            solution.u(1, n+1)  = alpha_n;
            lb      = 2;   % update u from u(2:)
            add2A11 = 0;   % added to A(1,1)
            add2A12 = 0;   % added to A(1,2)
            add2b1  = rKAddVh2(lb) * alpha_n; % added to b(1)
        elseif 'N' == bc0.type % Neumann left
            lb      = 1;   % update u from u(1:)
            add2A11 = 0;
            add2A12 = - rKAddVh2(lb);  % to -2Ka
            add2b1  = -2*h * rKAddVh2(lb) * alpha_n;
        elseif 'R' == bc0.type % Robin left
            lb      = 1;   % update u from  u(1:)
            add2A11 = -2*h * bc0.c(1)/bc0.c(2) * rKAddVh2(lb);
            add2A12 = - rKAddVh2(lb);
            add2b1  = -2*h/bc0.c(2) * rKAddVh2(lb) * alpha_n;
        else
            error('Unexpected boundary condition.');
        end
    end

    function [rb, add2Amm, add2Am2, add2bm] = apply_bcl(bcl)
        rKSubVh2 = solution.rKSubVh2;
        beta_n = valft(bcl.bc, t(n+1));

        if 'D' == bcl.type % Dirichlet right
            solution.u(M+1,n+1) = beta_n;
            rb      = M;   % update u until u(:M)
            add2Amm = 0;   % added to A(ma,ma)
            add2Am2 = 0;   % added to A(ma,ma-1)
            add2bm  = rKSubVh2(rb) * beta_n; % added to b(ma)
        elseif 'N' == bcl.type % Neumann right
            rb      = M+1; % update u until u(:M+1)
            add2Amm = 0;
            add2Am2 = - rKSubVh2(rb); % to -2Ka
            add2bm  = 2*h * rKSubVh2(rb) * beta_n;
        elseif 'R' == bcl.type % Robin right
            rb      = M+1; % update u until u(:M+1)
            add2Amm = 2*h * bcl.c(1)/bcl.c(2) * rKSubVh2(rb);
            add2Am2 = - rKSubVh2(rb);
            add2bm  = 2*h/bcl.c(2) * rKSubVh2(rb) * beta_n;
        else
            error('Unexpected boundary condition.');
        end
    end
end
