% Copyright (c) 2018 by Bo Zhuang <sdzhuangbo@hotmail.com>
% Jiangnan University, All Rights Reserved.
function pde = create_pde(varargin)

p = inputParser;

p.addParameter('Ka', 1.0);
p.addParameter('Va', 0.0);
p.addParameter('La', 0.0);
p.addParameter('f', 0.0);
p.addParameter('ic', 0.0);
p.addParameter('bc0', struct('type', 'D', 'bc', 0));
p.addParameter('bcl', struct('type', 'D', 'bc', 0));

parse(p, varargin{:});

pde.Ka = p.Results.Ka;
pde.Va = p.Results.Va;
pde.La = p.Results.La;
pde.f  = p.Results.f;
pde.ic = p.Results.ic;
pde.bc0 = p.Results.bc0;
pde.bcl = p.Results.bcl;
