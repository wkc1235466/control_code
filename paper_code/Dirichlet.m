% Copyright (c) 2018 by Bo Zhuang <sdzhuangbo@hotmail.com>
% Jiangnan University, All Rights Reserved.
function bc = Dirichlet(fbc)
    
    bc = struct('type', 'D', 'bc', fbc);
end
