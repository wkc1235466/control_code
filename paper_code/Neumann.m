% Copyright (c) 2018 by Bo Zhuang <sdzhuangbo@hotmail.com>
% Jiangnan University, All Rights Reserved.
function bc = Neumann(fbc)
    
    bc = struct('type', 'N', 'bc', fbc);
end
