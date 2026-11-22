% Copyright (c) 2018 by Bo Zhuang <sdzhuangbo@hotmail.com>
% Jiangnan University, All Rights Reserved.
function bc = Robin(c0, c1, fbc)
    
    assert(c1 ~= 0, 'Robin BC: `c1` should not be zero');
    bc = struct('type', 'R', 'c', [c0, c1], 'bc', fbc);
end
