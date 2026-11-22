% Copyright (c) 2018 by Bo Zhuang <sdzhuangbo@hotmail.com>
% Jiangnan University, All Rights Reserved.
function val = valft(fv, t)

    if isa(fv, 'function_handle')
        val = fv(t);
    else
        val = fv(1);
    end
    if ~ isscalar(val)
        val = val(1);
    end
end
