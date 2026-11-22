% Copyright (c) 2018 by Bo Zhuang <sdzhuangbo@hotmail.com>
% Jiangnan University, All Rights Reserved.
function v = vecfxt(fv, x, t)

    if isa(fv, 'function_handle')
        try
            v = fv(x, t);
        catch
            m = length(x);
            v = zeros(size(x));
            for i = 1:m
                v(i) = fv(x(i), t);
            end
        end
    elseif length(fv) == length(x)
        v = fv;
    else
        v = repmat(fv(1), size(x));
    end
    if ~ iscolumn(v)
        v = v(:);
    end
end
