% Copyright (c) 2018 by Bo Zhuang <sdzhuangbo@hotmail.com>
% Jiangnan University, All Rights Reserved.
function v = vecfx(fv, x)

    if isa(fv, 'function_handle')
        try
            v = fv(x);
        catch
            m = length(x);
            v = zeros(size(x));
            for i = 1:m
                v(i) = fv(x(i));
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
