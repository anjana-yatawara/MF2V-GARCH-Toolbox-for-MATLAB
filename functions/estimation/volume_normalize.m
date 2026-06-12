function Vbar = volume_normalize(vol, L)

if nargin < 2, L = 252; end

T = numel(vol);
Vbar = ones(T, 1);

for t = L+1:T
    ma = mean(vol(t-L+1:t));
    if ma > 0
        Vbar(t) = vol(t) / ma;
    end
end

end
