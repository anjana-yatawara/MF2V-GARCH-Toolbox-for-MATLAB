function Vbar = volume_normalize(vol, L)
% VOLUME_NORMALIZE  Normalize daily volume by its trailing L-day average.
%
%   Vbar = volume_normalize(vol, L)
%
%   Computes the volume ratio (Yatawara 2026, Eq. 4):
%     Vbar_t = Volume_t / ( (1/L) * sum_{j=1}^{L} Volume_{t-j} )
%
%   Inputs:
%     vol  (Tx1) raw daily trading volume
%     L    trailing window length (default 252, one trading year)
%
%   Output:
%     Vbar (Tx1) normalized volume ratios, centered around 1.
%          The first L observations are set to 1.0.

if nargin < 2, L = 252; end

T = numel(vol);
Vbar = ones(T, 1);

for t = L+1:T
    ma = mean(vol(t-L:t-1));
    if ma > 0
        Vbar(t) = vol(t) / ma;
    end
end

end
