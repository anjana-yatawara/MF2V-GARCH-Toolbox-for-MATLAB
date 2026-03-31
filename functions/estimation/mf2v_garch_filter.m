function [h, tau, sigma2, Z, V] = mf2v_garch_filter(r, Vbar, phat, m)
% MF2V_GARCH_FILTER  Filter the MF2V-GARCH-rw-m to obtain h, tau, Z.
%
%   [h, tau, sigma2, Z, V] = mf2v_garch_filter(r, Vbar, phat, m)
%
%   Inputs:
%     r      (Tx1) demeaned returns (percentage scale)
%     Vbar   (Tx1) normalized volume ratios
%     phat   [alpha gamma beta lam0 lam1 delta lam2]  (7x1)
%     m      rolling window length (default 63)
%
%   Outputs:
%     h       (Tx1) short-term component
%     tau     (Tx1) long-term component
%     sigma2  (Tx1) conditional variance h*tau
%     Z       (Tx1) standardized residuals r/sqrt(h*tau)
%     V       (Tx1) deGARCHed returns r^2/h = tau*Z^2

if nargin < 4, m = 63; end

alpha = phat(1);  gamma = phat(2);  beta = phat(3);
lam0  = phat(4);  lam1  = phat(5);
delta = phat(6);  lam2  = phat(7);
phi   = alpha + gamma/2 + beta;

T   = numel(r);
h   = ones(T, 1);
tau = ones(T, 1) * mean(r.^2);
V   = zeros(T, 1);

for t = 2:T
    h(t) = (1-phi) + (alpha + gamma*(r(t-1)<0)) * r(t-1)^2/tau(t-1) ...
          + beta*h(t-1);
    V(t-1) = r(t-1)^2 / h(t-1);
    if t > m
        tau(t) = lam0 + lam1*mean(V(t-m:t-1)) ...
               + delta*mean(Vbar(t-m:t-1)) + lam2*tau(t-1);
    else
        tau(t) = lam0 + lam2*tau(t-1);
    end
end

V(T) = r(T)^2 / h(T);
sigma2 = h .* tau;
Z = r ./ sqrt(max(sigma2, 1e-12));

end
