function nLL = mf2v_garch_nll(theta, r, Vbar, m)

if nargin < 4, m = 63; end

alpha = theta(1);  gamma = theta(2);  beta  = theta(3);
lam0  = theta(4);  lam1  = theta(5);
delta = theta(6);  lam2  = theta(7);

T   = numel(r);
phi = alpha + gamma/2 + beta;

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

sigma2 = h .* tau;
idx = 505:T;

if any(sigma2(idx) <= 0) || ~isreal(sigma2(idx))
    nLL = 1e12;
else
    nLL = 0.5*sum(log(2*pi) + log(sigma2(idx)) + r(idx).^2./sigma2(idx));
end

end
