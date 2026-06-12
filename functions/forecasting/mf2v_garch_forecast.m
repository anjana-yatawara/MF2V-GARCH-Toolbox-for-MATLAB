function [sigma2_fc, h_fc, tau_fc, sigma_annual_fc, tau_annual_fc] = ...
         mf2v_garch_forecast(r, Vbar, phat, S, m)

if nargin < 5, m = 63; end

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

V_recent    = V(T:-1:T-m+1);
V_cumsum    = cumsum(V_recent);
Vbar_recent = Vbar(T:-1:T-m+1);
Vbar_cumsum = cumsum(Vbar_recent);

h1 = (1-phi) + (alpha + gamma*(r(T)<0)) * r(T)^2/tau(T) + beta*h(T);
tau1 = lam0 + lam1*mean(V(T-m+1:T)) + delta*mean(Vbar(T-m+1:T)) + lam2*tau(T);

h_fc      = zeros(S, 1);
tau_fc    = zeros(S, 1);
sigma2_fc = zeros(S, 1);

h_fc(1)      = h1;
tau_fc(1)    = tau1;
sigma2_fc(1) = h1 * tau1;

for s = 2:S

    h_fc(s) = 1 + phi^(s-1) * (h1 - 1);

    if s <= m
        n_obs = m - s + 1;
        E_Vbar_m = (Vbar_cumsum(n_obs) + (s-1)) / m;
    else
        E_Vbar_m = 1.0;
    end

    if s <= m
        future_sum = 0;
        for j = 1:s-1
            future_sum = future_sum + tau_fc(j);
        end
        n_obs = m - s + 1;
        E_Vfe_m = (future_sum + V_cumsum(n_obs)) / m;
    else
        E_Vfe_m = sum(tau_fc(s-m:s-1)) / m;
    end

    tau_fc(s) = lam0 + lam1*E_Vfe_m + delta*E_Vbar_m + lam2*tau_fc(s-1);
    tau_fc(s) = max(tau_fc(s), 1e-8);
    tau_fc(s) = min(tau_fc(s), 1e4);

    sigma2_fc(s) = h_fc(s) * tau_fc(s);
end

sigma_annual_fc = sqrt(252 * sigma2_fc);
tau_annual_fc   = sqrt(252 * tau_fc);

end
