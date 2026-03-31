function [sigma2_fc, h_fc, tau_fc, sigma_annual_fc, tau_annual_fc] = ...
         mf2v_garch_forecast(r, Vbar, phat, S, m)
% MF2V_GARCH_FORECAST  Multi-step forecasts for the MF2V-GARCH-rw-m.
%
%   Uses the factored forecast (Yatawara 2026, Eq. 31):
%     E[sigma^2_{t+s}|F_t] = E[h_{t+s}|F_t] * E[tau_{t+s}|F_t]
%
%   The short-term forecast follows Eq. (24):
%     E[h_{t+s}|F_t] = 1 + phi^{s-1}*(h_{t+1} - 1)
%
%   The long-term forecast follows Theorem 2 (Eqs. 27-29):
%     For s <= m: uses observed V and Vbar from the recent past.
%     For s > m:  all terms are future; volume enters as mu_Vbar.
%
%   Inputs:
%     r      (Tx1) demeaned returns (percentage scale)
%     Vbar   (Tx1) normalized volume ratios
%     phat   [alpha gamma beta lam0 lam1 delta lam2]  (7x1)
%     S      maximum forecast horizon (integer)
%     m      rolling window length (default 63)
%
%   Outputs:   (all Sx1)
%     sigma2_fc         forecasted conditional variance
%     h_fc              forecasted short-term component
%     tau_fc            forecasted long-term component
%     sigma_annual_fc   annualized volatility forecast
%     tau_annual_fc     annualized long-term volatility forecast

if nargin < 5, m = 63; end

alpha = phat(1);  gamma = phat(2);  beta = phat(3);
lam0  = phat(4);  lam1  = phat(5);
delta = phat(6);  lam2  = phat(7);
phi   = alpha + gamma/2 + beta;

% -- Filter to obtain current state --
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

% -- Observed lookbacks --
V_recent    = V(T:-1:T-m+1);          % V_T, ..., V_{T-m+1}
V_cumsum    = cumsum(V_recent);
Vbar_recent = Vbar(T:-1:T-m+1);
Vbar_cumsum = cumsum(Vbar_recent);

% -- One-step-ahead (known at time T) --
h1 = (1-phi) + (alpha + gamma*(r(T)<0)) * r(T)^2/tau(T) + beta*h(T);
tau1 = lam0 + lam1*mean(V(T-m+1:T)) + delta*mean(Vbar(T-m+1:T)) + lam2*tau(T);

% -- Allocate --
h_fc      = zeros(S, 1);
tau_fc    = zeros(S, 1);
sigma2_fc = zeros(S, 1);

h_fc(1)      = h1;
tau_fc(1)    = tau1;
sigma2_fc(1) = h1 * tau1;

% -- Recursive forecasts for s = 2, ..., S --
for s = 2:S

    % Short-term (Eq. 24)
    h_fc(s) = 1 + phi^(s-1) * (h1 - 1);

    % Expected volume average (Eq. 28)
    if s <= m
        n_obs = m - s + 1;
        E_Vbar_m = (Vbar_cumsum(n_obs) + (s-1)) / m;
    else
        E_Vbar_m = 1.0;
    end

    % Expected forecast-error average (Eq. 27)
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

    % Long-term (Theorem 2)
    tau_fc(s) = lam0 + lam1*E_Vfe_m + delta*E_Vbar_m + lam2*tau_fc(s-1);
    tau_fc(s) = max(tau_fc(s), 1e-8);
    tau_fc(s) = min(tau_fc(s), 1e4);

    % Total: factored forecast (Eq. 31)
    sigma2_fc(s) = h_fc(s) * tau_fc(s);
end

% -- Annualize --
sigma_annual_fc = sqrt(252 * sigma2_fc);
tau_annual_fc   = sqrt(252 * tau_fc);

end
