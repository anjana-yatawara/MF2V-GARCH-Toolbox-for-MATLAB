function [coeff, qmle_se, p_value_qmle, Z, h, tau, sigma_annual, ...
         tau_annual, annual_unconditional_vola, foptions] = ...
         mf2v_garch_estimation(y, vol, foptions)

if nargin < 3, foptions = struct(); end
if ~isfield(foptions, 'choice'),  foptions.choice  = 'fix'; end
if ~isfield(foptions, 'm'),       foptions.m       = 63;    end
if ~isfield(foptions, 'L'),       foptions.L       = 252;   end
if ~isfield(foptions, 'nStarts'), foptions.nStarts = 30;    end

T = numel(y);
mu = mean(y);
r  = y - mu;

Vbar = volume_normalize(vol, foptions.L);

if strcmpi(foptions.choice, 'BIC')
    m_grid = 20:150;
    bic_vals = NaN(numel(m_grid), 1);
    fprintf('BIC search over m = %d to %d ...\n', m_grid(1), m_grid(end));
    for mi = 1:numel(m_grid)
        m_try = m_grid(mi);
        try
            nLL = mf2v_garch_nll([0.003 0.16 0.84 0.018 0.11 0.01 0.87], ...
                                  r, Vbar, m_try);

            [~, fval] = fmincon(@(p) mf2v_garch_nll(p, r, Vbar, m_try), ...
                [0.003 0.16 0.84 0.018 0.11 0.01 0.87], ...
                [1 0.5 1 0 0 0 0; 0 0 0 0 1 0 1], [0.999; 0.999], ...
                [], [], [0 -0.5 0 0 0 0 0], [1 0.5 0.999 Inf 0.999 Inf 0.999], ...
                [], optimoptions('fmincon','Display','off'));
            T_eff = T - 504;
            bic_vals(mi) = log(T_eff)*7 + 2*fval;
        catch
        end
    end
    [~, idx] = min(bic_vals);
    foptions.m = m_grid(idx);
    fprintf('Optimal m = %d\n', foptions.m);
end

m = foptions.m;

A = [1 0.5 1 0 0 0 0; 0 0 0 0 1 0 1];
b = [0.999; 0.999];
lb = [0 -0.5 0 0 0 0 0];
ub = [1  0.5 0.999 Inf 0.999 Inf 0.999];
x0 = [0.003, 0.16, 0.84, 0.018, 0.11, 0.01, 0.87];

opts = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'Display', 'off', 'OptimalityTolerance', 1e-8, ...
    'StepTolerance', 1e-12, 'FunctionTolerance', 1e-10, ...
    'MaxFunctionEvaluations', 1e6);

problem = createOptimProblem('fmincon', ...
    'objective', @(p) mf2v_garch_nll(p, r, Vbar, m), ...
    'x0', x0, 'Aineq', A, 'bineq', b, ...
    'lb', lb, 'ub', ub, 'options', opts);

ms = MultiStart('UseParallel', true, 'Display', 'off');
[phat, fval] = run(ms, problem, foptions.nStarts);

[h, tau, sigma2, Z, V] = mf2v_garch_filter(r, Vbar, phat, m);

k = 7;
eps_h = 5e-4;
H = zeros(k);
p_eval = phat;
p_eval(p_eval < 1e-4) = 1e-4;
for i = 1:k
    for j = i:k
        pp = p_eval; pp(i) = pp(i)+eps_h; pp(j) = pp(j)+eps_h;
        pm = p_eval; pm(i) = pm(i)+eps_h; pm(j) = pm(j)-eps_h;
        mp = p_eval; mp(i) = mp(i)-eps_h; mp(j) = mp(j)+eps_h;
        mm = p_eval; mm(i) = mm(i)-eps_h; mm(j) = mm(j)-eps_h;
        H(i,j) = (mf2v_garch_nll(pp,r,Vbar,m) - mf2v_garch_nll(pm,r,Vbar,m) ...
                 - mf2v_garch_nll(mp,r,Vbar,m) + mf2v_garch_nll(mm,r,Vbar,m)) ...
                 / (4*eps_h^2);
        H(j,i) = H(i,j);
    end
end
se = sqrt(abs(diag(inv(H))));
se_mu = std(y) / sqrt(T);

qmle_se = [se_mu; se];
coeff = [mu; phat(:)];

p_value_qmle = 2*(1 - normcdf(abs(coeff ./ qmle_se)));

sigma_annual = sqrt(252 * sigma2);
tau_annual   = sqrt(252 * tau);

phi = phat(1) + phat(2)/2 + phat(3);
lam0_V = phat(4) + phat(6)*mean(Vbar(foptions.L+1:end));
E_tau  = lam0_V / (1 - phat(5) - phat(7));
annual_unconditional_vola = sqrt(252 * E_tau);

kappa = mean(Z(505:end).^4);
varphi_kappa = (phat(1) + phat(2)/2)*kappa + phat(3);
Gamma_m = (phat(5)/m)*varphi_kappa + phat(7)*phi;
for j = 2:m
    Gamma_m = Gamma_m + phat(5)*varphi_kappa*(phi^(j-1))/m;
end
foptions.Gamma_m = Gamma_m;
foptions.kappa = kappa;
foptions.LLF = -fval;
foptions.T_eff = T - 504;
foptions.BIC = log(T-504)*8 - 2*(-fval);

fprintf('\n===================== Estimation results MF2V-GARCH-rw-%d =====================\n', m);
fprintf('Log-Likelihood Function = %.3f, BIC = %.3f\n', -fval, foptions.BIC/(T-504));
fprintf('Estimated fourth moment of the innovations: kappa = %.3f\n', kappa);
fprintf('%-12s %12s %14s %10s %12s\n', 'Parameter', 'Coefficient', 'Standard Error', 'p-value', 'Significance');
fprintf('%s\n', repmat('_', 1, 62));

pnames = {'mu','alpha','gamma','beta','lambda_0','lambda_1','delta','lambda_2'};
for i = 1:8
    pv = p_value_qmle(i);
    if pv < 0.01,     sig = '***';
    elseif pv < 0.05, sig = '**';
    elseif pv < 0.10, sig = '*';
    else,             sig = '';
    end
    fprintf('{''%-10s''} %11.6f %14.6f %10.5g "%s"\n', ...
        pnames{i}, coeff(i), qmle_se(i), pv, sig);
end

fprintf('\nCovariance stationarity condition: Gamma_m = %.3f\n', Gamma_m);
fprintf('Annualized unconditional volatility = %.3f\n', annual_unconditional_vola);
fprintf('=============================================================================\n');

end
