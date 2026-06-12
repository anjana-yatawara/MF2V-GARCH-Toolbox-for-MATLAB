function mf2v_garch_nic(Z, h, tau, Vbar, phat, m)

alpha = phat(1);  gamma = phat(2);  beta = phat(3);
lam0  = phat(4);  lam1  = phat(5);
delta = phat(6);  lam2  = phat(7);
phi   = alpha + gamma/2 + beta;

BURNIN = 504;
tau_eff = tau(BURNIN+1:end);
tau_p10 = prctile(tau_eff, 10);
tau_p90 = prctile(tau_eff, 90);

r_grid = linspace(-5, 5, 501)';
nR = numel(r_grid);

figure('Position', [100 100 800 550], 'Color', 'w');  hold on;

tau_vals  = [tau_p10, tau_p90];
clrs      = [0 0.55 0; 0.8 0 0];

for ti = 1:2
    tau_fix = tau_vals(ti);
    NIC = zeros(nR, 1);
    for i = 1:nR
        rt = r_grid(i);
        h_next = (1-phi) + (alpha + gamma*(rt<0)) * rt^2/tau_fix + beta;
        V_new  = rt^2 / 1;
        Vm     = V_new/m + (m-1)/m * tau_fix;
        tau_next = lam0 + lam1*Vm + delta*1.0 + lam2*tau_fix;
        NIC(i) = sqrt(252 * h_next * tau_next);
    end
    plot(r_grid, NIC, 'Color', clrs(ti,:), 'LineWidth', 2.5);
end

hold off;  xline(0, 'k-', 'HandleVisibility', 'off');
xlabel('Return r_t (%)');  ylabel('Annualized volatility (%)');
title('News Impact Curve: MF2V-GARCH by Volatility Regime');
legend({sprintf('Low tau (10th pctl, %.1f%%)', sqrt(252*tau_p10)), ...
        sprintf('High tau (90th pctl, %.1f%%)', sqrt(252*tau_p90))}, ...
       'Location', 'north');
set(gca, 'FontSize', 13);  ylim([0 inf]);  grid on;  box on;

figure('Position', [100 100 800 550], 'Color', 'w');  hold on;

Vbar_levels = [0.70, 1.00, 1.30];
vol_clrs    = [0 0.55 0; 0 0 0.8; 0.8 0 0];
tau_fix     = tau_p90;

for vi = 1:3
    NIC = zeros(nR, 1);
    for i = 1:nR
        rt = r_grid(i);
        h_next = (1-phi) + (alpha + gamma*(rt<0)) * rt^2/tau_fix + beta;
        V_new  = rt^2 / 1;
        Vm     = V_new/m + (m-1)/m * tau_fix;
        tau_next = lam0 + lam1*Vm + delta*Vbar_levels(vi) + lam2*tau_fix;
        NIC(i) = sqrt(252 * h_next * tau_next);
    end
    plot(r_grid, NIC, 'Color', vol_clrs(vi,:), 'LineWidth', 2.5);
end

NIC_mf2 = zeros(nR, 1);
for i = 1:nR
    rt = r_grid(i);
    h_next = (1-phi) + (alpha + gamma*(rt<0)) * rt^2/tau_fix + beta;
    V_new  = rt^2 / 1;
    Vm     = V_new/m + (m-1)/m * tau_fix;
    tau_next = lam0 + lam1*Vm + lam2*tau_fix;
    NIC_mf2(i) = sqrt(252 * h_next * tau_next);
end
plot(r_grid, NIC_mf2, 'k--', 'LineWidth', 1.5);

hold off;  xline(0, 'k-', 'HandleVisibility', 'off');
xlabel('Return r_t (%)');  ylabel('Annualized volatility (%)');
title(sprintf('Volume Level-Shift at High tau (90th pctl, %.1f%%)', ...
      sqrt(252*tau_p90)));
legend({'Vbar = 0.70 (Low)', 'Vbar = 1.00 (Normal)', ...
        'Vbar = 1.30 (High)', 'MF2-GARCH (delta=0)'}, ...
       'Location', 'north');
set(gca, 'FontSize', 13);  ylim([0 inf]);  grid on;  box on;

end
