clear; clc; close all;

addpath(genpath('functions'));

T = readtable(fullfile('data', 'SP500_daily.csv'));
y   = T.LogRet * 100;
vol = T.Volume;

foptions.choice  = 'fix';
foptions.m       = 63;
foptions.L       = 252;
foptions.nStarts = 30;

[coeff, se, pval, Z, h, tau, sigma_annual, tau_annual, ...
 annual_unconditional_vola, foptions] = mf2v_garch_estimation(y, vol, foptions);

r    = y - mean(y);
Vbar = volume_normalize(vol, foptions.L);

S = 120;
[sigma2_fc, h_fc, tau_fc, sigma_annual_fc, tau_annual_fc] = ...
    mf2v_garch_forecast(r, Vbar, coeff(2:8), S, foptions.m);

figure('Color', 'w');
plot(1:S, sigma_annual_fc, 'LineWidth', 2);
xlabel('Horizon (trading days)');
ylabel('Annualized volatility (%)');
title('MF2V-GARCH multi-step volatility forecast: S&P 500');
grid on; box on;

mf2v_garch_nic(Z, h, tau, Vbar, coeff(2:8), foptions.m);
