% =====================================================================
%  run_oos_evaluation.m  --  replication of the paper's out-of-sample
%  forecast evaluation (Tables 6-9 and the per-asset appendix tables)
%
%  Yatawara (2026), "Does Trading Volume Improve Long-Term Volatility
%  Forecasts? Evidence from the MF2-GARCH Framework", J. of Forecasting.
%
%  For every asset in data/_all_returns_vol.csv:
%    - rolling-window re-estimation every 21 trading days from 2015
%      (fixed window length = observations before the first origin),
%      MF2-GARCH (delta = 0) and MF2V-GARCH, MultiStart(10);
%    - factored multi-step forecasts to 168 days; 12 horizons
%      (1, 5, 10, 15 days; 1-8 months as forward monthly blocks);
%    - realized-variance proxy: sum of squared demeaned returns;
%    - historical benchmark: 10-year trailing mean of daily r^2;
%    - outlier rule: an origin is excluded when its 8-month target
%      window contains a day whose squared return exceeds the 99th
%      percentile over the evaluation span (van Dijk-Franses 2003;
%      Conrad-Engle 2025);
%    - symmetric stability rule: an origin is excluded for BOTH models
%      if either model's forecast path is non-finite, non-positive, or
%      exceeds 1e6 at any horizon (paper, Section 7.1);
%    - losses: QLIKE and squared error, relative to the benchmark;
%    - DM tests: Newey-West HAC (bandwidth max{floor(4(n/100)^(2/9)),
%      h-1}, capped at n/4), Harvey-Leybourne-Newbold correction,
%      one-sided p-values from a circular stationary bootstrap
%      (Politis-Romano 1994; block logic per Kunsch 1989), B = 9999.
%
%  Output: replication/output/oos_<TICKER>.csv   (per-asset, 12 rows)
%          replication/output/table9_summary.csv (cross-section summary)
%
%  RUNTIME: several hours (rolling re-estimation of ~127 origins x 2
%  models x 16 assets). Reduce TICKERS below for a quick check.
% =====================================================================

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..', 'functions')));

outDir = fullfile(here, 'output');
if ~exist(outDir, 'dir'), mkdir(outDir); end

% ---- protocol constants (paper, Sections 5 and 7.1) ----
OOS_START_YEAR = 2015;
REEST_STEP     = 21;
MAX_S          = 168;
m              = 63;
HV_WINDOW      = 2520;
L_vol          = 252;
BURNIN         = 504;
NSTARTS        = 10;
GUARD_MAX      = 1e6;
B_BOOT         = 9999;

H_DAYS   = [1 5 10 15 21 42 63 84 105 126 147 168];
H_START  = [1 1  1  1  1 22 43 64  85 106 127 148];
H_LAB    = {'1d','5d','10d','15d','1m','2m','3m','4m','5m','6m','7m','8m'};
nH = numel(H_DAYS);

raw = readtable(fullfile(here, '..', 'data', '_all_returns_vol.csv'), ...
                'TextType', 'string');
if ~isdatetime(raw.OBS), raw.OBS = datetime(raw.OBS); end

% The paper's 16 assets (the CSV contains extra tickers):
TICKERS = ["SP500","XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLB", ...
           "AAPL","MSFT","AMZN","JPM","XOM","JNJ"];

summary = table();

for a = 1:numel(TICKERS)
    tk  = TICKERS(a);
    sub = sortrows(raw(raw.Ticker == tk, :), 'OBS');
    if isempty(sub), fprintf('%s not in data - skipped\n', tk); continue; end
    dates = sub.OBS;
    r     = sub.RET - mean(sub.RET);
    RV    = r.^2;
    Vbar  = volume_normalize(sub.Volume, L_vol);
    T_all = numel(r);

    yr         = year(dates);
    first_oos  = find(yr >= OOS_START_YEAR, 1, 'first');
    min_origin = max(BURNIN + 1, HV_WINDOW + 1);
    if first_oos < min_origin, first_oos = min_origin; end
    last_origin = T_all - MAX_S;
    origins = first_oos : REEST_STEP : last_origin;
    K       = numel(origins);
    WIN_LEN = first_oos - 1;

    fprintf('\n[%2d/%2d] %s  T=%d  K=%d origins  window=%d\n', ...
        a, numel(TICKERS), tk, T_all, K, WIN_LEN);

    % resume from checkpoint if this asset already ran
    csvf = fullfile(outDir, sprintf('oos_%s.csv', tk));
    if isfile(csvf) && isfile(fullfile(outDir, sprintf('oos_%s.mat', tk)))
        fprintf('   checkpoint exists - skipping (delete output files to re-run)\n');
        tab = readtable(csvf);
        ckp = load(fullfile(outDir, sprintf('oos_%s.mat', tk)), 'valid');
        srow = table(tk, sum(ckp.valid), ...
            sum(tab.relQLIKE_MF2V < tab.relQLIKE_MF2), ...
            sum(tab.p_HLN_QL  < 0.05), ...
            sum(tab.p_boot_QL < 0.05), ...
            'VariableNames', {'Ticker','n_valid','QL_wins','QL_sig_HLN','QL_sig_boot'});
        summary = [summary; srow]; %#ok<AGROW>
        continue;
    end

    FV = NaN(K, nH, 2);       % forecasts: (:,:,1)=MF2, (:,:,2)=MF2V
    RL = NaN(K, nH);          % realized
    HB = NaN(K, nH);          % historical benchmark

    for k = 1:K
        t0 = origins(k);
        idx_in  = (t0 - WIN_LEN + 1) : t0;
        r_in    = r(idx_in);
        Vbar_in = Vbar(idx_in);

        rng(1000 + k, 'twister');
        p1 = local_estimate(r_in, Vbar_in, NSTARTS, true);    % MF2 (delta=0)
        p2 = local_estimate(r_in, Vbar_in, NSTARTS, false);   % MF2V

        s1 = NaN(MAX_S,1); s2 = NaN(MAX_S,1);
        if all(isfinite(p1)), s1 = mf2v_garch_forecast(r_in, Vbar_in, p1, MAX_S, m); end
        if all(isfinite(p2)), s2 = mf2v_garch_forecast(r_in, Vbar_in, p2, MAX_S, m); end

        for hh = 1:nH
            j1 = H_START(hh); j2 = H_DAYS(hh);
            fidx = t0 + j1 : t0 + j2;
            if fidx(end) <= T_all, RL(k, hh) = sum(RV(fidx)); end
            if ~any(isnan(s1)), FV(k, hh, 1) = sum(s1(j1:j2)); end
            if ~any(isnan(s2)), FV(k, hh, 2) = sum(s2(j1:j2)); end
        end
        hv0 = max(1, t0 - HV_WINDOW + 1);
        mRV = mean(RV(hv0:t0));
        HB(k, :) = (H_DAYS - H_START + 1) * mRV;

        if mod(k, 10) == 1
            fprintf('   origin %3d/%d [%s]\n', k, K, datestr(dates(t0), 'yyyy-mm-dd'));
        end
    end

    % ---- outlier rule ----
    pct99 = prctile(RV(min(origins):min(max(origins)+MAX_S, T_all)), 99);
    outlier = false(K, 1);
    for k = 1:K
        w = origins(k)+1 : min(origins(k)+MAX_S, T_all);
        if any(RV(w) > pct99), outlier(k) = true; end
    end
    valid = ~outlier & all(isfinite(RL), 2);

    % ---- symmetric stability rule (origin level) ----
    bad = any(any(~isfinite(FV) | FV <= 0 | FV >= GUARD_MAX, 3), 2);
    valid = valid & ~bad;

    % ---- losses + DM per horizon ----
    out = struct();
    for hh = 1:nH
        f1 = FV(:,hh,1); f2 = FV(:,hh,2); rv = RL(:,hh); hb = HB(:,hh);
        ok = valid & isfinite(f1) & isfinite(f2) & isfinite(rv) & ...
             f1>0 & f2>0 & rv>0 & isfinite(hb) & hb>0;
        n = sum(ok);
        q1 = log(f1(ok)) + rv(ok)./f1(ok);  q2 = log(f2(ok)) + rv(ok)./f2(ok);
        qb = log(hb(ok)) + rv(ok)./hb(ok);
        e1 = (rv(ok)-f1(ok)).^2;  e2 = (rv(ok)-f2(ok)).^2;  eb = (rv(ok)-hb(ok)).^2;
        h_org = ceil(H_DAYS(hh)/21);
        [dmq, pq, pbq] = dm_hln_boot(q1 - q2, h_org, B_BOOT);
        [dms, ps, pbs] = dm_hln_boot(e1 - e2, h_org, B_BOOT);

        out.Horizon(hh,1) = string(H_LAB{hh});    out.n(hh,1) = n;
        out.relQLIKE_MF2(hh,1)  = mean(q1)/mean(qb);
        out.relQLIKE_MF2V(hh,1) = mean(q2)/mean(qb);
        out.relRMSE_MF2(hh,1)   = sqrt(mean(e1)/mean(eb));
        out.relRMSE_MF2V(hh,1)  = sqrt(mean(e2)/mean(eb));
        out.DM_HLN_QL(hh,1) = dmq;  out.p_HLN_QL(hh,1) = pq;  out.p_boot_QL(hh,1) = pbq;
        out.DM_HLN_SE(hh,1) = dms;  out.p_HLN_SE(hh,1) = ps;  out.p_boot_SE(hh,1) = pbs;
    end
    tab = struct2table(out);
    writetable(tab, fullfile(outDir, sprintf('oos_%s.csv', tk)));

    % per-origin checkpoint (used by run_qlike_unfiltered.m and for resume)
    origin_dates = dates(origins);
    save(fullfile(outDir, sprintf('oos_%s.mat', tk)), ...
        'FV', 'RL', 'HB', 'origins', 'origin_dates', 'valid', 'outlier', ...
        'WIN_LEN', 'tk');

    srow = table(tk, sum(valid), ...
        sum(tab.relQLIKE_MF2V < tab.relQLIKE_MF2), ...
        sum(tab.p_HLN_QL  < 0.05), ...
        sum(tab.p_boot_QL < 0.05), ...
        'VariableNames', {'Ticker','n_valid','QL_wins','QL_sig_HLN','QL_sig_boot'});
    summary = [summary; srow]; %#ok<AGROW>
    fprintf('   %s: n=%d wins=%d/12 sig=%d/12\n', tk, sum(valid), ...
        srow.QL_wins, srow.QL_sig_HLN);
end

writetable(summary, fullfile(outDir, 'table9_summary.csv'));
fprintf('\nTotals: wins %d/192, HLN-sig %d/192, boot-sig %d/192\n', ...
    sum(summary.QL_wins), sum(summary.QL_sig_HLN), sum(summary.QL_sig_boot));

% =====================================================================
function p = local_estimate(r, Vbar, nStarts, fix_delta_zero)
    % QMLE via MultiStart fmincon; MF2 baseline = delta fixed at zero.
    A  = [1 0.5 1 0 0 0 0; 0 0 0 0 1 0 1];
    b  = [0.999; 0.999];
    lb = [0 -0.5 0 0 0 0 0];
    ub = [1  0.5 0.999 Inf 0.999 Inf 0.999];
    x0 = [0.003, 0.16, 0.84, 0.018, 0.11, 0.01, 0.87];
    if fix_delta_zero, lb(6) = 0; ub(6) = 0; x0(6) = 0; end
    opts = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
        'Display', 'off', 'OptimalityTolerance', 1e-8, ...
        'StepTolerance', 1e-12, 'FunctionTolerance', 1e-10, ...
        'MaxFunctionEvaluations', 1e6);
    problem = createOptimProblem('fmincon', ...
        'objective', @(x) mf2v_garch_nll(x, r, Vbar), ...
        'x0', x0, 'Aineq', A, 'bineq', b, 'lb', lb, 'ub', ub, 'options', opts);
    ms = MultiStart('UseParallel', true, 'Display', 'off');
    try
        p = run(ms, problem, nStarts);
    catch
        p = NaN(1, 7);
    end
end

function [dm_hln, p_hln, p_boot] = dm_hln_boot(d, h_org, B)
    % DM with NW-HAC variance, HLN correction, stationary-bootstrap p.
    d = d(:); n = numel(d);
    if n < 8 || all(d == 0), [dm_hln, p_hln, p_boot] = deal(NaN); return; end
    dbar = mean(d);
    J = max(floor(4*(n/100)^(2/9)), h_org - 1);
    J = min([J, n-2, max(1, floor(n/4))]);
    e = d - dbar;  lrv = (e'*e)/n;
    for j = 1:J
        lrv = lrv + 2*(1 - j/(J+1)) * (e(1+j:end)'*e(1:end-j))/n;
    end
    lrv = max(lrv, 1e-300);
    dm_hac = dbar / sqrt(lrv/n);
    hh = max(h_org, 1);
    dm_hln = dm_hac * sqrt((n + 1 - 2*hh + hh*(hh-1)/n) / n);
    p_hln  = 1 - tcdf(dm_hln, n - 1);

    rng(42, 'twister');
    Lexp = min(max(h_org + 1, ceil(n^(1/3))), max(2, floor(n/4)));
    pgeo = 1/Lexp;  cnt = 0;  e_c = d - dbar;
    for bb = 1:B
        idx = zeros(n,1);  idx(1) = randi(n);
        for t = 2:n
            if rand < pgeo, idx(t) = randi(n);
            else,           idx(t) = mod(idx(t-1), n) + 1;
            end
        end
        db = e_c(idx);  dbb = mean(db);  eb2 = db - dbb;
        lrb = (eb2'*eb2)/n;
        for j = 1:J
            lrb = lrb + 2*(1 - j/(J+1)) * (eb2(1+j:end)'*eb2(1:end-j))/n;
        end
        lrb = max(lrb, 1e-300);
        if dbb / sqrt(lrb/n) >= dm_hac, cnt = cnt + 1; end
    end
    p_boot = (cnt + 1) / (B + 1);
end
