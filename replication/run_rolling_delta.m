% =====================================================================
%  run_rolling_delta.m  --  replication of the rolling delta-hat paths
%  (paper, Section 7.5, Figure on the XLP/AAPL reconciliation).
%
%  Re-runs the rolling-window ESTIMATION (no forecasting) for a chosen
%  set of tickers, persisting the MF2V-GARCH parameter vector at every
%  21-day re-estimation origin. Reproduces the figure data showing that
%  the rolling delta-hat can be strongly positive in information-rich
%  episodes even when the full-sample estimate is at the delta = 0
%  boundary (XLP), and conversely (AAPL).
%
%  Protocol identical to run_oos_evaluation.m. Also records the
%  delta-unconstrained estimate (delta in [-5, 5]) from the constrained
%  optimum plus deterministic negative-delta starts.
%
%  Output: replication/output/rolling_delta_<TICKER>.csv
%  RUNTIME: ~0.5-1 hour per ticker (127 origins x MultiStart(10)).
% =====================================================================

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..', 'functions')));
outDir = fullfile(here, 'output');
if ~exist(outDir, 'dir'), mkdir(outDir); end

TICKERS = ["XLP", "AAPL"];        % as in the paper's figure

OOS_START_YEAR = 2015;  REEST_STEP = 21;  MAX_S = 168;
HV_WINDOW = 2520;  L_vol = 252;  BURNIN = 504;  NSTARTS = 10;

A  = [1 0.5 1 0 0 0 0; 0 0 0 0 1 0 1];
b  = [0.999; 0.999];
lb_c = [0 -0.5 0 0 0 0 0];  ub_c = [1 0.5 0.999 Inf 0.999 Inf 0.999];
lb_u = lb_c;  lb_u(6) = -5;  ub_u = ub_c;  ub_u(6) = 5;
opts = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'Display', 'off', 'OptimalityTolerance', 1e-8, ...
    'StepTolerance', 1e-12, 'FunctionTolerance', 1e-10, ...
    'MaxFunctionEvaluations', 1e6);
x0 = [0.003, 0.16, 0.84, 0.018, 0.11, 0.01, 0.87];

raw = readtable(fullfile(here, '..', 'data', '_all_returns_vol.csv'), ...
                'TextType', 'string');
if ~isdatetime(raw.OBS), raw.OBS = datetime(raw.OBS); end

for a = 1:numel(TICKERS)
    tk  = TICKERS(a);
    sub = sortrows(raw(raw.Ticker == tk, :), 'OBS');
    dates = sub.OBS;
    r     = sub.RET - mean(sub.RET);
    Vbar  = volume_normalize(sub.Volume, L_vol);
    T_all = numel(r);

    yr         = year(dates);
    first_oos  = find(yr >= OOS_START_YEAR, 1, 'first');
    min_origin = max(BURNIN + 1, HV_WINDOW + 1);
    if first_oos < min_origin, first_oos = min_origin; end
    origins = first_oos : REEST_STEP : (T_all - MAX_S);
    K       = numel(origins);
    WIN_LEN = first_oos - 1;
    origin_dates = dates(origins);
    fprintf('%s: K=%d origins, window %d\n', tk, K, WIN_LEN);

    P_con = NaN(K, 7);  P_unc = NaN(K, 7);
    parfor k = 1:K
        t0 = origins(k);
        idx_in  = (t0 - WIN_LEN + 1) : t0;
        r_in    = r(idx_in);
        Vbar_in = Vbar(idx_in);
        rng(1000 + k, 'twister');

        pc = NaN(1,7);
        try
            problem = createOptimProblem('fmincon', ...
                'objective', @(x) mf2v_garch_nll(x, r_in, Vbar_in), ...
                'x0', x0, 'Aineq', A, 'bineq', b, ...
                'lb', lb_c, 'ub', ub_c, 'options', opts);
            ms = MultiStart('UseParallel', false, 'Display', 'off');
            pc = run(ms, problem, NSTARTS);
        catch
        end

        pu = NaN(1,7);
        if all(isfinite(pc))
            try
                xs = pc;  xs(6) = max(xs(6), 1e-4);
                [pu, fu] = fmincon(@(x) mf2v_garch_nll(x, r_in, Vbar_in), xs, ...
                                   A, b, [], [], lb_u, ub_u, [], opts);
                for d0 = [-0.05 -0.01 -0.002]
                    xs2 = pc;  xs2(6) = d0;
                    [pt, ft] = fmincon(@(x) mf2v_garch_nll(x, r_in, Vbar_in), xs2, ...
                                       A, b, [], [], lb_u, ub_u, [], opts);
                    if ft < fu, pu = pt; fu = ft; end
                end
            catch
            end
        end
        P_con(k,:) = pc;  P_unc(k,:) = pu;
        fprintf('  %s %3d/%d [%s] d_con=%+.4f d_unc=%+.4f\n', tk, k, K, ...
            datestr(origin_dates(k), 'yyyy-mm-dd'), pc(6), pu(6));
    end

    Tout = table(origins(:), origin_dates(:), P_con(:,6), P_unc(:,6), ...
        'VariableNames', {'origin_idx','origin_date','delta_con','delta_unc'});
    writetable(Tout, fullfile(outDir, sprintf('rolling_delta_%s.csv', tk)));
    fprintf('SAVED rolling_delta_%s.csv\n', tk);
end
