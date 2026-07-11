% =====================================================================
%  run_qlike_unfiltered.m  --  replication of the outlier-rule
%  sensitivity analysis (paper, Section 7.7 and Table 10 therein).
%
%  Re-scores the QLIKE comparison on the UNFILTERED origin set (every
%  origin with a complete 8-month target window; no 99th-percentile
%  outlier screen) from the per-origin checkpoints written by
%  run_oos_evaluation.m, and reports it side by side with the filtered
%  (published) evaluation. The symmetric forecast-stability rule is
%  retained in both modes.
%
%  Run AFTER run_oos_evaluation.m.
%  Output: replication/output/qlike_unfiltered_summary.csv
% =====================================================================

clear; clc;
here   = fileparts(mfilename('fullpath'));
outDir = fullfile(here, 'output');

H_LAB  = {'1d','5d','10d','15d','1m','2m','3m','4m','5m','6m','7m','8m'};
H_DAYS = [1 5 10 15 21 42 63 84 105 126 147 168];
GUARD_MAX = 1e6;
B_BOOT = 9999;

dd = dir(fullfile(outDir, 'oos_*.mat'));
assert(~isempty(dd), 'Run run_oos_evaluation.m first.');

allrows = table();
for i = 1:numel(dd)
    S  = load(fullfile(outDir, dd(i).name));
    tk = string(erase(erase(dd(i).name, 'oos_'), '.mat'));
    K  = size(S.FV, 1);

    bad = any(any(~isfinite(S.FV) | S.FV <= 0 | S.FV >= GUARD_MAX, 3), 2);
    base_f = logical(S.valid(:)) & ~bad;                    % published set
    base_u = all(isfinite(S.RL), 2) & ~bad;                 % unfiltered set

    for md = ["filtered", "unfiltered"]
        if md == "filtered", base = base_f; else, base = base_u; end
        for hh = 1:numel(H_DAYS)
            f1 = S.FV(:,hh,1); f2 = S.FV(:,hh,2); rv = S.RL(:,hh); hb = S.HB(:,hh);
            ok = base & isfinite(f1) & isfinite(f2) & isfinite(rv) & ...
                 f1>0 & f2>0 & rv>0 & isfinite(hb) & hb>0;
            n = sum(ok);
            if n < 8, continue; end
            q1 = log(f1(ok)) + rv(ok)./f1(ok);
            q2 = log(f2(ok)) + rv(ok)./f2(ok);
            qb = log(hb(ok)) + rv(ok)./hb(ok);
            [dm, p, pb] = dm_hln_boot_local(q1 - q2, ceil(H_DAYS(hh)/21), B_BOOT);
            row = table(tk, string(H_LAB{hh}), md, n, ...
                mean(q1)/mean(qb), mean(q2)/mean(qb), dm, p, pb, ...
                'VariableNames', {'Ticker','Horizon','Mode','n', ...
                'relQLIKE_MF2','relQLIKE_MF2V','DM_HLN_QL','p_HLN_QL','p_boot_QL'});
            allrows = [allrows; row]; %#ok<AGROW>
        end
    end
    fprintf('%s done\n', tk);
end

writetable(allrows, fullfile(outDir, 'qlike_unfiltered_summary.csv'));
for md = ["filtered", "unfiltered"]
    s = allrows(allrows.Mode == md, :);
    fprintf('%-10s QLIKE: wins %d/%d | HLN-sig %d/%d\n', md, ...
        sum(s.relQLIKE_MF2V < s.relQLIKE_MF2), height(s), ...
        sum(s.p_HLN_QL < 0.05), height(s));
end

% ---------------------------------------------------------------------
function [dm_hln, p_hln, p_boot] = dm_hln_boot_local(d, h_org, B)
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
