function [r, h, tau, sigma2] = mf2v_garch_simulate(theta, T, Vbar, m, burnin, seed)
% MF2V_GARCH_SIMULATE  Generate simulated returns from the MF2V-GARCH-rw-m.
%
%   [r, h, tau, sigma2] = mf2v_garch_simulate(theta, T, Vbar, m, burnin, seed)
%
% Inputs
%   theta  : [alpha gamma beta lambda0 lambda1 delta lambda2]  (7 params)
%   T      : desired sample length (after burn-in)
%   Vbar   : normalized-volume path. Either
%            (a) a vector of length >= burnin + T, used as-is (exogenous
%                volume path, e.g. the empirical series from
%                volume_normalize), or
%            (b) a shorter vector, in which case a path of length
%                burnin + T is generated from it by a circular stationary
%                bootstrap (expected block length m), preserving the
%                serial dependence of the volume ratio, or
%            (c) empty/scalar 1, in which case Vbar_t = 1 for all t and
%                the model reduces to an MF2-GARCH with intercept
%                lambda0 + delta.
%   m      : rolling window length (default 63)
%   burnin : burn-in observations to discard (default 5000)
%   seed   : rng seed for reproducibility (optional)
%
% Output
%   r      : (Tx1) simulated demeaned returns (percentage scale if theta
%            was estimated on percentage returns)
%   h, tau, sigma2 : (Tx1) simulated components, after burn-in
%
% Volume is treated as exogenous to the return innovation Z_t, as in the
% paper's Assumption 1; the simulation draws Z_t ~ N(0,1) independently
% of the Vbar path.

if nargin < 6, seed = []; end
if nargin < 5 || isempty(burnin), burnin = 5000; end
if nargin < 4 || isempty(m),      m = 63;        end
if nargin < 3 || isempty(Vbar),   Vbar = 1;      end
if ~isempty(seed), rng(seed, 'twister'); end

alpha   = theta(1);
gamma   = theta(2);
beta    = theta(3);
lambda0 = theta(4);
lambda1 = theta(5);
delta   = theta(6);
lambda2 = theta(7);
phi     = alpha + gamma/2 + beta;

N = burnin + T;

% ---- volume path of length N ----
Vbar = Vbar(:);
if isscalar(Vbar)
    Vpath = ones(N, 1) * Vbar;
elseif numel(Vbar) >= N
    Vpath = Vbar(1:N);
else
    % circular stationary bootstrap (Politis-Romano), expected block m
    n  = numel(Vbar);
    p  = 1 / m;
    idx = zeros(N, 1);
    idx(1) = randi(n);
    for t = 2:N
        if rand < p, idx(t) = randi(n);
        else,        idx(t) = mod(idx(t-1), n) + 1;
        end
    end
    Vpath = Vbar(idx);
end

muV = mean(Vpath);

h     = ones(N, 1);
tau   = ones(N, 1);
V     = zeros(N, 1);
Z     = randn(N, 1);
r_sim = zeros(N, 1);

% Initialize tau at its unconditional mean (paper, Section 3):
% E[tau] = (lambda0 + delta*mu_V) / (1 - lambda1 - lambda2)
tau(1)   = (lambda0 + delta * muV) / (1 - lambda1 - lambda2);
r_sim(1) = sqrt(h(1) * tau(1)) * Z(1);

for t = 2:N
    h(t) = (1 - phi) ...
         + (alpha + gamma * (r_sim(t-1) < 0)) * (r_sim(t-1)^2 / tau(t-1)) ...
         + beta * h(t-1);

    V(t-1) = r_sim(t-1)^2 / h(t-1);

    if t > m
        tau(t) = lambda0 + lambda1 * mean(V(t-m:t-1)) ...
               + delta   * mean(Vpath(t-m:t-1)) ...
               + lambda2 * tau(t-1);
    else
        tau(t) = lambda0 + delta * muV + lambda2 * tau(t-1);
    end

    r_sim(t) = sqrt(h(t) * tau(t)) * Z(t);
end

keep   = burnin+1:N;
r      = r_sim(keep);
h      = h(keep);
tau    = tau(keep);
sigma2 = h .* tau;
end
