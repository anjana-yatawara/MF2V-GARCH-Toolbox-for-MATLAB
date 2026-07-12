# MF2V-GARCH Toolbox for MATLAB

A MATLAB package for estimating and forecasting volatility with the
Volume-augmented MF2-GARCH (MF2V-GARCH) model proposed in:

> Yatawara, A. (2026). "Does Trading Volume Improve Long-Term Volatility
> Forecasts? Evidence from the MF2-GARCH Framework." *Journal of Forecasting*.

The MF2V-GARCH extends the MF2-GARCH of Conrad and Engle (2025) by augmenting
the long-term volatility component with smoothed trading volume.

## Model

Daily log-returns: `r_t = sigma_t * Z_t = sqrt(h_t * tau_t) * Z_t`

Short-term component (GJR-GARCH):
```
h_t = (1-phi) + (alpha + gamma*1{r<0}) * r^2_{t-1}/tau_{t-1} + beta*h_{t-1}
```

Long-term component (multiplicative error model with volume):
```
tau_t = lambda_0 + lambda_1*V^(m)_{t-1} + delta*Vbar^(m)_{t-1} + lambda_2*tau_{t-1}
```

The model nests the MF2-GARCH when `delta = 0`.

## Contents

| File | Description |
|------|-------------|
| `example_mf2v_garch.m`  | End-to-end example: estimate, forecast, plot |
| `mf2v_garch_estimation` | QMLE estimation with standard errors |
| `mf2v_garch_nll`        | Negative quasi-log-likelihood |
| `mf2v_garch_filter`     | Filter h, tau, sigma2, Z from data |
| `mf2v_garch_forecast`   | Multi-step volatility forecasts |
| `mf2v_garch_nic`        | News impact curve figures |
| `mf2v_garch_simulate`   | Simulate returns from the model (exogenous or bootstrapped volume path) |
| `volume_normalize`      | Normalize volume by its trailing average |
| `replication/`          | Scripts that regenerate the paper's out-of-sample results (see below) |

## Quick start

```matlab
addpath(genpath('functions'));

T   = readtable(fullfile('data', 'SP500_daily.csv'));
y   = T.LogRet * 100;   % daily log-returns, percentage scale
vol = T.Volume;         % daily trading volume

foptions.m = 63;
[coeff, se, pval, Z, h, tau] = mf2v_garch_estimation(y, vol, foptions);

r    = y - mean(y);
Vbar = volume_normalize(vol, 252);
sigma2_fc = mf2v_garch_forecast(r, Vbar, coeff(2:8), 120, 63);

mf2v_garch_nic(Z, h, tau, Vbar, coeff(2:8), 63);
```

Or simply run `example_mf2v_garch.m`.

## Requirements

- MATLAB R2020a or later
- Optimization Toolbox (`fmincon`)
- Global Optimization Toolbox (`MultiStart`)

## Data

`data/SP500_daily.csv` — daily S&P 500 data, columns `Date, AdjClose, Volume,
LogRet`. **`LogRet` is in decimal units** (hence the `* 100` in the quick-start
snippet above).

`data/_all_returns_vol.csv` — long format keyed by `Ticker`, columns
`Ticker, Sector, OBS, RET, Volume, AdjClose`. **`RET` is a log-return already
in percent — do NOT multiply by 100.** Example:

```matlab
T = readtable(fullfile('data', '_all_returns_vol.csv'));
idx  = strcmp(T.Ticker, 'XLK');
y    = T.RET(idx);          % already percent
vol  = T.Volume(idx);
[coeff, se, pval] = mf2v_garch_estimation(y, vol, struct('m', 63));
```

The file contains 18 tickers. The paper uses the 16 with a complete
January 2000 – March 2026 history (6,596 observations each): SP500, XLK, XLF,
XLE, XLV, XLI, XLY, XLP, XLU, XLB, AAPL, MSFT, AMZN, JPM, XOM, JNJ. XLC and
XLRE, launched after 2000, are included for completeness but are excluded
from all results reported in the paper.

Note on `foptions.choice = 'BIC'`: this convenience option performs a
single-start BIC search over `m = 20:150` and differs from the paper's
Table (grid `m = 21:7:161`, MultiStart, per-`m` LR tests); to reproduce the
paper's m-selection table use the replication scripts below.

## Replication of the paper's forecast evaluation

The `replication/` folder regenerates the paper's out-of-sample results from
the data shipped with this repository:

| Script | Reproduces | Runtime |
|--------|------------|---------|
| `replication/run_oos_evaluation.m`   | The rolling-window out-of-sample evaluation: per-asset relative QLIKE / RMSE and Diebold–Mariano statistics (HAC + Harvey–Leybourne–Newbold correction + stationary-bootstrap p-values) at all 12 horizons; the paper's cross-sectional summary table and the per-asset appendix tables | several hours |
| `replication/run_qlike_unfiltered.m` | The outlier-rule sensitivity analysis (QLIKE on the unfiltered origin set), run after `run_oos_evaluation.m` | minutes |
| `replication/run_rolling_delta.m`    | The rolling delta-hat paths for XLP and AAPL (the boundary-reconciliation figure) | ~1–2 hours |

Each script is self-contained, uses only the functions and data in this
repository, and writes its outputs to `replication/output/`. Estimation uses
`MultiStart` with per-origin seeds (`rng(1000+k)`), so results are
reproducible up to the usual `fmincon` tolerance; forecast-evaluation
statistics are seeded (`rng(42)`) and exactly reproducible given the
estimated forecasts.

The horse-race and portfolio-sort inputs are also included:
`data/derived/` ships the liquidity measures (Amihud and Corwin–Schultz,
normalized as in the paper), the three sets of sorted-portfolio return/volume
series, the 238-stock universe list, and the institutional-ownership
snapshot used in the paper. The raw OHLCV panels behind them (~180 MB) are
not shipped, but `replication/python/` contains the exact builder scripts
(yfinance) that regenerate them: `download_ohlcv_15assets.py`,
`download_cross_section.py`, `build_liquidity_measures.py`,
`build_sorted_portfolios.py`, and `build_instown.py` (ownership shares are
current values at retrieval; the shipped snapshot is the one used in the
paper; see the note in the script header and in the paper's Section 8.2).

## References

- Yatawara, A. (2026). "Does Trading Volume Improve Long-Term Volatility
  Forecasts? Evidence from the MF2-GARCH Framework." *Journal of Forecasting*.

- Conrad, C. and R. F. Engle (2025). "Modelling Volatility Cycles: The
  MF2-GARCH Model." *Journal of Applied Econometrics*, 40(4): 438-454.

- Conrad, C. and J. T. Schoelkopf (2025). "MF2-GARCH Toolbox for MATLAB."
  https://github.com/juliustheodor/mf2garch

## Contact

Anjana Yatawara, Department of Mathematics, California State University,
Bakersfield. ayatawara@csub.edu
