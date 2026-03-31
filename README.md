# MF2V-GARCH Toolbox for MATLAB

A MATLAB package for estimating and forecasting volatility using the
Volume-augmented Multiplicative Factor Multi-Frequency GARCH (MF2V-GARCH)
model proposed in:

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

Long-term component (MEM + volume):
```
tau_t = lambda_0 + lambda_1*V^(m)_{t-1} + delta*Vbar^(m)_{t-1} + lambda_2*tau_{t-1}
```

The model nests the MF2-GARCH when `delta = 0`.

## Functions

| Function | Description |
|----------|-------------|
| `mf2v_garch_estimation` | QMLE estimation with standard errors |
| `mf2v_garch_nll`        | Log-likelihood |
| `mf2v_garch_filter`     | Filter h, tau, sigma2, Z from data |
| `mf2v_garch_forecast`   | Multi-step volatility forecasts |
| `mf2v_garch_nic`        | News impact curve figures |
| `volume_normalize`      | Normalize volume by trailing average |

## Quick Start

```matlab
% Load data
y   = ...;  % (Tx1) daily log-returns (percentage scale)
vol = ...;  % (Tx1) daily trading volume

% Estimate
foptions.m = 63;
[coeff, se, pval, Z, h, tau] = mf2v_garch_estimation(y, vol, foptions);

% Forecast 120 days ahead
r = y - mean(y);
Vbar = volume_normalize(vol, 252);
[sigma2_fc] = mf2v_garch_forecast(r, Vbar, coeff(2:8), 120, 63);

% News impact curve
mf2v_garch_nic(Z, h, tau, Vbar, coeff(2:8), 63);
```

## Requirements

- MATLAB R2020a or later
- Optimization Toolbox (for `fmincon`)
- Global Optimization Toolbox (for `MultiStart`)

## References

- Yatawara, A. (2026). "Does Trading Volume Improve Long-Term Volatility
  Forecasts? Evidence from the MF2-GARCH Framework."

- Conrad, C. and R.F. Engle (2025). "Modelling Volatility Cycles: The
  MF2-GARCH Model." *Journal of Applied Econometrics*, 40(4): 438-454.

- Conrad, C. and J.T. Schoelkopf (2025). "MF2-GARCH Toolbox for Matlab."
  https://github.com/juliustheodor/mf2garch

## Contact

Anjana Yatawara
Department of Mathematics
California State University, Bakersfield
