"""
build_liquidity_measures.py  (Revision: referee major comment 1)

Constructs the two competing information-arrival proxies at daily frequency
for the paper's 16 assets (15 published + AMZN, reinstated per decision D7):

  1. Amihud (2002) illiquidity: ILLIQ_t = |r_t| / (AdjClose_t * Volume_t) * 1e6
     -> built from the SUBMITTED data file so the sample matches exactly.
  2. Corwin & Schultz (2012) high-low spread S_t, with the paper's overnight
     adjustment and negative spreads set to zero.
     -> built from the fresh OHLCV download (high/low not in submitted file).

Raw daily series are saved; normalization (ratio to trailing 252-day mean) and
m=63 smoothing happen in MATLAB with the same volume_normalize.m used for
volume, guaranteeing identical treatment across measures.

Output: revision/data/liquidity_measures.csv
        (Ticker, Date, RET, Volume, AdjClose, Amihud, CSSpread)
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\ayatawara\Documents\1. Research next generation\MF2V-GARCH   6 11 2026")
SUBMITTED = ROOT / "codes and past work  paper" / "data" / "_all_returns_vol.csv"
OHLCV15   = ROOT / "revision" / "data" / "ohlcv_15assets.csv"
XSEC      = ROOT / "revision" / "data" / "cross_section_ohlcv.csv"
OUT       = ROOT / "revision" / "data" / "liquidity_measures.csv"

PAPER16 = ["SP500","XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLB",
           "AAPL","MSFT","JPM","XOM","JNJ","AMZN"]
CONST = 3 - 2 * np.sqrt(2)


def corwin_schultz(df):
    """df: per-ticker frame with High, Low, Close (unadjusted), ascending dates.
    Returns daily spread series (dated at the second day of each 2-day window),
    overnight-adjusted, negatives set to 0."""
    H = df["High"].to_numpy(dtype=float).copy()
    L = df["Low"].to_numpy(dtype=float).copy()
    C = df["Close"].to_numpy(dtype=float)
    n = len(df)
    S = np.full(n, np.nan)

    # overnight adjustment of day t's range relative to day t-1 close
    Ha, La = H.copy(), L.copy()
    for t in range(1, n):
        if not (np.isfinite(H[t]) and np.isfinite(L[t]) and np.isfinite(C[t-1])):
            continue
        if L[t] > C[t-1]:           # gapped up: shift range down
            gap = L[t] - C[t-1]
            Ha[t] -= gap
            La[t] -= gap
        elif H[t] < C[t-1]:         # gapped down: shift range up
            gap = C[t-1] - H[t]
            Ha[t] += gap
            La[t] += gap

    with np.errstate(divide="ignore", invalid="ignore"):
        for t in range(1, n):
            h1, l1 = Ha[t-1], La[t-1]
            h2, l2 = Ha[t],   La[t]
            if min(h1, l1, h2, l2) <= 0 or not np.all(np.isfinite([h1, l1, h2, l2])):
                continue
            if l1 <= 0 or l2 <= 0:
                continue
            beta = np.log(h1/l1)**2 + np.log(h2/l2)**2
            hi2, lo2 = max(h1, h2), min(l1, l2)
            gamma = np.log(hi2/lo2)**2
            alpha = (np.sqrt(2*beta) - np.sqrt(beta)) / CONST - np.sqrt(gamma / CONST)
            s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
            S[t] = max(s, 0.0)      # CS preferred treatment: negatives -> 0
    return S


def main():
    sub = pd.read_csv(SUBMITTED, parse_dates=["OBS"])
    sub = sub[sub["Ticker"].isin(PAPER16)].copy()
    sub = sub.sort_values(["Ticker", "OBS"]).rename(columns={"OBS": "Date"})

    # ---- Amihud from submitted data (exact sample) ----
    dollar_vol = sub["AdjClose"] * sub["Volume"]
    sub["Amihud"] = np.where(dollar_vol > 0,
                             sub["RET"].abs() / dollar_vol * 1e6, np.nan)

    # ---- OHLC sources for Corwin-Schultz ----
    ohlc = pd.read_csv(OHLCV15, parse_dates=["Date"])
    have = set(ohlc["Ticker"].unique())
    missing = [t for t in PAPER16 if t not in have]
    frames = [ohlc[ohlc["Ticker"].isin(PAPER16)]]
    if missing:
        print(f"Pulling {missing} from cross-section OHLCV ...")
        usecols = ["Ticker", "Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"]
        xs = pd.read_csv(XSEC, parse_dates=["Date"], usecols=lambda c: c in usecols)
        got = xs[xs["Ticker"].isin(missing)]
        still = set(missing) - set(got["Ticker"].unique())
        if still:
            raise SystemExit(f"OHLC missing for {still}; download separately.")
        frames.append(got)
    ohlc = pd.concat(frames, ignore_index=True).sort_values(["Ticker", "Date"])

    cs_parts = []
    for tk, g in ohlc.groupby("Ticker"):
        g = g.sort_values("Date").reset_index(drop=True)
        cs_parts.append(pd.DataFrame({"Ticker": tk, "Date": g["Date"],
                                      "CSSpread": corwin_schultz(g)}))
    cs = pd.concat(cs_parts, ignore_index=True)

    out = sub[["Ticker", "Date", "RET", "Volume", "AdjClose", "Amihud"]].merge(
        cs, on=["Ticker", "Date"], how="left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"Saved {OUT}  rows={len(out)}")
    for tk, g in out.groupby("Ticker"):
        print(f"  {tk:6s} n={len(g):5d}  Amihud non-na={g['Amihud'].notna().sum():5d} "
              f"CS non-na={g['CSSpread'].notna().sum():5d}  "
              f"CS>0 share={(g['CSSpread'] > 0).mean():.2f}  "
              f"CS mean={g['CSSpread'].mean():.5f}")


if __name__ == "__main__":
    main()
