"""
build_sorted_portfolios.py  (Revision: referee major comment 2)

Builds characteristic-sorted quintile portfolios from the 238-stock
cross-section, as the referee requested: "work on portfolios (say, deciles)
sorted on one or two such characteristics and test whether the importance of
including volume (delta) varies systematically across portfolios."

Sort characteristics (informed-trading proxies implementable at daily freq):
  SIZE     : full-sample mean daily dollar volume (AdjClose * Volume).
             (Crude but fully historical; current market cap is also saved
              for a robustness sort.)
  TURNOVER : full-sample mean of Volume / SharesOutstanding (current shares —
             caveat documented; affects levels, not cross-sectional ranks much).

Portfolios: quintiles (~47 stocks each), equal-weighted.
  Portfolio return: r_p,t = 100 * ln(1 + mean_i(simple return_i,t))
  Portfolio volume signal: Vbar_p,t = mean_i( Vbar_i,t ), where Vbar_i,t is
  each stock's OWN volume ratio to its trailing 252-day mean (scale-free, so
  cross-stock aggregation is meaningful and matches the paper's normalization).

Stocks enter a portfolio only if they have >= 6000 obs (full-history panel,
balanced from 2000; survivorship caveat documented in the paper).

Output:
  revision/data/portfolios_size.csv      (Date, Q1..Q5 returns, Q1..Q5 Vbar)
  revision/data/portfolios_turnover.csv
  revision/data/portfolio_assignments.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\ayatawara\Documents\1. Research next generation\MF2V-GARCH   6 11 2026")
XSEC = ROOT / "revision" / "data" / "cross_section_ohlcv.csv"
META = ROOT / "revision" / "data" / "cross_section_meta.csv"
OUTD = ROOT / "revision" / "data"

L = 252
MIN_OBS = 6000


def main():
    print("Loading cross-section ...")
    usecols = ["Ticker", "Date", "AdjClose", "Volume"]
    df = pd.read_csv(XSEC, parse_dates=["Date"], usecols=usecols)
    meta = pd.read_csv(META)

    counts = df.groupby("Ticker").size()
    keep = counts[counts >= MIN_OBS].index
    df = df[df["Ticker"].isin(keep)].sort_values(["Ticker", "Date"])
    print(f"{len(keep)} tickers with >= {MIN_OBS} obs (of {counts.size})")

    # restrict to the paper sample window
    df = df[(df["Date"] >= "1999-01-01") & (df["Date"] <= "2026-03-31")]

    # per-stock daily simple returns and volume ratios
    px = df.pivot(index="Date", columns="Ticker", values="AdjClose")
    vol = df.pivot(index="Date", columns="Ticker", values="Volume")
    simple_ret = px / px.shift(1) - 1.0
    vol_ma = vol.rolling(L, min_periods=L).mean()      # includes day t (matches volume_normalize.m)
    vbar = (vol / vol_ma).where(vol_ma > 0)

    # characteristics
    dollar_vol = (px * vol).mean()
    shares = meta.set_index("Ticker")["SharesOutstanding"].reindex(px.columns)
    turnover = (vol.divide(shares, axis=1)).mean()

    chars = pd.DataFrame({"DollarVol": dollar_vol, "Turnover": turnover})
    chars["MarketCap"] = meta.set_index("Ticker")["MarketCap"].reindex(px.columns)
    inst_path = OUTD / "cross_section_instown.csv"
    sorts = [("size", "SizeQ"), ("turnover", "TurnQ")]
    if inst_path.exists():
        inst = pd.read_csv(inst_path).set_index("Ticker")["HeldPctInstitutions"]
        chars["InstOwn"] = inst.reindex(px.columns)
        sorts.append(("instown", "InstQ"))
    assign = pd.DataFrame(index=chars.index)
    assign["SizeQ"] = pd.qcut(chars["DollarVol"].rank(method="first"), 5, labels=False) + 1
    assign["TurnQ"] = pd.qcut(chars["Turnover"].rank(method="first"), 5, labels=False) + 1
    if "InstOwn" in chars:
        assign["InstQ"] = pd.qcut(chars["InstOwn"].rank(method="first"), 5, labels=False) + 1
    assign = assign.join(chars)
    assign.to_csv(OUTD / "portfolio_assignments.csv", index_label="Ticker")

    for tag, qcol in sorts:
        out = pd.DataFrame(index=px.index)
        for q in range(1, 6):
            members = assign.index[assign[qcol] == q]
            # EW simple-return portfolio, then to log return in %
            rp = simple_ret[members].mean(axis=1, skipna=True)
            out[f"RET_Q{q}"] = 100 * np.log1p(rp)
            out[f"VBAR_Q{q}"] = vbar[members].mean(axis=1, skipna=True)
        out = out.dropna(subset=[f"RET_Q{q}" for q in range(1, 6)])
        out.to_csv(OUTD / f"portfolios_{tag}.csv", index_label="Date")
        print(f"portfolios_{tag}.csv: {out.shape[0]} days, "
              f"{out.index.min().date()} .. {out.index.max().date()}")
        for q in range(1, 6):
            members = assign.index[assign[qcol] == q]
            print(f"  Q{q}: {len(members)} stocks, "
                  f"median $vol {chars.loc[members,'DollarVol'].median()/1e6:.1f}M, "
                  f"median turnover {chars.loc[members,'Turnover'].median()*100:.2f}%")


if __name__ == "__main__":
    main()
