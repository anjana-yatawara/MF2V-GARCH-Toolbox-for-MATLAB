"""build_instown.py -- retrieve institutional-ownership shares for the
cross-section universe via yfinance .info (field 'heldPercentInstitutions').

Ownership shares are current values at retrieval time, not historical
point-in-time values; yfinance does not provide an ownership history
(see paper, Section 8.2). The snapshot used in the paper is shipped as
data/derived/cross_section_instown.csv (retrieved June 2026), so
re-running this script will produce slightly different values.

Usage:  python build_instown.py
Input:  data/derived/cross_section_meta.csv  (ticker universe)
Output: cross_section_instown_new.csv
"""
import os
import time

import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "..", "..", "data", "derived", "cross_section_meta.csv")
OUT = os.path.join(HERE, "cross_section_instown_new.csv")

meta = pd.read_csv(META)
tickers = meta["Ticker"].dropna().unique().tolist()
rows = []
for i, tk in enumerate(tickers):
    try:
        info = yf.Ticker(tk).info
        rows.append({"Ticker": tk,
                     "inst_own": info.get("heldPercentInstitutions"),
                     "retrieved": pd.Timestamp.now().strftime("%Y-%m-%d")})
    except Exception as e:  # noqa: BLE001 - log and continue
        rows.append({"Ticker": tk, "inst_own": None, "error": str(e)[:80]})
    if (i + 1) % 25 == 0:
        print(f"{i + 1}/{len(tickers)}")
        time.sleep(1.0)

pd.DataFrame(rows).to_csv(OUT, index=False)
print("wrote", OUT)
