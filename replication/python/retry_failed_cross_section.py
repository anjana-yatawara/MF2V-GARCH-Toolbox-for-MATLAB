"""
retry_failed_cross_section.py
=============================
Re-attempt the tickers listed in revision/data/cross_section_failed.csv
(transient Yahoo failures happen). Successful tickers are appended to
cross_section_ohlcv.csv and cross_section_meta.csv; cross_section_failed.csv
is rewritten with the remaining failures. Re-runnable; no-op if nothing
is recoverable. Uses the download/filter logic of download_cross_section.py.
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_cross_section as dcs  # noqa: E402

DATA = dcs.DATA


def main() -> int:
    if not dcs.OUT_FAIL.exists():
        print("No failed-ticker file; nothing to do.")
        return 0
    failed = pd.read_csv(dcs.OUT_FAIL)
    if failed.empty:
        print("No failed tickers; nothing to do.")
        return 0

    ohlcv = pd.read_csv(dcs.OUT_OHLCV)
    meta = pd.read_csv(dcs.OUT_META)
    have = set(meta["Ticker"])

    new_frames, new_meta, still_failed = [], [], []
    for _, row in failed.iterrows():
        tic, sector = row["Ticker"], row["Sector"]
        if tic in have:
            print(f"{tic}: already present; skipping.")
            continue
        print(f"Retrying {tic} ({sector}) ...", flush=True)
        df = dcs.fetch_ohlcv(tic)
        if df is None:
            still_failed.append({"Ticker": tic, "Sector": sector,
                                 "Reason": "download failed (retry)"})
            continue
        if len(df) < dcs.MIN_OBS:
            still_failed.append({"Ticker": tic, "Sector": sector,
                                 "Reason": f"only {len(df)} obs"})
            continue
        shares, mcap = dcs.fetch_info(tic)
        new_frames.append(df)
        new_meta.append({
            "Ticker": tic, "Sector": sector,
            "SharesOutstanding": shares, "MarketCap": mcap,
            "FirstDate": df["Date"].iloc[0], "LastDate": df["Date"].iloc[-1],
            "NObs": len(df),
        })
        print(f"  OK: {len(df)} obs", flush=True)
        time.sleep(dcs.SLEEP)

    if new_frames:
        pd.concat([ohlcv] + new_frames, ignore_index=True).to_csv(
            dcs.OUT_OHLCV, index=False)
        pd.concat([meta, pd.DataFrame(new_meta)], ignore_index=True) \
            .sort_values("Ticker").to_csv(dcs.OUT_META, index=False)
    pd.DataFrame(still_failed, columns=["Ticker", "Sector", "Reason"]) \
        .to_csv(dcs.OUT_FAIL, index=False)

    print(f"\nRecovered: {[m['Ticker'] for m in new_meta]}")
    print(f"Still failed: {[f['Ticker'] for f in still_failed]}")
    print(f"Total usable tickers now: {len(meta) + len(new_meta)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
