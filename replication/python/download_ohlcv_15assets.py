"""
download_ohlcv_15assets.py
==========================
Download daily OHLCV (Open, High, Low, Close, AdjClose, Volume) for the 15
assets used in the MF2V-GARCH paper, from 1999-01-01 to 2026-03-31 (sample
end fixed to match the paper -- do NOT extend).

High/Low are required downstream for the Corwin-Schultz spread estimator.

Output: revision/data/ohlcv_15assets.csv
        tidy columns: Ticker, Date, Open, High, Low, Close, AdjClose, Volume

Notes
-----
* auto_adjust=False so that both Close and Adj Close are returned.
* The S&P 500 index is downloaded as ^GSPC but labeled "SP500" in the output
  to match the ticker naming in the original paper data
  ("codes and past work  paper/data/_all_returns_vol.csv").
* Re-runnable: overwrites only its own output in revision/data/.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------- config ---
ROOT = Path(__file__).resolve().parents[1]          # .../revision
OUT_CSV = ROOT / "data" / "ohlcv_15assets.csv"

START = "1999-01-01"
END   = "2026-04-01"   # yfinance end is exclusive -> last obs 2026-03-31

# label used in output : yahoo symbol
ASSETS = {
    "SP500": "^GSPC",
    "XLK": "XLK", "XLF": "XLF", "XLE": "XLE", "XLV": "XLV", "XLI": "XLI",
    "XLY": "XLY", "XLP": "XLP", "XLU": "XLU", "XLB": "XLB",
    "AAPL": "AAPL", "MSFT": "MSFT", "JPM": "JPM", "XOM": "XOM", "JNJ": "JNJ",
}

COLS = ["Ticker", "Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"]


def fetch_one(label: str, symbol: str, retries: int = 2) -> pd.DataFrame | None:
    """Download one ticker; return tidy frame or None on failure."""
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                symbol, start=START, end=END,
                auto_adjust=False, actions=False,
                progress=False, threads=False,
            )
            if df is None or df.empty:
                raise RuntimeError("empty frame returned")
            # yfinance may return MultiIndex columns even for one ticker
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={"Adj Close": "AdjClose"})
            df = df.reset_index()
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            df["Ticker"] = label
            missing = [c for c in COLS if c not in df.columns]
            if missing:
                raise RuntimeError(f"missing columns {missing}")
            return df[COLS]
        except Exception as exc:                              # noqa: BLE001
            print(f"  [{label}] attempt {attempt} failed: {exc}", flush=True)
            time.sleep(2.0 * attempt)
    return None


def main() -> int:
    frames, failed = [], []
    for label, symbol in ASSETS.items():
        print(f"Downloading {label} ({symbol}) ...", flush=True)
        df = fetch_one(label, symbol)
        if df is None:
            failed.append(label)
            continue
        print(f"  {len(df)} rows  {df['Date'].iloc[0]} .. {df['Date'].iloc[-1]}",
              flush=True)
        frames.append(df)
        time.sleep(0.5)  # be polite

    if not frames:
        print("FATAL: no data downloaded.")
        return 1

    out = pd.concat(frames, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(out)} rows for {len(frames)} tickers -> {OUT_CSV}")
    if failed:
        print(f"FAILED tickers: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
