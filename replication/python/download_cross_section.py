"""
download_cross_section.py
=========================
Download daily OHLCV (1999-01-01 .. 2026-03-31, sample end fixed to match the
paper) for a broad cross-section of ~150-170 large/mid-cap US common stocks
with continuous trading history since ~2000, spanning all 11 GICS sectors.
Used for portfolio sorts in the revision.

Selection: long-standing S&P 500 members listed before ~2000 and still
trading under the same Yahoo symbol. Tickers that fail to download (twice)
or have fewer than MIN_OBS (6000) daily observations are dropped.

Also fetches sharesOutstanding and marketCap from yfinance .info per ticker.
CAVEAT: these are CURRENT values (as of download date), not historical --
they serve only as a crude size proxy.

Outputs
-------
revision/data/cross_section_ohlcv.csv
    Ticker, Date, Open, High, Low, Close, AdjClose, Volume
revision/data/cross_section_meta.csv
    Ticker, Sector, SharesOutstanding, MarketCap, FirstDate, LastDate, NObs
revision/data/cross_section_failed.csv
    Ticker, Sector, Reason   (download failures / too-short histories)
"""

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------- config ---
ROOT = Path(__file__).resolve().parents[1]          # .../revision
DATA = ROOT / "data"
OUT_OHLCV = DATA / "cross_section_ohlcv.csv"
OUT_META = DATA / "cross_section_meta.csv"
OUT_FAIL = DATA / "cross_section_failed.csv"

START = "1999-01-01"
END   = "2026-04-01"        # exclusive -> last obs 2026-03-31
MIN_OBS = 6000
SLEEP = 0.4                 # polite pause between requests (seconds)

COLS = ["Ticker", "Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"]

# GICS sector -> tickers (long-standing S&P 500 members, listed pre-~2000,
# still trading under the same Yahoo symbol as of June 2026).
UNIVERSE = {
    "Information Technology": [
        "IBM", "INTC", "CSCO", "ORCL", "AAPL", "MSFT", "TXN", "QCOM",
        "ADBE", "AMAT", "MU", "HPQ", "ADI", "KLAC", "LRCX", "INTU",
        "ADSK", "GLW", "SNPS", "CDNS", "NTAP", "WDC", "TER", "MSI",
    ],
    "Communication Services": [
        "T", "VZ", "CMCSA", "DIS", "OMC", "IPG", "EA", "TTWO", "LUMN",
    ],
    "Consumer Discretionary": [
        "HD", "MCD", "NKE", "LOW", "TGT", "F", "SBUX", "YUM", "TJX",
        "ROST", "BBY", "AZO", "GPC", "WHR", "RCL", "CCL", "HAS", "MGM",
        "LEN", "PHM", "DHI", "EBAY", "HOG", "VFC",
    ],
    "Consumer Staples": [
        "KO", "PG", "PEP", "WMT", "COST", "CL", "KMB", "GIS", "K", "HSY",
        "SYY", "KR", "ADM", "MO", "CLX", "TSN", "CPB", "CAG", "HRL", "MKC",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "HAL", "OXY", "EOG", "APA", "DVN",
        "HES", "VLO", "WMB", "OKE", "BKR",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "PNC", "TFC",
        "BK", "STT", "SCHW", "COF", "AIG", "ALL", "AFL", "PGR", "TRV",
        "CB", "CINF", "L", "MMC", "AON", "NTRS", "KEY", "FITB", "HBAN",
        "MTB", "ZION", "CMA", "BEN", "TROW", "BLK", "SPGI",
    ],
    "Health Care": [
        "JNJ", "PFE", "MRK", "ABT", "LLY", "BMY", "AMGN", "GILD", "BIIB",
        "MDT", "BSX", "BDX", "BAX", "SYK", "CVS", "UNH", "HUM", "CI",
        "CAH", "MCK", "DHR", "TMO", "A", "IDXX", "MTD", "WAT", "UHS",
    ],
    "Industrials": [
        "GE", "HON", "MMM", "CAT", "DE", "BA", "LMT", "NOC", "GD", "RTX",
        "EMR", "ETN", "ITW", "PH", "CMI", "FDX", "UNP", "CSX", "NSC",
        "LUV", "PCAR", "ROK", "DOV", "SWK", "GWW", "WM", "FAST", "MAS",
        "TXT", "JCI", "CTAS", "EXPD",
    ],
    "Materials": [
        "APD", "ECL", "SHW", "PPG", "NUE", "NEM", "FCX", "IP", "VMC",
        "AVY", "ALB", "EMN", "WY", "SEE", "MLM", "LIN",
    ],
    "Utilities": [
        "DUK", "SO", "NEE", "D", "AEP", "EXC", "XEL", "ED", "EIX", "PEG",
        "WEC", "DTE", "PPL", "AES", "FE", "AEE", "CMS", "CNP", "NI",
        "PNW", "ATO", "ES",
    ],
    "Real Estate": [
        "SPG", "PSA", "O", "AVB", "EQR", "VTR", "WELL", "BXP", "HST",
        "KIM", "FRT", "AMT", "VNO", "REG", "MAA", "ESS", "UDR", "PLD",
        "CCI", "IRM",
    ],
}


def fetch_ohlcv(symbol: str, retries: int = 2) -> pd.DataFrame | None:
    """Download daily OHLCV for one symbol; tidy frame or None."""
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                symbol, start=START, end=END,
                auto_adjust=False, actions=False,
                progress=False, threads=False,
            )
            if df is None or df.empty:
                raise RuntimeError("empty frame")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={"Adj Close": "AdjClose"}).reset_index()
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            df["Ticker"] = symbol
            return df[COLS]
        except Exception as exc:                              # noqa: BLE001
            print(f"    attempt {attempt} failed: {exc}", flush=True)
            if attempt < retries:
                time.sleep(3.0 * attempt)
    return None


def fetch_info(symbol: str) -> tuple:
    """Return (sharesOutstanding, marketCap) -- current values, crude proxy."""
    try:
        info = yf.Ticker(symbol).info or {}
        return info.get("sharesOutstanding"), info.get("marketCap")
    except Exception as exc:                                  # noqa: BLE001
        print(f"    info failed: {exc}", flush=True)
        return None, None


def main() -> int:
    t0 = time.time()
    tickers = [(t, sec) for sec, lst in UNIVERSE.items() for t in lst]
    assert len({t for t, _ in tickers}) == len(tickers), "duplicate tickers"
    print(f"Universe: {len(tickers)} candidate tickers, "
          f"{len(UNIVERSE)} sectors. MIN_OBS={MIN_OBS}", flush=True)

    frames, meta_rows, failed = [], [], []
    for i, (tic, sector) in enumerate(tickers, 1):
        print(f"[{i:3d}/{len(tickers)}] {tic} ({sector}) "
              f"[{time.time() - t0:5.0f}s]", flush=True)
        df = fetch_ohlcv(tic)
        if df is None:
            failed.append({"Ticker": tic, "Sector": sector,
                           "Reason": "download failed"})
            time.sleep(SLEEP)
            continue
        if len(df) < MIN_OBS:
            print(f"    SKIP: only {len(df)} obs (< {MIN_OBS})", flush=True)
            failed.append({"Ticker": tic, "Sector": sector,
                           "Reason": f"only {len(df)} obs"})
            time.sleep(SLEEP)
            continue
        shares, mcap = fetch_info(tic)
        frames.append(df)
        meta_rows.append({
            "Ticker": tic, "Sector": sector,
            "SharesOutstanding": shares, "MarketCap": mcap,
            "FirstDate": df["Date"].iloc[0], "LastDate": df["Date"].iloc[-1],
            "NObs": len(df),
        })
        print(f"    OK: {len(df)} obs  {df['Date'].iloc[0]} .. "
              f"{df['Date'].iloc[-1]}  mcap={mcap}", flush=True)
        time.sleep(SLEEP)

    if not frames:
        print("FATAL: nothing downloaded.")
        return 1

    DATA.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(OUT_OHLCV, index=False)
    pd.DataFrame(meta_rows).to_csv(OUT_META, index=False)
    pd.DataFrame(failed, columns=["Ticker", "Sector", "Reason"]).to_csv(
        OUT_FAIL, index=False)

    print(f"\nDone in {time.time() - t0:.0f}s.")
    print(f"  Usable tickers : {len(meta_rows)}")
    print(f"  Failed/skipped : {len(failed)}"
          + (f" -> {[f['Ticker'] for f in failed]}" if failed else ""))
    by_sec = pd.DataFrame(meta_rows).groupby("Sector").size()
    print("  Per sector:\n" + by_sec.to_string())
    print(f"  OHLCV -> {OUT_OHLCV}")
    print(f"  Meta  -> {OUT_META}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
