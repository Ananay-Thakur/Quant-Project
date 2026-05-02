from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "XOM", "UNH", "PG",
    "HD", "MA", "BAC", "CVX", "ABBV", "PFE", "KO", "PEP", "AVGO", "COST",
    "WMT", "MRK", "DIS", "ADBE", "CSCO", "NFLX", "TMO", "MCD", "DHR", "ABT",
]


def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".yfinance_tz_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir.resolve()))

    frames = []
    for t in TICKERS:
        df = yf.download(t, period="10y", auto_adjust=False, progress=False)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.reset_index()
        if "Datetime" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        df["Ticker"] = t
        frames.append(df[["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]])

    if not frames:
        raise RuntimeError("No data downloaded. Check network and ticker availability.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["Date", "Adj Close", "Volume"]).sort_values(["Date", "Ticker"])

    out_path = out_dir / "us_largecap_10y_daily.csv"
    panel.to_csv(out_path, index=False)
    print(f"Saved {len(panel):,} rows to {out_path}")


if __name__ == "__main__":
    main()
