"""
Data ingestion for the mutual fund performance prediction pipeline.

This module fetches historical price data for mutual funds or ETFs from
Alpha Vantage and Yahoo Finance. Alpha Vantage’s `TIME_SERIES_DAILY_ADJUSTED`
endpoint returns daily open, high, low, close and volume values as well as
adjusted closing prices. Because the free API key imposes limits (5 requests
per minute and 500 per day), the code gracefully falls back to Yahoo Finance
via the `yfinance` library when the Alpha Vantage call fails or the quota is
exhausted.

Fetched data are stored as CSV files in the directory specified in
`config.yaml` (default: `data/`). After ingestion, the module can optionally
publish each record to a Kafka topic to support streaming architectures. The
Kafka integration is disabled by default and can be enabled by setting
`enable_kafka=True` in the `ingest_data` function.
"""

import os
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf
import yaml

# Kafka integration is optional.  Importing KafkaProducer only if needed.
try:
    from kafka import KafkaProducer
except ImportError:
    KafkaProducer = None  # type: ignore

from feature_engineering import engineer_features


# ---------- helpers ----------

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from the YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _pick_price_column(cols: list[str]) -> Optional[str]:
    """
    Try to identify a reasonable price column from lowercase column names.
    Preference:
      1) adjusted_close
      2) contains both 'adj' and 'close'
      3) close
      4) contains 'nav'       (some mutual funds expose NAV)
      5) contains 'price'
    """
    s = set(cols)
    if "adjusted_close" in s:
        return "adjusted_close"
    for c in cols:
        if "adj" in c and "close" in c:
            return c
    if "close" in s:
        return "close"
    for c in cols:
        if "nav" in c:
            return c
    for c in cols:
        if "price" in c:
            return c
    return None


# ---------- fetchers ----------

def fetch_alpha_vantage(symbol: str, api_key: str) -> Optional[pd.DataFrame]:
    """
    Fetch daily adjusted time series data from Alpha Vantage.

    Returns a DataFrame with columns:
    ['open','high','low','close','adjusted_close','volume'] indexed by date.
    """
    url = (
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
        f"&symbol={symbol}&outputsize=full&apikey={api_key}"
    )
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print(f"[AlphaVantage] HTTP {response.status_code} for {symbol}")
            return None
        data = response.json()
        if "Time Series (Daily)" not in data:
            # API limit or invalid symbol
            print(f"[AlphaVantage] Missing data for {symbol}: {data.get('Note') or data.get('Error Message')}")
            return None

        ts = data["Time Series (Daily)"]
        records = []
        for date_str, values in ts.items():
            records.append(
                {
                    "date": datetime.strptime(date_str, "%Y-%m-%d"),
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "adjusted_close": float(values["5. adjusted close"]),
                    "volume": float(values["6. volume"]),
                }
            )
        df = pd.DataFrame(records).sort_values("date")
        df.set_index("date", inplace=True)
        return df
    except Exception as e:
        print(f"[AlphaVantage] Exception for {symbol}: {e}")
        return None


def fetch_yahoo(symbol: str, lookback_years: int = 10) -> pd.DataFrame:
    """
    Fetch historical price data from Yahoo Finance using yfinance.

    Returns a DataFrame with canonical columns:
      ['open','high','low','close','adjusted_close','volume']
    plus 'symbol', indexed by 'date'.
    Handles the case where yfinance returns per-ticker MultiIndex columns
    like ('Adj Close','SWPPX') by mapping them to generic names.
    """
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=lookback_years * 365)

    # Keep OHLCV + Adj Close; do NOT auto-adjust so 'Adj Close' is present
    data = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
    )

    if data is None or data.empty:
        raise ValueError(f"No data returned for {symbol} from Yahoo Finance")

    # Flatten any MultiIndex and lowercase columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join([c for c in tup if c]).strip().lower() for tup in data.columns]
    else:
        data.columns = [c.lower().strip() for c in data.columns]

    df = data.reset_index()

    # yfinance typically uses 'Date'
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    else:
        df = df.rename(columns={"date": "date"})

    # Lowercase once more for safety
    df.columns = [str(c).strip().lower() for c in df.columns]

    # For some funds, yfinance produces suffixed columns per symbol like:
    # 'open_vfinx','high_vfinx','adj close_vfinx', etc. Map those to canonical names.
    sym_lower = symbol.lower()
    candidates = {
        "open": [f"open_{sym_lower}", "open"],
        "high": [f"high_{sym_lower}", "high"],
        "low":  [f"low_{sym_lower}",  "low"],
        "close": [f"close_{sym_lower}", "close"],
        "adjusted_close": [f"adj close_{sym_lower}", f"adj_close_{sym_lower}", "adj close", "adj_close", "adjusted_close"],
        "volume": [f"volume_{sym_lower}", "volume"],
        # Some data sources expose NAV — if present we can use it as price fallback
        "nav": [f"nav_{sym_lower}", "nav"],
    }

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    def pick_first(cols: list[str]) -> Optional[str]:
        for c in cols:
            if c in df.columns:
                return c
        return None

    # Map to canonical columns if present
    for canon, options in candidates.items():
        src = pick_first(options)
        if src:
            out[canon] = df[src]

    # Build final canonical schema
    # Prefer adjusted_close; fall back to close; if still missing, use nav if available
    if "adjusted_close" not in out.columns:
        if "close" in out.columns:
            out["adjusted_close"] = out["close"]
        elif "nav" in out.columns:
            out["adjusted_close"] = out["nav"]

    out["symbol"] = symbol

    # Keep only known canonical columns (those that exist)
    keep = ["date", "symbol", "open", "high", "low", "close", "adjusted_close", "volume"]
    out = out[[c for c in keep if c in out.columns]].sort_values("date")
    out.set_index("date", inplace=True)
    return out

# ---------- kafka ----------

def produce_to_kafka(df: pd.DataFrame, topic: str, bootstrap_servers: str = "localhost:9092") -> None:
    """
    Produce each row of the DataFrame to a Kafka topic. The DataFrame index must
    contain the date. Each record is encoded as JSON with an ISO formatted date.
    """
    if KafkaProducer is None:
        print("[Kafka] kafka-python not installed; skipping Kafka publishing.")
        return
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    for date, row in df.iterrows():
        message = {"date": date.isoformat()}
        message.update(row.dropna().to_dict())
        producer.send(topic, value=message)
    producer.flush()
    producer.close()


# ---------- main ingest ----------

def ingest_data(enable_kafka: bool = False) -> None:
    """
    Fetch data for all tickers specified in config.yaml, engineer features,
    save CSVs and optionally send records to Kafka.
    """
    cfg = load_config()
    api_key = cfg.get("alpha_vantage_api_key")
    tickers: List[str] = cfg.get("tickers", [])
    data_dir = cfg.get("paths", {}).get("data_dir", "data")
    os.makedirs(data_dir, exist_ok=True)
    combined_frames = []

    for symbol in tickers:
        print(f"[Ingestion] Fetching data for {symbol}...")
        df = None

        # Attempt Alpha Vantage first
        if api_key:
            df = fetch_alpha_vantage(symbol, api_key)
            # Respect API call rate (<= 5 per minute)
            time.sleep(12)

        # If Alpha Vantage fails, fall back to Yahoo
        if df is None or df.empty:
            print(f"[Ingestion] Falling back to Yahoo Finance for {symbol}")
            df = fetch_yahoo(symbol)

        # Guarantee we have a ticker column; keep all price columns intact
        df["ticker"] = symbol

        # Save raw data (keep everything)
        csv_path = os.path.join(data_dir, f"{symbol}_prices.csv")
        df.to_csv(csv_path)
        print(f"[Ingestion] Saved raw data to {csv_path}")

        # Optionally send to Kafka
        if enable_kafka:
            produce_to_kafka(df, topic="mutual_funds")

        combined_frames.append(df.reset_index())

    # Merge data for feature engineering
    if not combined_frames:
        print("[Ingestion] No data fetched; exiting.")
        return

    merged = pd.concat(combined_frames, ignore_index=True)

    # --------- SAFETY NET before feature engineering ----------
    # Normalize lowercase, ensure a price column, and guarantee 'adjusted_close'
    merged.columns = [str(c).strip().lower() for c in merged.columns]

    # Ensure 'date' column present and datetime
    if "date" not in merged.columns and "index" in merged.columns:
        merged = merged.rename(columns={"index": "date"})
    merged["date"] = pd.to_datetime(merged["date"])

    # Promote 'symbol' to 'ticker' if needed
    if "ticker" not in merged.columns and "symbol" in merged.columns:
        merged = merged.rename(columns={"symbol": "ticker"})

    # If Yahoo used a different close-like name, promote it
    price_col = _pick_price_column(list(merged.columns))
    if price_col and price_col != "adjusted_close":
        merged.rename(columns={price_col: "adjusted_close"}, inplace=True)

    # Final fallback: if still no adjusted_close but we have close
    if "adjusted_close" not in merged.columns and "close" in merged.columns:
        merged["adjusted_close"] = merged["close"]

    # ---- debug (first run) ----
    # Print once to help diagnose odd symbol schemas
    try:
        unique_cols = sorted(set(merged.columns))
        print(f"[Ingestion] Merged columns: {', '.join(unique_cols)}")
    except Exception:
        pass

    # Compute engineered features and save
    features = engineer_features(merged, risk_free_rate=cfg.get("risk_free_rate", 0.0))
    features_path = os.path.join(data_dir, "features.csv")
    features.to_csv(features_path, index=False)
    print(f"[Ingestion] Saved engineered features to {features_path}")


if __name__ == "__main__":
    ingest_data(enable_kafka=False)
