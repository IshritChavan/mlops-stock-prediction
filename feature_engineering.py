"""
Feature engineering for mutual fund NAV prediction.

This module computes a rich set of features from raw price data, including
daily returns, rolling returns over multiple windows, volatility measures,
and the Sharpe ratio. Rolling returns calculate the average return over
overlapping periods to provide a more comprehensive view of performance.
The Sharpe ratio divides excess returns by the standard deviation of those
returns to measure risk-adjusted performance.
"""

import numpy as np
import pandas as pd


def _pick_price_column(cols: list[str]) -> str | None:
    """
    Try to find a reasonable price column from a list of columns (already lowercased).
    Preference order:
      1) adjusted_close
      2) contains both 'adj' and 'close'
      3) close
      4) contains 'price'
    """
    s = set(cols)
    if "adjusted_close" in s:
        return "adjusted_close"

    # contains both 'adj' and 'close' (e.g., 'adj close', 'adj_close')
    for c in cols:
        if "adj" in c and "close" in c:
            return c

    if "close" in s:
        return "close"

    for c in cols:
        if "price" in c:
            return c

    return None


def engineer_features(df: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Compute engineered features for a DataFrame of price data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain price history columns and either 'ticker' or 'symbol',
        plus a 'date' column or DatetimeIndex.
    risk_free_rate : float
        Daily risk-free rate used in Sharpe ratio calculation. If annual Rf is
        2.5%, use risk_free_rate = 0.025/252.

    Returns
    -------
    pd.DataFrame
        Engineered features with a target variable (next-day return).
    """
    # --------- Normalize schema defensively ---------
    df = df.copy()

    # If date is index, bring it out; ensure datetime
    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})
        elif "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])

    # Lowercase all columns to simplify matching
    df.columns = [str(c).strip().lower() for c in df.columns]

    # If using 'symbol', rename to 'ticker'
    if "ticker" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "ticker"})

    # Ensure we know which price column to use
    price_col = _pick_price_column(list(df.columns))
    if price_col is None:
        cols_preview = ", ".join(sorted(df.columns))
        raise KeyError(
            "engineer_features: could not find a price column. "
            f"Tried adjusted_close/close/price variants. Got columns: {cols_preview}"
        )

    # Ensure 'adjusted_close' exists for downstream logic
    if "adjusted_close" not in df.columns:
        df["adjusted_close"] = df[price_col]

    # Required columns present?
    for col in ["date", "ticker", "adjusted_close"]:
        if col not in df.columns:
            raise KeyError(f"engineer_features: required column missing: {col}")

    # Sort for stable rolling ops
    df = df.sort_values(["ticker", "date"])

    # --------- Feature computation ---------
    def per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()

        # Basic daily return
        g["daily_return"] = g["adjusted_close"].pct_change()

        # Rolling returns over 7, 14, 30 days
        for window in [7, 14, 30]:
            g[f"rolling_return_{window}"] = g["adjusted_close"].pct_change(periods=window)

        # Rolling volatility (std of daily returns)
        for window in [7, 14, 30]:
            g[f"volatility_{window}"] = g["daily_return"].rolling(window).std()

        # Sharpe ratio on 30-day window (annualized, 252 trading days)
        rolling_mean = g["daily_return"].rolling(30).mean()
        rolling_std = g["daily_return"].rolling(30).std()
        g["sharpe_30"] = (rolling_mean - risk_free_rate) / (rolling_std + 1e-8) * np.sqrt(252)

        # Lags
        g["lag1_close"] = g["adjusted_close"].shift(1)
        g["lag1_volume"] = g["volume"].shift(1) if "volume" in g.columns else np.nan

        # Target: next-day return
        g["target_return"] = g["daily_return"].shift(-1)

        # Drop rows with NaNs introduced by rolling/shift
        g = g.dropna()
        return g

    out = df.groupby("ticker", group_keys=False).apply(per_ticker)

    # Make 'date' a column for downstream CSVs
    out = out.reset_index(drop=True)
    return out
