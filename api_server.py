"""
FastAPI service exposing endpoints for real-time inference, status reporting and
pipeline management.

Endpoints
---------
* GET  /predict?ticker=SYMBOL     – Predict the next-day return and price for a given ticker.
* GET  /status                    – Return health information, current champion models and drift metrics.
* POST /trigger_retrain           – Trigger a background training run to update models.
* GET  /recommendations           – Generate simple investment recommendations based on predictions.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from champion_challenger import (
    load_champion_info,
    load_champion_model,
    load_champion,  # full JSON (for metrics)
)
from drift_detection import detect_drift
from model_training import load_config

app = FastAPI(title="Mutual Fund Prediction API", version="0.1.1")

# ----------------------------------------------------
# Models / Schemas
# ----------------------------------------------------
class PredictionResponse(BaseModel):
    ticker: str
    predicted_return: float
    predicted_price: float | None
    model_version: str

class StatusResponse(BaseModel):
    models: Dict[str, str]                    # {ticker: model_filename}
    metrics: Dict[str, Dict[str, float]]      # {ticker: {"mse":..., "r2":...}}
    drift: Dict[str, Dict[str, float]]        # arbitrary keys from detect_drift()

class Recommendation(BaseModel):
    ticker: str
    predicted_return: float
    recommendation: str

class RecommendationsResponse(BaseModel):
    recommendations: Dict[str, Recommendation]

# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
NON_FEATURE_COLS = {"date", "ticker", "symbol", "target_return"}

def _get_cfg_and_tickers() -> Tuple[Dict, List[str]]:
    """Load config (with shims from model_training.load_config) and resolve trained tickers."""
    cfg = load_config()
    info = load_champion_info()
    trained = info["tickers"]                       # what we actually have champions for
    cfg_tickers = cfg.get("tickers")
    if cfg_tickers:
        tickers = [t.upper() for t in cfg_tickers if t.upper() in trained]
    else:
        tickers = trained
    return cfg, tickers

def _load_features_df(cfg: Dict) -> pd.DataFrame:
    features_path = os.path.join(cfg["paths"]["data_dir"], "features.csv")
    if not os.path.exists(features_path):
        raise HTTPException(status_code=500, detail="features.csv not found. Run data_ingestion.py first.")
    df = pd.read_csv(features_path)
    if "ticker" not in df.columns:
        raise HTTPException(status_code=500, detail="features.csv missing 'ticker' column.")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def _select_numeric_features(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def _load_feature_list(ticker: str, cfg: Dict) -> List[str] | None:
    """
    Return the exact feature list used at training time (models/features_<ticker>.txt),
    or None if not found.
    """
    models_dir = cfg["paths"]["models_dir"]
    path = os.path.join(models_dir, f"features_{ticker}.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            feats = [ln.strip() for ln in f.readlines() if ln.strip()]
        return feats or None
    return None

def _align_features_for_inference(latest: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """
    Reindex to the trained feature list:
      - add any missing columns (fill 0.0)
      - drop any extra columns
      - keep the trained column order
    """
    latest = latest.copy()
    for col in feature_list:
        if col not in latest.columns:
            latest[col] = 0.0
    # keep trained order and drop extras
    latest = latest[feature_list]
    # coerce to float, replace inf with nan, fillna 0
    latest = latest.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return latest

def _latest_row_features_for(ticker: str, cfg: Dict, df_all: pd.DataFrame) -> Tuple[pd.DataFrame, float | None]:
    """
    Build the latest feature row for a ticker, aligned to the trained feature list if available.
    Returns (X_latest_df, last_price_or_None).
    """
    dft = df_all[df_all["ticker"] == ticker].sort_values("date")
    if dft.empty:
        raise HTTPException(status_code=404, detail=f"No data for ticker {ticker}.")

    # Try to use the exact trained feature list for this ticker
    trained_feats = _load_feature_list(ticker, cfg)

    if trained_feats:
        latest_src = dft.iloc[[-1]].copy()
        # Keep only columns that intersect with trained features for now
        cols_present = [c for c in trained_feats if c in latest_src.columns]
        latest_partial = latest_src[cols_present] if cols_present else pd.DataFrame(index=latest_src.index)
        latest = _align_features_for_inference(latest_partial, trained_feats)
    else:
        # Fallback: select all numeric features (legacy behavior)
        feature_cols = _select_numeric_features(dft, exclude=list(NON_FEATURE_COLS))
        if not feature_cols:
            raise HTTPException(status_code=500, detail=f"No numeric features for ticker {ticker}.")
        latest = dft.iloc[[-1]][feature_cols].astype(float)
        latest = latest.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    last_price = None
    if "adjusted_close" in dft.columns:
        try:
            last_price = float(dft["adjusted_close"].iloc[-1])
        except Exception:
            last_price = None
    return latest, last_price

# ----------------------------------------------------
# Cold caches (refresh after retrain)
# ----------------------------------------------------
_config_cache, _tickers_cache = _get_cfg_and_tickers()
_features_cache = _load_features_df(_config_cache)

def _refresh_caches() -> None:
    global _config_cache, _tickers_cache, _features_cache
    _config_cache, _tickers_cache = _get_cfg_and_tickers()
    _features_cache = _load_features_df(_config_cache)

# ----------------------------------------------------
# Endpoints
# ----------------------------------------------------
@app.get("/status", response_model=StatusResponse)
def status():
    info = load_champion_info()         # {'tickers': [...], 'index': {...}, 'models_dir': ...}
    full_json = load_champion()         # full champion JSON with metrics

    # Models map: ticker -> filename
    models_map = {t: os.path.basename(info["index"].get(t, "")) for t in _tickers_cache}

    # Metrics map from champion.json (has mse & r2)
    metrics_map: Dict[str, Dict[str, float]] = {}
    tickers_obj = (full_json or {}).get("tickers", {})
    for t in _tickers_cache:
        entry = tickers_obj.get(t, {})
        m = entry.get("metrics", {}) or {}
        row = {}
        if "mse" in m: row["mse"] = float(m["mse"])
        if "r2"  in m: row["r2"]  = float(m["r2"])
        metrics_map[t] = row

    # Drift metrics (baseline = all but last 30 rows; current = last 30)
    drift_map: Dict[str, Dict[str, float]] = {}
    for t in _tickers_cache:
        dft = _features_cache[_features_cache["ticker"] == t].sort_values("date")
        if len(dft) < 40:
            continue
        baseline = dft.iloc[:-30]
        current  = dft.iloc[-30:]
        feature_cols = _select_numeric_features(dft, exclude=list(NON_FEATURE_COLS))
        if not feature_cols:
            continue
        features_dict = {c: c for c in feature_cols}
        try:
            drift_map[t] = detect_drift(baseline, current, features_dict)
        except Exception:
            # keep API healthy even if a metric blows up
            continue

    return StatusResponse(models=models_map, metrics=metrics_map, drift=drift_map)

@app.get("/predict", response_model=PredictionResponse)
def predict(ticker: str = Query(..., description="Ticker symbol for prediction")):
    ticker = ticker.upper().strip()
    if ticker not in _tickers_cache:
        # Surface helpful message with what *is* available
        raise HTTPException(status_code=404, detail=f"Model for ticker {ticker} not found. Available: {_tickers_cache}")

    # Load the champion fresh each call (prevents stale cache bugs)
    try:
        model = load_champion_model(ticker)
        latest, last_price = _latest_row_features_for(ticker, _config_cache, _features_cache)
        yhat_return = float(model.predict(latest)[0])
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Show the real reason (e.g., feature mismatch) instead of generic 500
        raise HTTPException(status_code=500, detail=f"Prediction failed for {ticker}: {e}")

    info = load_champion_info()
    model_path = info["index"].get(ticker, "") or "unknown"
    predicted_price = float(last_price * (1.0 + yhat_return)) if last_price is not None else None

    return PredictionResponse(
        ticker=ticker,
        predicted_return=yhat_return,
        predicted_price=predicted_price,
        model_version=os.path.basename(model_path) if model_path else "unknown",
    )

def _run_retraining() -> None:
    """Run the training pipeline and refresh caches."""
    from model_training import train_models
    train_models()
    _refresh_caches()

@app.post("/trigger_retrain")
def trigger_retrain():
    """Trigger a background retraining job."""
    thread = threading.Thread(target=_run_retraining, daemon=True)
    thread.start()
    return {"status": "Retraining started"}

@app.get("/recommendations", response_model=RecommendationsResponse)
def recommendations():
    recs: Dict[str, Recommendation] = {}
    for t in _tickers_cache:
        try:
            p = predict(t)  # reuse endpoint logic; raises HTTPException when missing
        except HTTPException:
            continue
        if p.predicted_return > 0.0:
            decision = "Buy"
        elif p.predicted_return < 0.0:
            decision = "Sell"
        else:
            decision = "Hold"
        recs[t] = Recommendation(ticker=t, predicted_return=p.predicted_return, recommendation=decision)
    return RecommendationsResponse(recommendations=recs)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
