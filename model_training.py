"""
Model training pipeline for mutual fund NAV prediction.

Reads engineered features, splits chronologically, trains a RandomForest,
logs to MLflow, and manages champion/challenger models.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from champion_challenger import update_champion_if_better


# ---------------------------
# Config
# ---------------------------
def load_config(path: str | Path = "config.yaml") -> Dict:
    """
    Loads config.yaml and returns a dict. Provides sane defaults if the file/keys are missing.

    Canonical schema used by this file (all keys optional):
      paths:
        data_dir:   str (default: "data")
        models_dir: str (default: "models")
        mlruns_dir: str (default: "mlruns")
      model:
        random_forest:
          n_estimators:     int (default: 300)
          max_depth:        Optional[int] (default: None)
          min_samples_leaf: int (default: 2)
      min_rows_per_ticker: int (default: 40)
      tickers: Optional[List[str]]

    Compatibility shims supported:
      - paths.model_dir  -> paths.models_dir
      - paths.mlflow_dir -> paths.mlruns_dir
      - model.type=random_forest + model.params -> model.random_forest
    """
    defaults: Dict = {
        "paths": {
            "data_dir": "data",
            "models_dir": "models",
            "mlruns_dir": "mlruns",
        },
        "model": {
            "random_forest": {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_leaf": 2,
            }
        },
        "min_rows_per_ticker": 40,
        "tickers": None,
    }

    p = Path(path)
    if not p.exists():
        return defaults

    with p.open("r") as f:
        user_cfg = yaml.safe_load(f) or {}

    def merge(base, upd):
        if isinstance(base, dict) and isinstance(upd, dict):
            out = dict(base)
            for k, v in upd.items():
                out[k] = merge(base.get(k), v)
            return out
        return upd if upd is not None else base

    cfg = merge(defaults, user_cfg)

    # ---- Compatibility shims ----
    paths = cfg.get("paths", {})
    if "model_dir" in paths and "models_dir" not in paths:
        paths["models_dir"] = paths["model_dir"]
    if "mlflow_dir" in paths and "mlruns_dir" not in paths:
        paths["mlruns_dir"] = paths["mlflow_dir"]
    cfg["paths"] = paths

    m = cfg.get("model", {})
    if "random_forest" not in m:
        # Accept model.type/params style
        if isinstance(m, dict) and m.get("type") == "random_forest" and "params" in m:
            m["random_forest"] = m["params"]
    cfg["model"] = m

    # If user supplied min_samples_split but not min_samples_leaf, keep leaf default
    rf = cfg["model"].get("random_forest", {})
    if "min_samples_leaf" not in rf and "min_samples_split" in rf:
        rf["min_samples_leaf"] = defaults["model"]["random_forest"]["min_samples_leaf"]
    cfg["model"]["random_forest"] = rf

    return cfg


# ---------------------------
# Helpers
# ---------------------------
def _coerce_numeric(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    """Coerce all non-excluded columns to numeric (errors -> NaN)."""
    df = df.copy()
    for c in df.columns:
        if c in exclude:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _select_numeric_features(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Return strictly numeric feature columns excluding provided labels."""
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _clean_xy(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Replace inf with NaN, drop all-NaN cols, then drop any rows with NaN."""
    X = X.replace([np.inf, -np.inf], np.nan)
    # drop columns that are entirely NaN
    X = X.dropna(axis=1, how="all")
    # align y and drop rows with any NaN in X
    mask = ~X.isna().any(axis=1)
    X = X.loc[mask]
    y = y.loc[X.index]
    return X, y


def train_and_evaluate(
    X: pd.DataFrame, y: pd.Series, params: Dict
) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """Fit a RandomForest and return model + metrics."""
    n = len(X)
    if n < 5:
        raise ValueError(f"Not enough samples to split/train (n={n}).")

    split_idx = max(1, int(0.8 * n))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    if len(X_test) > 0:
        preds = model.predict(X_test)
        mse = float(mean_squared_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        n_train, n_test = len(X_train), len(X_test)
    else:
        # Fallback if tiny dataset after split
        preds = model.predict(X_train)
        mse = float(mean_squared_error(y_train, preds))
        r2 = float(r2_score(y_train, preds))
        n_train, n_test = len(X_train), 0

    return model, {"mse": mse, "r2": r2, "n_train": n_train, "n_test": n_test}


# ---------------------------
# Main entry
# ---------------------------
def train_models(cfg: Optional[Dict] = None) -> None:
    """
    Train one model per ticker found in features.csv (or restricted by cfg['tickers']),
    log to MLflow, and update champion/challenger.
    """
    cfg = cfg or load_config()

    data_dir = cfg["paths"]["data_dir"]
    models_dir = cfg["paths"]["models_dir"]
    mlruns_dir = cfg["paths"]["mlruns_dir"]

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(mlruns_dir, exist_ok=True)

    features_path = os.path.join(data_dir, "features.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found at {features_path}. Run data_ingestion.py first.")

    df = pd.read_csv(features_path)
    if df.empty:
        raise ValueError("features.csv is empty. Re-run ingestion/feature engineering.")

    # Ensure proper dtypes
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Must have ticker column
    if "ticker" not in df.columns:
        raise ValueError("features.csv missing 'ticker' column.")

    # Infer available tickers from data
    inferred_tickers = sorted(df["ticker"].dropna().unique().tolist())

    # Optionally filter by config tickers (only keep those that exist in data)
    cfg_tickers = cfg.get("tickers")
    if cfg_tickers:
        tickers = [t for t in cfg_tickers if t in inferred_tickers]
    else:
        tickers = inferred_tickers

    if not tickers:
        raise ValueError("No tickers found to train. Check ingestion output or config tickers filter.")

    # MLflow setup
    mlflow.set_tracking_uri("file:" + os.path.abspath(mlruns_dir))

    for ticker in tickers:
        dft = df[df["ticker"] == ticker].sort_values("date").copy()

        # Validate target
        if "target_return" not in dft.columns:
            raise KeyError("features.csv missing 'target_return' column.")

        dft = dft.dropna(subset=["target_return"])

        # Exclude non-feature columns
        exclude_cols = ["date", "ticker", "symbol", "target_return"]

        # Coerce everything else to numeric, then pick numeric features
        dft_num = _coerce_numeric(dft, exclude=exclude_cols)
        feature_cols = _select_numeric_features(dft_num, exclude=exclude_cols)

        # Ensure we have enough data to train
        if len(feature_cols) == 0 or len(dft_num) < int(cfg["min_rows_per_ticker"]):
            print(
                f"[Train] Skipping {ticker}: not enough features or rows "
                f"(rows={len(dft_num)}, features={len(feature_cols)})."
            )
            continue

        X = dft_num[feature_cols]
        y = dft_num["target_return"].astype(float)

        # Clean X,y
        X, y = _clean_xy(X, y)
        if len(X) < 5:
            print(f"[Train] Skipping {ticker}: not enough clean samples after NaN/inf filtering (n={len(X)}).")
            continue

        params = cfg["model"]["random_forest"]

        # Set experiment BEFORE starting the run
        mlflow.set_experiment(f"MF_NAV_{ticker}")
        with mlflow.start_run(run_name=f"run_{datetime.now(timezone.utc).isoformat()}"):
            # params/meta
            mlflow.log_params(params)
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("rows_raw", len(dft_num))
            mlflow.log_param("rows_clean", len(X))
            mlflow.log_param("features", ",".join(feature_cols))

            # record timespan of training data
            if "date" in dft.columns:
                try:
                    mlflow.log_param("date_min", str(dft["date"].min()))
                    mlflow.log_param("date_max", str(dft["date"].max()))
                except Exception:
                    pass

            # train + metrics
            model, metrics = train_and_evaluate(X, y, params)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Save challenger model
            model_path = os.path.join(models_dir, f"challenger_{ticker}.pkl")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)

            # Save feature list for inference alignment
            feat_list_path = os.path.join(models_dir, f"features_{ticker}.txt")
            with open(feat_list_path, "w") as f:
                f.write("\n".join(feature_cols))
            mlflow.log_artifact(feat_list_path)

            # Update champion if better (lower MSE is better)
            champion_info = update_champion_if_better(
                ticker=ticker,
                challenger_metrics=metrics,
                challenger_path=model_path,
                metric_name="mse",
                smaller_is_better=True,
                models_dir=models_dir,
            )
            print(
                f"[Train] {ticker} :: features={len(feature_cols)} rows={len(X)} "
                f"-> {metrics} :: {champion_info}"
            )

    print("[Train] Done.")


if __name__ == "__main__":
    train_models()
