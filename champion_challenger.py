# champion_challenger.py
from __future__ import annotations
import json, os, shutil, pickle
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import joblib  # ✅ REQUIRED for loading .pkl models

# Default models dir is ./models alongside this file
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
_CHAMPION_JSON = "champion.json"


# ---------------------------
# Internal utils
# ---------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _champion_json_path(models_dir: str) -> str:
    return os.path.join(models_dir, _CHAMPION_JSON)

def _ensure_models_dir(models_dir: Optional[str]) -> str:
    # Always return an absolute path
    mdir = os.path.abspath(models_dir or DEFAULT_MODELS_DIR)
    os.makedirs(mdir, exist_ok=True)
    return mdir

def _load_json(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def _save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------
# Public JSON read/write
# ---------------------------
def load_champion(models_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns the full champion JSON structure. Shape:
    {
      "tickers": {
        "SWPPX": {
          "path": "/abs/path/to/champion_SWPPX.pkl",
          "metrics": {...},
          "metric_name": "mse",
          "smaller_is_better": true,
          "updated_at": "..."
        },
        ...
      }
    }
    """
    mdir = _ensure_models_dir(models_dir)
    return _load_json(_champion_json_path(mdir))

def save_champion(data: Dict[str, Any], models_dir: Optional[str] = None) -> None:
    mdir = _ensure_models_dir(models_dir)
    _save_json(_champion_json_path(mdir), data)


# ---------------------------
# Convenience accessors used by the API
# ---------------------------
def list_tickers(models_dir: Optional[str] = None) -> List[str]:
    champs = load_champion(models_dir)
    return sorted((champs.get("tickers") or {}).keys())

def load_champion_index(models_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Returns a simple {ticker: absolute_model_path} map for quick lookups.
    """
    mdir = _ensure_models_dir(models_dir)
    champs = load_champion(mdir)
    out: Dict[str, str] = {}
    for tkr, entry in (champs.get("tickers") or {}).items():
        path = entry.get("path")
        if not path:
            continue
        # Normalize to absolute path
        if not os.path.isabs(path):
            path = os.path.join(mdir, os.path.basename(path))
        out[tkr] = os.path.abspath(path)
    return out

def get_champion_path(ticker: str, models_dir: Optional[str] = None) -> str:
    index = load_champion_index(models_dir)
    if ticker not in index:
        raise KeyError(f"Ticker {ticker} not found in champion index. Available: {sorted(index.keys())}")
    path = index[ticker]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Champion model file missing: {path}")
    return path

def load_champion_model(ticker: str, models_dir: Optional[str] = None):
    """
    Loads and returns the unpickled champion model for the given ticker.
    """
    path = get_champion_path(ticker, models_dir)
    try:
        return joblib.load(path)
    except Exception as e:
        # Surface a clear error for the API
        raise RuntimeError(f"Failed to load model for {ticker} from {path}: {e}")

def load_champion_info(models_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Lightweight metadata for API startup/status endpoints.
    Matches what api_server.py imports.
    """
    mdir = _ensure_models_dir(models_dir)
    index = load_champion_index(mdir)
    return {
        "tickers": sorted(index.keys()),
        "index": index,                 # {ticker: absolute_path}
        "models_dir": mdir,             # absolute dir
    }


# ---------------------------
# Promotion logic
# ---------------------------
def update_champion_if_better(
    *,
    ticker: str,
    challenger_metrics: Dict[str, float],
    challenger_path: str,
    metric_name: str = "mse",
    smaller_is_better: bool = True,
    models_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare challenger vs current champion on `metric_name`. If better, promote challenger.
    Returns a summary dict with status ('promoted' or 'kept') and current champion info.
    """
    mdir = _ensure_models_dir(models_dir)
    champs = load_champion(mdir)
    champs.setdefault("tickers", {})
    entry = champs["tickers"].get(ticker)

    challenger_score = float(challenger_metrics.get(metric_name, float("inf")))

    def _promote() -> Dict[str, Any]:
        champion_path = os.path.abspath(os.path.join(mdir, f"champion_{ticker}.pkl"))
        os.makedirs(mdir, exist_ok=True)
        shutil.copyfile(challenger_path, champion_path)
        record = {
            "path": champion_path,
            "metrics": challenger_metrics,
            "metric_name": metric_name,
            "smaller_is_better": smaller_is_better,
            "updated_at": _now_iso(),
        }
        champs["tickers"][ticker] = record
        save_champion(champs, mdir)
        return {"status": "promoted", "champion_path": champion_path, "champion_metrics": challenger_metrics}

    # No champion yet -> promote
    if entry is None:
        return _promote()

    current_score = float(entry.get("metrics", {}).get(metric_name, float("inf")))
    better = (challenger_score < current_score) if smaller_is_better else (challenger_score > current_score)

    if better:
        return _promote()
    else:
        return {
            "status": "kept",
            "champion_path": entry.get("path"),
            "champion_metrics": entry.get("metrics"),
        }
