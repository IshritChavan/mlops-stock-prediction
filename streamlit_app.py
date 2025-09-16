# streamlit_app.py
from __future__ import annotations
import os
from datetime import timedelta
import requests
import pandas as pd
import streamlit as st
import yaml

DEFAULT_API = os.environ.get("MF_API_URL", "http://localhost:8000")

@st.cache_data(show_spinner=False)
def load_cfg(path: str = "config.yaml"):
    if not os.path.exists(path):
        return {"paths": {"data_dir": "data"}}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {"paths": {"data_dir": "data"}}

@st.cache_data(show_spinner=False)
def load_features(data_dir: str) -> pd.DataFrame:
    p = os.path.join(data_dir, "features.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def api_get(url: str):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def api_post(url: str):
    try:
        r = requests.post(url, timeout=15)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

# ---------------- UI ----------------
st.set_page_config(page_title="Mutual Fund Predictions", layout="wide")

cfg = load_cfg()
data_dir = cfg.get("paths", {}).get("data_dir", "data")

st.sidebar.header("Settings")
api_base = st.sidebar.text_input("API base URL", value=DEFAULT_API, help="Your FastAPI server URL")
st.sidebar.caption("Tip: export MF_API_URL=http://localhost:8000")

status_json, err = api_get(f"{api_base}/status")
if err:
    st.sidebar.error(f"/status error: {err}")
    trained_tickers = []
else:
    trained_tickers = list((status_json or {}).get("models", {}).keys())

# Fallback: infer tickers from features.csv if /status unavailable
features = load_features(data_dir)
if not trained_tickers and not features.empty and "ticker" in features.columns:
    trained_tickers = sorted(features["ticker"].dropna().unique().tolist())

st.title("📈 Mutual Fund / ETF Prediction")

if not trained_tickers:
    st.warning("No trained tickers found yet. Run ingestion + training, or start the API.")
    st.stop()

ticker = st.selectbox("Select ticker", trained_tickers, index=0)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔁 Refresh status"):
        status_json, err = api_get(f"{api_base}/status")
        if err:
            st.error(f"/status error: {err}")
with col2:
    if st.button("🧪 Trigger retrain"):
        with st.spinner("Starting retrain…"):
            _, err2 = api_post(f"{api_base}/trigger_retrain")
        if err2:
            st.error(f"/trigger_retrain error: {err2}")
        else:
            st.success("Retrain triggered! Re-check /status in a bit.")

# --- Prediction card ---
pred_json, pred_err = api_get(f"{api_base}/predict?ticker={ticker}")
pred_col, chart_col = st.columns([1, 2])

with pred_col:
    st.subheader("Prediction")
    if pred_err:
        st.error(pred_err)
    else:
        pr = pred_json or {}
        ret = pr.get("predicted_return", 0.0)
        price = pr.get("predicted_price", None)
        version = pr.get("model_version", "n/a")

        st.metric("Predicted return (next day)", f"{ret:.4%}")
        if price is not None:
            st.metric("Predicted price", f"{price:,.4f}")
        st.caption(f"Model: {version}")

        # simple rec
        decision = "Buy" if ret > 0 else ("Sell" if ret < 0 else "Hold")
        st.markdown(f"**Recommendation:** {decision}")

# --- Chart of recent price (from local features.csv) ---
with chart_col:
    st.subheader(f"{ticker} — recent adjusted_close")
    dft = features[features.get("ticker", "") == ticker].copy() if not features.empty else pd.DataFrame()
    if dft.empty or "adjusted_close" not in dft.columns:
        st.info("No local price history found (data/features.csv).")
    else:
        dft = dft.sort_values("date")
        # last ~6 months
        cutoff = dft["date"].max() - timedelta(days=180)
        dft_recent = dft[dft["date"] >= cutoff]
        st.line_chart(dft_recent.set_index("date")["adjusted_close"], height=280)

# --- Metrics & drift from /status ---
st.divider()
st.subheader("Model metrics & drift (from /status)")
if status_json:
    m = status_json.get("metrics", {}).get(ticker, {})
    d = status_json.get("drift", {}).get(ticker, {})

    mcol, dcol = st.columns(2)
    with mcol:
        st.write("**Metrics**")
        if m:
            dfm = pd.DataFrame([m]).T.rename(columns={0: "value"})
            st.dataframe(dfm)
        else:
            st.caption("No metrics reported.")
    with dcol:
        st.write("**Drift (last 30 vs. history)**")
        if d:
            dfd = pd.DataFrame([d]).T.rename(columns={0: "value"})
            st.dataframe(dfd)
        else:
            st.caption("No drift values yet.")
else:
    st.caption("Status unavailable.")
