"""
Streamlit front‑end for visualising the mutual fund prediction pipeline.

The UI allows users to select a ticker, view historical prices, inspect
engineered features, view predicted next‑day returns/prices and see drift
metrics.  It communicates with the FastAPI service via HTTP.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt


API_URL = os.environ.get("API_URL", "http://localhost:8000")


def get_config() -> Dict[str, Any]:
    """Read config.yaml to list available tickers."""
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml"), "r") as f:
        return yaml.safe_load(f)


def fetch_features() -> pd.DataFrame:
    """Load engineered features from disk for plotting."""
    cfg = get_config()
    data_dir = cfg.get("paths", {}).get("data_dir", "data")
    path = os.path.join(os.path.dirname(__file__), "..", data_dir, "features.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def main():
    st.set_page_config(page_title="Mutual Fund Predictor", layout="wide")
    st.title("Mutual Fund Performance Predictor")

    cfg = get_config()
    tickers = cfg.get("tickers", [])
    selected_ticker = st.selectbox("Select a mutual fund/ETF ticker", tickers)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict Next‑Day Return & Price"):
            try:
                resp = requests.get(f"{API_URL}/predict", params={"ticker": selected_ticker})
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"Predicted return: {data['predicted_return']*100:.2f}% | Predicted price: ${data['predicted_price']:.2f} \nModel version: {data['model_version']}"
                    )
                else:
                    st.error(f"Prediction API error: {resp.status_code} {resp.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")

    with col2:
        if st.button("Show Recommendations for All Tickers"):
            try:
                resp = requests.get(f"{API_URL}/recommendations")
                if resp.status_code == 200:
                    recs = resp.json()["recommendations"]
                    st.write(pd.DataFrame.from_dict(recs, orient="index"))
                else:
                    st.error("Failed to retrieve recommendations")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")

    # Plot historical adjusted close price for selected ticker
    st.subheader("Historical Adjusted Close Price")
    features_df = fetch_features()
    if not features_df.empty and selected_ticker in features_df["ticker"].unique():
        ticker_df = features_df[features_df["ticker"] == selected_ticker].copy()
        ticker_df["date"] = pd.to_datetime(ticker_df["date"])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ticker_df["date"], ticker_df["adjusted_close"], label="Adjusted Close")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.set_title(f"{selected_ticker} Adjusted Close History")
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("No historical data available. Run ingestion and training first.")

    # Display drift metrics
    st.subheader("Drift Metrics (PSI & KL Divergence)")
    try:
        status_resp = requests.get(f"{API_URL}/status")
        if status_resp.status_code == 200:
            status_data = status_resp.json()
            drift = status_data.get("drift", {}).get(selected_ticker, {})
            if drift:
                drift_df = pd.DataFrame(drift).T.reset_index().rename(columns={"index": "feature"})
                st.dataframe(drift_df)
            else:
                st.info("Drift metrics not available yet")
        else:
            st.error("Failed to fetch status from API")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")


if __name__ == "__main__":
    main()