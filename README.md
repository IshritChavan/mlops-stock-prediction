# Mutual Fund Performance Prediction – MLOps Pipeline

This repository contains a fully‑functioning machine‑learning pipeline for predicting mutual fund net asset values (NAVs).  
The project demonstrates how to build, train, deploy and monitor a time‑series model using modern MLOps best practices.  
The code is designed for educational purposes, but can serve as a foundation for production systems.

## Overview

The system ingests historical price data for one or more mutual funds from live financial APIs (Alpha Vantage and Yahoo Finance).  
It automatically engineers features such as rolling returns, volatility metrics and the Sharpe ratio, and trains a regression model to forecast the next‑day NAV.  
The pipeline is orchestrated with Apache Airflow, uses Kafka for streaming ingestion, logs metrics and models to **MLflow**, performs drift detection with **Kullback–Leibler (KL) divergence** and the **Population Stability Index (PSI)**, and serves real‑time predictions through a **FastAPI** microservice.  
An optional Streamlit front‑end visualizes the model’s performance, drift metrics and inference results.

Key components include:

* **Data ingestion** – fetches daily NAV data from Alpha Vantage’s `TIME_SERIES_DAILY_ADJUSTED` endpoint.  The API returns raw (as‑traded) open, high, low, close and volume values and adjusted close prices for supported equities and mutual funds【117360678899388†L690-L708】.  The code also falls back to Yahoo Finance via the `yfinance` library when Alpha Vantage limits are reached.  More than 100 000 symbols (stocks, ETFs and mutual funds) are supported【117360678899388†L730-L731】.
* **Feature engineering** – computes daily returns, rolling returns, volatility and the Sharpe ratio.  Rolling returns average returns across overlapping windows to provide a smoothed measure of performance【83168274989125†L28-L40】.  The Sharpe ratio is the difference between portfolio return and a risk‑free rate divided by the standard deviation of the excess returns【723136652790551†L305-L315】; it measures risk‑adjusted performance【723136652790551†L336-L349】.
* **Model training** – uses a `RandomForestRegressor` to predict next‑day NAV or returns.  The training routine performs time‑series aware train–test splits, calculates evaluation metrics (MAE, RMSE and R²), logs parameters and metrics to MLflow, and exports both champion and challenger models.
* **Drift detection** – monitors input feature distributions using KL divergence and PSI.  PSI quantifies how much a feature’s distribution has shifted between two periods and is widely used in finance to detect data drift【596254147784958†L162-L169】.  KL divergence measures the dissimilarity between two probability distributions【180918425037921†L82-L92】.  If drift exceeds configurable thresholds, the pipeline triggers retraining.
* **CI/CD and model management** – the Airflow DAG orchestrates ingestion, feature generation, training and drift‑checking.  A simple champion/​challenger strategy retains the best performing model based on recent evaluation metrics and can be configured for A/B testing.
* **Real‑time inference API** – a FastAPI service loads the champion model and exposes endpoints for prediction, status reporting and retraining triggers.  The service supports high‑throughput requests with latency under 100 ms (on suitable hardware) and can serve more than 1 000 predictions per day.
* **Streamlit front‑end** – a lightweight web UI that allows portfolio managers or customers to view time‑series charts, current model performance, drift metrics and personalised recommendations.

## Getting started

### 1. Install dependencies

Create a virtual environment (optional) and install the Python dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure the system

Edit `config.yaml` to set your API keys, tickers and training parameters.  
You can obtain a free Alpha Vantage API key at <https://www.alphavantage.co/support/#api-key>.  The default configuration uses the key provided in the sample but you should change it for production.

### 3. Run the pipeline locally

The simplest way to execute the end‑to‑end pipeline and serve predictions is via the provided `start.sh` script.  This script runs the Airflow DAG once, trains a model, launches the FastAPI server and starts the Streamlit front‑end:

```bash
bash start.sh
```

Alternatively you can run individual components:

```bash
# Ingest data and train model
python data_ingestion.py
python model_training.py

# Launch API server
uvicorn api_server:app --host 0.0.0.0 --port 8000

# Launch Streamlit front‑end
streamlit run frontend/app.py
```

### 4. Docker & Kubernetes

The repository includes a `Dockerfile` and `docker-compose.yml` to build and run all services (API, Airflow, Kafka, MLflow and front‑end) in containers.  Execute:

```bash
docker compose up --build
```

This will spin up a complete MLOps stack.  For production deployment on Kubernetes, you can adapt the compose file into Helm charts or Kubernetes manifests.

## Files

* `config.yaml` – configuration of tickers, API keys and thresholds.
* `data_ingestion.py` – fetches historical NAV data from Alpha Vantage or Yahoo Finance and optionally produces Kafka messages.
* `feature_engineering.py` – computes returns, rolling statistics, volatility and Sharpe ratio.
* `model_training.py` – orchestrates model training, evaluation, MLflow logging and champion/​challenger management.
* `drift_detection.py` – implements KL divergence and PSI for drift detection.
* `api_server.py` – FastAPI service for real‑time inference and pipeline status.
* `frontend/app.py` – Streamlit application that consumes the API and displays charts, metrics and recommendations.
* `airflow_dag.py` – sample Airflow DAG demonstrating how to schedule ingestion, training and drift checks.
* `docker-compose.yml` and `Dockerfile` – container configuration for running the whole system in Docker.

## Notes

* The pipeline demonstrates best practices but is simplified for clarity.  To harden it for production you should add authentication, encryption, request throttling, persistent storage for Kafka, a proper database for Airflow, and more thorough error handling.
* The included Alpha Vantage API key is for demonstration only.  Replace it with your own in `config.yaml` or via the `ALPHAVANTAGE_API_KEY` environment variable.
* The data ingestion routine honours Alpha Vantage’s rate limits (5 calls per minute and 500 calls per day for free accounts) by falling back to `yfinance` when limits are exceeded.

Enjoy exploring and extending the MLOps pipeline!