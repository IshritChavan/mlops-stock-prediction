#!/usr/bin/env bash
set -euo pipefail

# Start FastAPI
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1 &

# Start Streamlit UI (change to frontend/app.py if that’s your entry)
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

wait -n
