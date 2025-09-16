FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1

# System deps often needed by numpy/pandas/scikit
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# Project
COPY . .

# Ensure launcher is executable
RUN chmod +x /app/start.sh

# Expose API + UI
EXPOSE 8000 8501

# Start both services
CMD ["/app/start.sh"]
