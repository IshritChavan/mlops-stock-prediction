"""
Sample Airflow DAG for the mutual fund prediction pipeline.

This DAG orchestrates daily ingestion of fresh price data, feature engineering,
model training, drift detection and champion management.  It uses Python
operators for each task and demonstrates how to integrate MLOps workflows
within Airflow.  To use this DAG, copy it into your Airflow DAGs folder.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

import os
import sys

# Add project path so we can import pipeline modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_ingestion import ingest_data
from model_training import train_models
from champion_challenger import load_champion_info
from drift_detection import detect_drift


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="mutual_fund_prediction_dag",
    default_args=default_args,
    description="Daily mutual fund performance prediction pipeline",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    def ingest_task():
        ingest_data(enable_kafka=False)

    def train_task():
        train_models()

    # PythonOperator for ingestion
    ingestion_op = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_task,
    )
    # PythonOperator for training
    training_op = PythonOperator(
        task_id="train_model",
        python_callable=train_task,
    )

    ingestion_op >> training_op