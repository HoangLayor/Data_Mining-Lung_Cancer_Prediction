from datetime import datetime
from airflow import DAG
from docker.types import Mount 
from airflow.providers.docker.operators.docker import DockerOperator
import os
from airflow.models import Variable

TRAINING_DIR = Variable.get(
    "HOST_TRAINING_DIR", 
    default_var=os.environ.get("HOST_TRAINING_DIR", "/tmp")
)

with DAG(dag_id="xgboost_training", start_date=datetime(2025, 1, 1), schedule=None, catchup=False) as dag:
    train_task = DockerOperator(
        task_id="train_task",
        image="datamining/xgboost-train:latest",
        api_version="auto",
        auto_remove=True,
        network_mode="datamining_data_network",
        docker_url="tcp://docker-proxy:2375",
        mount_tmp_dir=False,
        mounts=[
            Mount(
                source=f"{TRAINING_DIR}",
                target="/training/code",
                type='bind',
            )
        ],
        working_dir="/training/code",
        command='python airflow/dags/xgboost/train.py',
        environment={
            "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            "POSTGRES_USER": os.environ.get("POSTGRES_USER", "postgres"),
            "POSTGRES_PASSWORD": os.environ.get("POSTGRES_PASSWORD", "123456"),
            "POSTGRES_HOST": "db",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": os.environ.get("POSTGRES_DB", "lung_cancer_db"),
        }
    )
