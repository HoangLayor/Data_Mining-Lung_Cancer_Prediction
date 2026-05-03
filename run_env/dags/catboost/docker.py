from datetime import datetime

from airflow import DAG
from docker.types import Mount 
from airflow.providers.docker.operators.docker import DockerOperator

import os
from airflow.models import Variable

TRAINING_DIR = Variable.get(
    "HOST_TRAINING_DIR", 
    default_var=os.environ.get("HOST_TRAINING_DIR", "/home/sy/Documents/PTIT/test_dm/run_env")
)
# 0 0 1 * * : schedule at 00:00 on day-of-month 1 : 1/5/2026
# 0 0 * * 0 : shcedule at 00:00 on Sunday

with DAG(dag_id="catboost_training", start_date=datetime(2026, 4, 27), schedule="0 0 1 * *") as dag:
    train_task = DockerOperator(
        task_id="train_task",
        image="bitis2004/airflow-catboost-stage:0.0.1",
        api_version="auto",
        auto_remove=True,
        network_mode="data_mining-lung_cancer_prediction_data_network",
        docker_url="tcp://docker-proxy:2375",
        mount_tmp_dir=False,
        mounts=[
            Mount(
                source=f"{TRAINING_DIR}/dags/catboost",
                target="/training/code",
                type='bind',
            )
        ],
        working_dir="/training",
        command='python code/train.py',
    )
