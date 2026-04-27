#!/bin/bash
docker build -t bitis2004/airflow-xgboost-stage:0.0.1 .
docker push bitis2004/airflow-xgboost-stage:0.0.1
