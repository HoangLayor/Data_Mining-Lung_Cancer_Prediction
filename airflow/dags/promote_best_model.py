'''
DAG: promote_best_model
Runs after training DAGs to find the best model by Recall and promote it to Production.
'''
import os
from datetime import datetime

from airflow.decorators import dag, task

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "Lung_Cancer_Prediction"
REGISTERED_MODEL_NAME = "lung_cancer_model"
SERVING_RELOAD_URL = os.environ.get("SERVING_RELOAD_URL", "http://model-serving:8000/reload-model")

@task.virtualenv(
    task_id="find_and_promote_best_model",
    requirements=["mlflow==2.3.2"],
    system_site_packages=False,
)
def find_and_promote_best_model(
    mlflow_tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
) -> dict:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.recall > 0",
        order_by=["metrics.recall DESC"],
        max_results=10,
    )

    if not runs:
        raise ValueError("No runs found with recall metric.")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    model_type = best_run.data.tags.get("model_type", "unknown")

    recall_val = best_run.data.metrics.get("recall", 0)
    print(f"Best run: run_id={best_run_id}, model_type={model_type}, Recall={recall_val:.4f}")

    all_versions = client.search_model_versions(f"name='{registered_model_name}'")

    best_version = None
    for v in all_versions:
        if v.run_id == best_run_id:
            best_version = v
            break

    if best_version is None:
        raise ValueError(f"No model version found for run_id={best_run_id}.")

    # Archive old Production versions
    for v in all_versions:
        if v.current_stage == "Production" and v.version != best_version.version:
            print(f"Archiving old Production version: v{v.version}")
            client.transition_model_version_stage(
                name=registered_model_name,
                version=v.version,
                stage="Archived",
            )

    # Promote best version
    if best_version.current_stage != "Production":
        client.transition_model_version_stage(
            name=registered_model_name,
            version=best_version.version,
            stage="Production",
        )
        print(f"Promoted model v{best_version.version} ({model_type}) to Production!")
    else:
        print(f"Model v{best_version.version} is already in Production.")

    return {
        "promoted_version": best_version.version,
        "model_type": model_type,
        "run_id": best_run_id,
        "recall": recall_val,
    }

@task(task_id="reload_serving_model")
def reload_serving_model(serving_url: str):
    import urllib.request
    import urllib.error

    req = urllib.request.Request(serving_url, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            print(f"Serving reload response: {body}")
    except urllib.error.URLError as e:
        print(f"WARNING: Could not reach serving container at {serving_url}: {e}")

@dag(
    dag_id="promote_best_model",
    start_date=datetime(2025, 1, 1),
    schedule="0 2 * * 0", # Runs every Sunday at 02:00
    catchup=False,
    tags=["mlflow", "model-management"],
)
def promote_best_model_dag():
    promote = find_and_promote_best_model(
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=EXPERIMENT_NAME,
        registered_model_name=REGISTERED_MODEL_NAME,
    )
    reload = reload_serving_model(serving_url=SERVING_RELOAD_URL)
    promote >> reload

promote_best_model_dag()
