"""
DAG: promote_best_model
Chạy sau khi tất cả các DAG train hoàn thành (catboost, xgboost, random_forest).
Nhiệm vụ:
  1. Query MLflow experiment "Car_Price_Prediction" để tìm run có R² cao nhất
  2. Lấy model version đã được register tương ứng với run đó
  3. Promote model đó lên stage "Production"
  4. Archive tất cả các version cũ đang ở Production

Schedule: Chạy vào ngày 1 hàng tháng, SAU các DAG train (delay 2 giờ).
"""
import os
import logging
from datetime import datetime

from airflow.decorators import dag, task

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "Car_Price_Prediction"
REGISTERED_MODEL_NAME = "car_price_model"
SERVING_RELOAD_URL = os.environ.get("SERVING_RELOAD_URL", "http://model-serving:8000/reload-model")


# Dùng @task.virtualenv để Airflow tự tạo venv riêng và cài mlflow vào đó.
# Airflow worker KHÔNG cần cài mlflow trước.
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
    """
    Chạy trong virtualenv riêng (có mlflow).
    Tìm run R² cao nhất → promote version tương ứng lên Production.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    # --- Bước 1: Tìm experiment ---
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found. "
            f"Hãy chạy ít nhất 1 DAG train trước."
        )

    print(f"Found experiment: {experiment_name} (id={experiment.experiment_id})")

    # --- Bước 2: Lấy run tốt nhất theo R² ---
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.r2 > 0",
        order_by=["metrics.r2 DESC"],
        max_results=10,
    )

    if not runs:
        raise ValueError(
            "Không tìm thấy run nào có metric r2. Hãy chạy các DAG train trước."
        )

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    model_type = best_run.data.tags.get("model_type", "unknown")

    r2_val = best_run.data.metrics.get("r2", 0)
    rmse_val = best_run.data.metrics.get("rmse", 0)
    print(
        f"Best run: run_id={best_run_id}, model_type={model_type}, "
        f"R2={r2_val:.4f}, RMSE={rmse_val:,.0f}"
    )

    # --- Bước 3: Tìm model version tương ứng với run tốt nhất ---
    all_versions = client.search_model_versions(f"name='{registered_model_name}'")

    best_version = None
    for v in all_versions:
        if v.run_id == best_run_id:
            best_version = v
            break

    if best_version is None:
        raise ValueError(
            f"Không tìm thấy model version nào ứng với run_id={best_run_id}. "
            f"Hãy đảm bảo train script có gọi mlflow.sklearn.log_model "
            f"với registered_model_name='{registered_model_name}'."
        )

    print(
        f"Found model version: v{best_version.version} "
        f"(current stage: {best_version.current_stage})"
    )

    # --- Bước 4: Archive các version Production cũ ---
    for v in all_versions:
        if v.current_stage == "Production" and v.version != best_version.version:
            print(f"Archiving old Production version: v{v.version}")
            client.transition_model_version_stage(
                name=registered_model_name,
                version=v.version,
                stage="Archived",
            )

    # --- Bước 5: Promote version tốt nhất lên Production ---
    if best_version.current_stage != "Production":
        client.transition_model_version_stage(
            name=registered_model_name,
            version=best_version.version,
            stage="Production",
        )
        print(f"Promoted model v{best_version.version} ({model_type}) to Production!")
    else:
        print(f"Model v{best_version.version} is already in Production. No change.")

    # --- Bước 6: Trả về summary (XCom) ---
    summary = {
        "promoted_version": best_version.version,
        "model_type": model_type,
        "run_id": best_run_id,
        "r2": best_run.data.metrics.get("r2"),
        "rmse": best_run.data.metrics.get("rmse"),
        "mae": best_run.data.metrics.get("mae"),
        "mape": best_run.data.metrics.get("mape"),
    }

    print("\n=== PROMOTION SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


@task(task_id="reload_serving_model")
def reload_serving_model(serving_url: str):
    """
    Gọi POST /reload-model trên serving container để load model Production mới nhất.
    Chỉ dùng stdlib nên không cần virtualenv riêng.
    Không fail DAG nếu serving chưa lên — chỉ warn.
    """
    import urllib.request
    import urllib.error

    req = urllib.request.Request(serving_url, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            print(f"Serving reload response: {body}")
    except urllib.error.URLError as e:
        print(
            f"WARNING: Could not reach serving container at {serving_url}: {e}. "
            f"Model will be loaded on next container startup."
        )


# 0 2 1 * * : chạy lúc 02:00 ngày 1 hàng tháng (2 tiếng sau các DAG train)
@dag(
    dag_id="promote_best_model",
    start_date=datetime(2026, 4, 27),
    schedule="0 2 1 * *",
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
