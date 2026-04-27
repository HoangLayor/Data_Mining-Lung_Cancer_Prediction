# 🚗 Car Price Prediction — MLOps Pipeline

Hệ thống dự đoán giá xe sử dụng **Apache Airflow**, **MLflow**, và **FastAPI**, được đóng gói bằng Docker Compose. Hỗ trợ nhiều thuật toán (CatBoost, XGBoost, Random Forest), tự động chọn model tốt nhất dựa trên metric R² và phục vụ dự đoán qua REST API.

---

## 📐 Kiến trúc tổng quan

```
                      ┌─────────────────────────────────────────────┐
                      │          Apache Airflow (Orchestration)      │
                      │                                             │
                      │  [catboost_training]  ──┐                   │
                      │  [xgboost_training]   ──┼──► [promote_best] │
                      │  [random_forest_training]┘        │         │
                      └──────────────────────────────┬────┘─────────┘
                                                     │ reload-model
               ┌─────────────────┐                   │
               │  PostgreSQL (Data)│                   ▼
               │  car_source.CarInfo│        ┌─────────────────┐
               └────────┬────────┘          │  Model Serving  │
                        │ data              │  FastAPI :8000  │
                        ▼                   │  /predict       │
               ┌─────────────────┐          └────────┬────────┘
               │  MLflow Server  │◄──────────────────┘
               │  :5001          │  models:/car_price_model/Production
               │  - Experiments  │
               │  - Registry     │
               └─────────────────┘
```

---

## 📁 Cấu trúc thư mục

```
.
├── .env                          # Biến môi trường (DB, MLflow, Airflow)
├── docker-compose.yaml           # Toàn bộ stack: Airflow, PostgreSQL, MLflow, Serving
├── Dockerfile                    # Custom Airflow image
├── serving/
│   ├── app.py                    # FastAPI serving app
│   ├── Dockerfile
│   └── requirements.txt
└── run_env/
    ├── mlflow/
    │   └── Dockerfile            # MLflow tracking server
    └── dags/
        ├── promote_best_model.py # DAG: so sánh metric, promote model lên Production
        ├── catboost/
        │   ├── train.py          # Script train CatBoost
        │   ├── docker.py         # Airflow DAG
        │   ├── Dockerfile        # Training container
        │   ├── requirements.txt
        │   └── build.sh          # Build + push image lên DockerHub
        ├── xgboost/
        │   ├── train.py          # Script train XGBoost
        │   ├── docker.py         # Airflow DAG
        │   ├── Dockerfile
        │   ├── requirements.txt
        │   └── build.sh
        └── random_forest/
            ├── train.py          # Script train Random Forest
            ├── docker.py         # Airflow DAG
            ├── Dockerfile
            ├── requirements.txt
            └── build.sh
```

---

## 🛠️ Yêu cầu hệ thống

| Công cụ | Phiên bản tối thiểu |
|---------|-------------------|
| Docker  | 24+               |
| Docker Compose | 2.0+       |
| RAM     | 6 GB trở lên      |
| Disk    | 10 GB trống       |

---

## 🚀 Hướng dẫn chạy End-to-End

### Bước 1 — Cấu hình môi trường

Chỉnh sửa file `.env` theo môi trường của bạn:

```dotenv
AIRFLOW_PROJ_DIR=./run_env
AIRFLOW_UID=1000                         # UID của user Linux hiện tại (chạy: id -u)
HOST_TRAINING_DIR=/đường/dẫn/tuyệt/đối/run_env  # Thay bằng đường dẫn thực tế

POSTGRES_USER=admin
POSTGRES_PASSWORD=admin
POSTGRES_DB=postgres

MLFLOW_VERSION=2.3.2
```

> **Lưu ý:** `HOST_TRAINING_DIR` phải là **đường dẫn tuyệt đối** trên host (không dùng `./`), vì Docker cần mount volume từ host.

---

### Bước 2 — Build và push các Training Image lên DockerHub

Mỗi thuật toán có Docker image riêng. Build và push trước khi chạy Airflow:

```bash
# CatBoost
cd run_env/dags/catboost
bash build.sh

# XGBoost
cd ../xgboost
bash build.sh

# Random Forest
cd ../random_forest
bash build.sh
```

> **Lưu ý:** Đảm bảo đã `docker login` trước khi push. Thay `bitis2004` trong `build.sh` bằng DockerHub username của bạn nếu cần.

---

### Bước 3 — Khởi động toàn bộ stack

```bash
cd /đường/dẫn/đến/test_dm

docker compose up -d --build
```

Chờ khoảng 1–2 phút để tất cả services healthy. Kiểm tra trạng thái:

```bash
docker compose ps
```

Tất cả services phải ở trạng thái `healthy` hoặc `running`:

| Service | Địa chỉ | Mô tả |
|---------|---------|-------|
| Airflow Webserver | http://localhost:8080 | Orchestration UI (user: `airflow` / pass: `airflow`) |
| MLflow Tracking | http://localhost:5001 | Experiment tracking & Model Registry |
| Model Serving API | http://localhost:8000 | REST API dự đoán giá xe |
| PostgreSQL (DWH) | localhost:5433 | Database chứa dữ liệu xe |

---

### Bước 4 — Chạy pipeline training thủ công (lần đầu)

Vào Airflow UI tại `http://localhost:8080`, bật và trigger thủ công các DAG theo thứ tự:

**4.1** Bật và trigger 3 DAG training (có thể chạy song song):
- `catboost_training`
- `xgboost_training`
- `random_forest_training`

Hoặc trigger qua CLI:

```bash
docker exec airflow-webserver airflow dags trigger catboost_training
docker exec airflow-webserver airflow dags trigger xgboost_training
docker exec airflow-webserver airflow dags trigger random_forest_training
```

**4.2** Chờ cả 3 DAG training hoàn thành (xem logs trên Airflow UI).

**4.3** Trigger DAG `promote_best_model`:

```bash
docker exec airflow-webserver airflow dags trigger promote_best_model
```

DAG này sẽ:
1. Query MLflow tìm run có **R² cao nhất** trong experiment `Car_Price_Prediction`
2. Promote model version đó lên stage **Production** trong MLflow Registry
3. Tự động gọi `/reload-model` trên serving container để load model mới

---

### Bước 5 — Kiểm tra kết quả trên MLflow UI

Truy cập `http://localhost:5001`:

- **Experiments** → `Car_Price_Prediction`: xem tất cả runs của 3 thuật toán, so sánh metrics
- **Models** → `car_price_model`: xem các version đã register, version nào đang ở **Production**

---

### Bước 6 — Thử nghiệm Serving API

Kiểm tra API đang hoạt động:

```bash
curl http://localhost:8000/health
```

Dự đoán giá xe:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Toyota",
    "model": "Camry",
    "year": 2020,
    "mileage": 30000,
    "fuel_type": "Gasoline",
    "transmission": "Automatic",
    "color": "White",
    "num_owners": 1
  }'
```

Xem tài liệu API đầy đủ (Swagger UI):

```
http://localhost:8000/docs
```

---

## 🔄 Pipeline tự động hàng tháng

Các DAG được lên lịch tự động, **không cần can thiệp thủ công**:

| DAG | Lịch chạy | Mô tả |
|-----|-----------|-------|
| `catboost_training` | 00:00 ngày 1 hàng tháng | Train CatBoost với dữ liệu mới nhất |
| `xgboost_training` | 00:00 ngày 1 hàng tháng | Train XGBoost với dữ liệu mới nhất |
| `random_forest_training` | 00:00 ngày 1 hàng tháng | Train Random Forest với dữ liệu mới nhất |
| `promote_best_model` | 02:00 ngày 1 hàng tháng | So sánh R², promote model tốt nhất lên Production |

---

## 🔧 Xử lý sự cố thường gặp

### Serving API trả về 503 "Model not loaded"

Model chưa có ở stage Production. Cần chạy pipeline training + promote trước:

```bash
docker exec airflow-webserver airflow dags trigger catboost_training
# ... chờ hoàn thành ...
docker exec airflow-webserver airflow dags trigger promote_best_model
```

### DAG training thất bại — "Cannot pull image"

Image training chưa được push lên DockerHub. Chạy lại `build.sh` trong thư mục tương ứng và đảm bảo đã `docker login`.

### MLflow không kết nối được PostgreSQL

Kiểm tra `pgsql` container đã healthy chưa:

```bash
docker compose ps pgsql
```

Nếu chưa, chờ thêm hoặc restart:

```bash
docker compose restart pgsql
docker compose restart mlflow
```

### Reload model thủ công

Nếu cần force reload model mới nhất vào serving container mà không cần chạy lại DAG:

```bash
curl -X POST http://localhost:8000/reload-model
```

---

## 📊 Metrics đánh giá model

| Metric | Ý nghĩa | Model tốt khi |
|--------|---------|---------------|
| **R²** | Hệ số xác định (0–1) | R² càng cao càng tốt (> 0.85) |
| **RMSE** | Root Mean Squared Error | Càng thấp càng tốt |
| **MAE** | Mean Absolute Error | Càng thấp càng tốt |
| **MAPE** | Mean Absolute Percentage Error | Càng thấp càng tốt (< 15%) |

> DAG `promote_best_model` mặc định chọn model theo **R²** cao nhất. Có thể thay đổi trong hàm `find_and_promote_best_model()` nếu muốn dùng RMSE hay MAPE.

---

## ➕ Thêm thuật toán mới

Để thêm một thuật toán mới (ví dụ: LightGBM):

1. Tạo thư mục `run_env/dags/lightgbm/` với 5 file: `train.py`, `docker.py`, `Dockerfile`, `requirements.txt`, `build.sh`
2. Trong `train.py`: đặt `mlflow.set_experiment("Car_Price_Prediction")` và `registered_model_name="car_price_model"`
3. Build và push image: `bash build.sh`
4. Thêm library vào `serving/requirements.txt` nếu cần
5. Rebuild serving container: `docker compose up -d --build serving`
6. Trigger DAG training mới trên Airflow UI

DAG `promote_best_model` sẽ **tự động** so sánh và chọn model tốt nhất mà không cần sửa thêm gì.