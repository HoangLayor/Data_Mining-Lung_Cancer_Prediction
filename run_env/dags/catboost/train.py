import os
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100

if __name__ == "__main__":
    print("Loading data from PostgreSQL...")
    
    # DB parameters matching docker-compose.yaml .env values
    DB_USER = os.environ.get("POSTGRES_USER", "admin")
    DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "admin")
    DB_HOST = os.environ.get("POSTGRES_HOST", "pgsql")
    DB_PORT = os.environ.get("POSTGRES_PORT", "5432")
    DB_NAME = os.environ.get("POSTGRES_DB", "postgres")
    
    db_uri = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_uri)
    
    query = "SELECT * FROM car_source.CarInfo;"
    df = pd.read_sql(query, engine)
    
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    X = df.drop(columns=["price"]).copy()
    y = df["price"].copy()

    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    for col in cat_features:
        X[col] = X[col].astype(str)

    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

    cat_features_idx = [
        X.columns.get_loc(col)
        for col in cat_features
    ]
    
    print("Starting training...")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("Car_Price_Prediction")
    
    with mlflow.start_run(run_name="catboost"):
        mlflow.set_tag("model_type", "catboost")
        # Best params used for regular automated retraining
        best_params = {
            "iterations": 1200,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5.0,
            "min_data_in_leaf": 50,
            "bagging_temperature": 1.0,
            "random_strength": 2.0,
            "border_count": 100,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "verbose": 200,
            "cat_features": cat_features_idx
        }
    
        mlflow.log_params(best_params)
    
        final_model = CatBoostRegressor(**best_params)
    
        preprocess = ColumnTransformer(
            transformers=[
                ("mileage_scaler", StandardScaler(), ["mileage"])
            ],
            remainder="passthrough"
        )
    
        x_pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", final_model)
        ])
    
        final_pipe = TransformedTargetRegressor(
            regressor=x_pipe,
            transformer=PowerTransformer(method="yeo-johnson")
        )
    
        final_pipe.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
    
        print("Starting predicting...")
        y_pred = final_pipe.predict(X_test)
    
        metrics = {
            "rmse": rmse(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "mape": mape(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }
    
        print(f"RMSE : {metrics['rmse']:,.0f}")
        print(f"MAE  : {metrics['mae']:,.0f}")
        print(f"MAPE : {metrics['mape']:.2f}%")
        print(f"R²   : {metrics['r2']:.3f}")
        
        mlflow.log_metrics(metrics)
    
        print("Saving model to file...")
        # Save the model relative to current directory for the Docker pipeline
        model_path = "model.pkl"
    
        with open(model_path, "wb") as f:
            pickle.dump(final_pipe, f)
    
        print(f"Model saved to: {model_path}")
        
        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(final_pipe, "catboost_pipeline",
                                 registered_model_name="car_price_model")
        print("MLflow tracking completed.")