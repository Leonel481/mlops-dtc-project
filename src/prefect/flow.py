from prefect import flow, task
from pipeline.transform import TransformData
from pipeline.model_train import ModelTrain
from pipeline.model_evaluate import ModelEvaluate

import pandas as pd
import joblib


BUCKET_PATH = "gs://tu_bucket/dataset.csv"
EXPORT_PARQUET_PATH = "gs://tu_bucket/data_transformada/data_2024-01-01-2024-07-01-7.parquet"
MODEL_PATH = "gs://tu_bucket/modelos/best_model.pkl"
METRICS_PATH = "gs://tu_bucket/metrics/metrics.json"

# --- Tasks como envoltorio para cada paso ---
@task
def task_transform_data(csv_path: str) -> pd.DataFrame:
    transformer = TransformData(window_obs=3, window_churn=3)
    df = transformer.load_data(csv_path)
    df = transformer.generate_features(df)
    transformer.export_data(df, EXPORT_PARQUET_PATH)
    return df

@task
def task_train_model(df: pd.DataFrame):
    trainer = ModelTrain(months_window_obs=3, months_window_churn=3)
    X_train, y_train, X_test, y_test, scaler = trainer.prepare_data(df)
    model = trainer.train_best_model(X_train, y_train)
    joblib.dump(model, "best_model.pkl")  # local
    joblib.dump(scaler, "scaler.pkl")
    # subir a GCS si deseas aquí
    return model, X_test, y_test

@task
def task_evaluate_model(model, X_test, y_test):
    evaluator = ModelEvaluate(model)
    metrics = evaluator.evaluate(X_test, y_test)
    return metrics

# --- Flow principal ---
@flow(name="ML Pipeline Churn")
def ml_pipeline():
    df = task_transform_data(BUCKET_PATH)
    model, X_test, y_test = task_train_model(df)
    metrics = task_evaluate_model(model, X_test, y_test)
    return metrics

# --- Ejecutar localmente ---
if __name__ == "__main__":
    ml_pipeline()