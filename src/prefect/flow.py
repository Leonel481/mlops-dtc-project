import sys
# sys.path.append("/app")

from prefect import flow, task
from src.pipeline.transform import TransformData
from src.pipeline.model_train import ModelTrain
from src.pipeline.model_evaluate import ModelEvaluate

import pandas as pd
import joblib


BUCKET_PATH = "s3://mlops-bucket/data_raw/online_retail_cleaned_2009-2011.csv"
EXPORT_PARQUET_PATH = "s3://mlops-bucket/data_processed"
# MODEL_PATH = "gs://tu_bucket/modelos/best_model.pkl"
# METRICS_PATH = "gs://tu_bucket/metrics/metrics.json"

# --- Tasks como envoltorio para cada paso ---
@task
def task_ETL_data(path_ini: str, path_end: str, df_override: pd.DataFrame = None , **kwargs) -> str:

    transformer = TransformData(months_window_obs=3, months_window_churn=3)

    # If a DataFrame is passed, we use it instead of loading from storage. For tested Dataframe o unit test
    if df_override is not None:
        df = df_override
        transformer.data_start = df['InvoiceDate'].min().strftime('%Y%m')
        transformer.data_end = df['InvoiceDate'].max().strftime('%Y%m')
    else:
        df = transformer.load_data(path_ini, **kwargs)

    df_grouped = transformer.group_daily_dates(df)
    df_transform = transformer.transform_data(df_grouped, churn_treshold = 0.2)

    cols = df_transform.drop(columns=['CustomerID', 'window_id', 'churn']).columns.tolist()
    df_cleaned = transformer.handle_outliers(df_transform, cols)

    final_path = transformer.load_data_clean(df_cleaned, path_end, **kwargs)

    return final_path

# @task
# def task_train_model(df: pd.DataFrame):
#     trainer = ModelTrain(months_window_obs=3, months_window_churn=3)
#     X_train, y_train, X_test, y_test, scaler = trainer.prepare_data(df)
#     model = trainer.train_best_model(X_train, y_train)
#     joblib.dump(model, "best_model.pkl")  # local
#     joblib.dump(scaler, "scaler.pkl")
#     # subir a GCS si deseas aquí
#     return model, X_test, y_test

# @task
# def task_evaluate_model(model, X_test, y_test):
#     evaluator = ModelEvaluate(model)
#     metrics = evaluator.evaluate(X_test, y_test)
#     return metrics

# --- Flow principal ---
@flow(name="ML Pipeline")
def ml_pipeline(path_ini, path_end, df_override=None, **kwargs):
    clean_data_path = task_ETL_data(path_ini, path_end, df_override = df_override, **kwargs)
    # model, X_test, y_test = task_train_model(df)
    # metrics = task_evaluate_model(model, X_test, y_test)
    return clean_data_path

if __name__ == "__main__":
    ml_pipeline.serve(
        parameters = {
            'path_ini': BUCKET_PATH,
            'path_end': EXPORT_PARQUET_PATH
        } 
    )

    models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            "LightGBM": LGBMClassifier(random_state=42)
        }