import sys
import os
# sys.path.append("/app")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from hyperopt import hp
from hyperopt.pyll.base import scope

from prefect import flow, task
from src.pipeline.transform import TransformData
from src.pipeline.model_train import ModelTrain
# from src.pipeline.model_deploy import ModelEvaluate
from dotenv import load_dotenv

import pandas as pd
import joblib


load_dotenv()

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

    df, encoder = transformer.features(df_cleaned)

    final_path, encoder_path = transformer.load_data_clean(df_cleaned, path_end, encoder, **kwargs)

    return final_path, encoder_path

@task
def task_train_model(project, bucket, path_data_process: str, path_artifacts: str, path_models: str, path_metrics:str, models, param_space):

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "LightGBM": LGBMClassifier(random_state=42)
    }
    
    param_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 15, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
    }

    trainer = ModelTrain(project = project, bucket = bucket,
                         path_artifacts = path_artifacts,
                         path_models = path_models, 
                         path_metrics = path_metrics)
                         
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = trainer.load_and_prepare_initial_splits(path_data_process)

    start_date ='18182025'
    end_date = '20202025'

    trainer.save_artifacts(start_date = start_date, end_date = end_date, X_train_scaled = X_train_scaled, 
                           X_valid_scaled = X_valid_scaled, X_test_scaled = X_test_scaled, 
                           y_train = y_train, y_valid = y_valid, y_test = y_test)
    
    metrics = trainer.train_base_models(models = models, data_source = path_data_process, X_train = X_train_scaled, 
                                        y_train = y_train, X_val = X_valid_scaled, y_val= y_valid)
    
    trainer.registry_best_model(results = metrics)

    trainer.generate_forward_chaining_splits()

    path_final_model = trainer.tune_model(param_space = param_space, max_evals = 100 )

    trainer.evaluate_model(X_test_scaled = X_test_scaled, y_test = y_test)

    trainer.register_final_model(path_final_model,y_test)


# @task
# def task_evaluate_model(model, X_test, y_test):
#     evaluator = ModelEvaluate(model)
#     metrics = evaluator.evaluate(X_test, y_test)
#     return metrics


# --- Flow principal ---
@flow(name="ML Pipeline")
def ml_pipeline(project, bucket, path_ini, path_end, path_artifacts, path_models, path_metrics, df_override=None, **kwargs):
    clean_data_path, encoder_path  = task_ETL_data(path_ini, path_end, df_override = df_override, **kwargs)
    task_train_model(project, bucket, path_data_process = clean_data_path, 
                     path_artifacts = path_artifacts, path_models = path_models, 
                     path_metrics = path_metrics)
    # metrics = task_evaluate_model(model, X_test, y_test)
    return clean_data_path

if __name__ == "__main__":
    ml_pipeline.serve(
        parameters = {
            'path_ini': os.getenv('DATA_RAW_PATH'),
            'path_end': os.getenv('EXPORT_PARQUET_PATH'),
            'project': os.getenv('PROJECT'),
            'bucket': os.getenv('BUCKET'),
            'path_artifacts': os.getenv('PATH_ARTIFACTS'),
            'path_models': os.getenv('PATH_MODELS'),
            'path_metrics': os.getenv('PATH_METRICS')
        } 
    )


    
    

