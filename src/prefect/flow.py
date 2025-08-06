import sys
# sys.path.append("/app")

from prefect import flow, task
from src.pipeline.transform import TransformData
from src.pipeline.model_train import ModelTrain
# from src.pipeline.model_deploy import ModelEvaluate

import pandas as pd
import joblib


BUCKET_PATH = "s3://mlops-bucket/data_raw/online_retail_cleaned_2009-2011.csv"
EXPORT_PARQUET_PATH = "s3://mlops-bucket/data_processed"
PROJECT = ''
BUCKET = ''
EXPERIMENT_NAME = ''
PATH_DESTINY = ''


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

@task
def task_train_model(gcs_path: str, models, param_space):

    trainer = ModelTrain(project = PROJECT, bucket = BUCKET, experiment_name = EXPERIMENT_NAME)
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = trainer.load_and_prepare_initial_splits(gcs_path)

    start_date =''
    end_date = ''
    data_source = ''

    trainer.save_artifacts(path_destiny = PATH_DESTINY, start_date = start_date, 
                           end_date = end_date, X_train_scaled = X_train_scaled, 
                           X_valid_scaled = X_valid_scaled, X_test_scaled = X_test_scaled, 
                           y_train = y_train, y_valid = y_valid, y_test = y_test)
    metrics = trainer.train_base_models(models = models, gcs_model_path = gcs_path, 
                                        data_source = data_source, X_train = X_train_scaled, 
                                        y_train = y_train, X_val = X_valid_scaled, y_val= y_valid)
    trainer.registry_best_model(metrics = metrics, max_evals = 100)
    trainer.generate_forward_chaining_splits()
    gcs_eval_path = trainer.tune_model(param_space = param_space, max_evals = 100 )
    trainer.evaluate_model(gcs_eval_path = gcs_eval_path, X_test_scaled = X_test_scaled, y_test = y_test)
    trainer.register_final_model()


# @task
# def task_evaluate_model(model, X_test, y_test):
#     evaluator = ModelEvaluate(model)
#     metrics = evaluator.evaluate(X_test, y_test)
#     return metrics


# --- Flow principal ---
@flow(name="ML Pipeline")
def ml_pipeline(path_ini, path_end, models, param_space, df_override=None, **kwargs):
    clean_data_path = task_ETL_data(path_ini, path_end, df_override = df_override, **kwargs)
    task_train_model(gcs_path = clean_data_path, models = models, param_space = param_space)
    # metrics = task_evaluate_model(model, X_test, y_test)
    return clean_data_path

if __name__ == "__main__":
    ml_pipeline.serve(
        parameters = {
            'path_ini': BUCKET_PATH,
            'path_end': EXPORT_PARQUET_PATH,
            'models': {
                    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                    "RandomForest": RandomForestClassifier(random_state=42),
                    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                    "LightGBM": LGBMClassifier(random_state=42)
                },
            'param_space' : {
                    'max_depth': scope.int(hp.quniform('max_depth', 4, 15, 1)),
                    'learning_rate': hp.loguniform('learning_rate', -3, 0),         # ~0.05–1.0
                    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
                    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
                    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
                    'subsample': hp.uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
                }
        } 
    )


    
    

