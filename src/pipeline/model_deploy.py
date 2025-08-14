import pandas as pd
import pickle
import fsspec
from typing import Any, Dict, List
from google.cloud import bigquery
import json

from evidently import Report
# from evidently.metrics import *
from evidently import DataDefinition, BinaryClassification
from evidently.presets import DataDriftPreset, ClassificationPreset


def load_pickle(obj: Any, path: str, **kwargs):
    """
    Load an object as a pickle file to local or remote storage (e.g., GCS).

    Args:
        obj (Any): Object to be pickled.
        path (str): Full path to save the pickle file (supports GCS, S3, local).
    """
    fs, _, paths = fsspec.get_fs_token_paths(path , **kwargs)
    with fs.open(paths[0], "wb") as f_in:
        return pickle.load(f_in)


class ModelDeploy:
    """
    Class for performing batch predictions and monitoring with Evidently AI.
    """
    def __init__(self, project: str, bigquery_table_id: str):
        self.project = project
        self.bigquery_table_id = bigquery_table_id
        self.model = None
        self.scaler = None
        self.oneHot = None
        self.bq_client = bigquery.Client(project=self.project)

    def load_artifacts(self, model_path: str, scaler_path: str, encoder_path : str, **kwargs) -> None:
        """
        Loads the pickled model and scaler from their respective paths.
        
        Args:
            model_path (str): GCS path to the pickled model.
            scaler_path (str): GCS path to the pickled scaler.
            encoder_path (str): GCS path to the pickled encoder.
        """

        self.model = load_pickle(model_path, **kwargs)
        self.scaler = load_pickle(scaler_path, **kwargs)
        self.ohe = load_pickle(encoder_path, **kwargs)
    
    def get_new_data(self, data_path: str, **kwargs) -> tuple[pd.DataFrame, pd.Series]:
        """
        Loads new data for prediction and separates features from the target column.
        
        Args:
            data_path (str): GCS path to the new data file (e.g., Parquet).
        
        Returns:
            tuple[pd.DataFrame, pd.Series]: Features and target columns.
        """

        new_data_df = pd.read_parquet(data_path, **kwargs)
        
        features_df = new_data_df.drop(columns=['CustomerID', 'window_id', 'churn'])
        target_series = new_data_df['churn']
        
        return features_df, target_series

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Scales the new data and generates batch predictions.
        
        Args:
            features_df (pd.DataFrame): DataFrame with new features.
        
        Returns:
            pd.DataFrame: DataFrame containing the predictions.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model and scaler must be loaded before making predictions.")

        # Scale the new data using the pre-fitted scaler
        scaled_features = self.scaler.transform(features_df)
        
        # Generate predictions
        predictions = self.model.predict_proba(scaled_features)[:, 1]
        predictions_df = pd.DataFrame({'prediction_proba': predictions}, index=features_df.index)
        
        return predictions_df
    
    def _flatten_report_to_bq_rows(self, report_dict: Dict, timestamp: str) -> List[Dict]:
        """
        Schema of the bigquery table
        """

        rows = []

        for metric in report_dict.get('metrics', []):
            metric_type = metric.get('metric_type')
            result = metric.get('result', {})

            if metric_type == 'ClassificationPreset':
                metrics = {
                    "accuracy_score": result.get('accuracy_score', {}).get('value'),
                    "f1_score": result.get('f1_score', {}).get('value'),
                    "precision": result.get('precision', {}).get('value'),
                    "recall": result.get('recall', {}).get('value'),
                    "roc_auc": result.get('roc_auc', {}).get('value'),
                    "log_loss": result.get('log_loss', {}).get('value')
                }
                for name, value in metrics.items():
                    if value is not None:
                        rows.append({
                            "timestamp": timestamp,
                            "metric_type": name,
                            "feature_name": None,
                            "metric_value": float(value)
                        })

            elif metric_type == 'DataQualityPreset':
                for drift_data in result.get('dataset_drift', {}).get('drift_by_columns', {}).values():
                    rows.append({
                        "timestamp": timestamp,
                        "metric_type": "drift_pvalue",
                        "feature_name": drift_data.get('column_name'),
                        "metric_value": float(drift_data.get('p_value'))
                    })
                for column_name, data in result.get('dataset_missing_values', {}).get('different_missing_values', {}).items():
                    rows.append({
                        "timestamp": timestamp,
                        "metric_type": "missing_value_percentage",
                        "feature_name": column_name,
                        "metric_value": float(data.get('different_percentage'))
                    })
        
        return rows

    def generate_and_save_report(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
        """
        Generates an Evidently AI report and saves it to a BigQuery table.
        
        Args:
            reference_data (pd.DataFrame): Reference data (e.g., historical training data).
            current_data (pd.DataFrame): Current data with new predictions.
        """
        # Create an Evidently Report with data quality and classification metrics

        data_definition = DataDefinition(
            numerical_columns=[],
            categorical_columns=[],
            classification= [BinaryClassification(target= 'Churn', prediction_labels='Predicted_label', prediction_probas= 'Predicted_proba', pos_label=1)]
        )

        data_and_model_report = Report(metrics=[
            DataDriftPreset(),
            ClassificationPreset()
        ])
        
        data_and_model_report.run(
            reference_data=reference_data, 
            current_data=current_data,
            data_definition=data_definition
        )
        
        timestamp_now = pd.Timestamp.now().isoformat()
        report_dict = data_and_model_report.as_dict()
        rows_to_insert = self._flatten_report_to_bq_rows(report_dict, timestamp_now)
        
        if rows_to_insert:
            print(f"Insert {len(rows_to_insert)} on Bigquery: {self.bigquery_table_id}")
            errors = self.bq_client.insert_rows_json(self.bigquery_table_id, rows_to_insert)
            
            if errors:
                print(f'error: {errors}')
            else:
                print('Insert successful')
        else:
            print("No metrics found to insert.")