import os
from dotenv import load_dotenv
from unittest.mock import MagicMock
import fsspec
from src.pipeline.transform import TransformData
from src.pipeline.model_train import ModelTrain
import pytest
import pandas as pd
import numpy as np

load_dotenv()

def storage_options():

    return {
        'key': os.getenv('S3_ACCESS_KEY'),
        'secret': os.getenv('S3_SECRET_KEY'),
        'client_kwargs': {'endpoint_url': os.getenv('S3_ENDPOINT_URL')}
        }

def minio_path_extrac():
    
    bucket = 'mlops-bucket'
    file = 'data_raw'
    name = 'online_retail_cleaned_2009-2011.csv'

    return f's3://{bucket}/{file}/{name}'

def minio_path_data_clean():

    bucket = 'mlops-bucket'
    file = 'data_processed'

    return f's3://{bucket}/{file}'

def file_exists_minio(path):

    fs = fsspec.filesystem('s3', **storage_options())

    return fs.exists(path)

def load_sample_data(n_per_month=100):

    tranformer = TransformData()

    df = tranformer.load_data(minio_path_extrac(), storage_options = storage_options())
    df['Month'] = df['InvoiceDate'].dt.to_period('M')
    
    df_sample = (
        df.groupby('Month')
            .apply(lambda x: x.sample(n=min(len(x), n_per_month), random_state=42))
            .reset_index(drop=True)
            .drop(columns='Month'))
    
    return df_sample, tranformer

@pytest.fixture
def model_train_instance():
    return ModelTrain(project="test-project", location="us-central1", bucket="gs://test-bucket")

@pytest.fixture
def mock_df():
    return pd.DataFrame({
        'CustomerID': range(10),
        'window_id': [1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
        'churn': [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
    })

@pytest.fixture
def mock_scalar_data():
    mock_scaler = MagicMock()
    mock_scaler.fit_transform.return_value = np.random.rand(4, 2)
    mock_scaler.transform.side_effect = [np.random.rand(2, 2), np.random.rand(4, 2)]
    return mock_scaler, (
        np.random.rand(4, 2), pd.Series([0, 1, 0, 1]),
        np.random.rand(2, 2), pd.Series([1, 0]),
        np.random.rand(4, 2), pd.Series([0, 0, 1, 0]),
    )

class DummyModelForTest:
    def __init__(self, **kwargs):
        self.params = kwargs

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        return np.random.rand(X.shape[0], 2)