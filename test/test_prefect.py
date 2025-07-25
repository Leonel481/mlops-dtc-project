import pytest

from src.prefect.flow import ml_pipeline
from test.config import storage_options, minio_path_data_clean, file_exists_minio, load_sample_data



def test_smoke_pipeline():

    df_sample, _ = load_sample_data(n_per_month=10)
    path_end = minio_path_data_clean()

    result_path = ml_pipeline(path_ini = None, path_end = path_end, df_override = df_sample, storage_options = storage_options())

    assert result_path.endswith('.parquet')
    assert file_exists_minio(result_path), f'File {result_path} was not uploaded to MinIO'