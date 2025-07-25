import pytest
import pandas as pd
import logging
from src.pipeline.transform import TransformData
from test.config import storage_options , minio_path_data_clean, file_exists_minio, load_sample_data

logging.basicConfig(level = logging.INFO)


def test_load_minio():

    df, _ = load_sample_data()
    logging.info(f'Data Frame clumns: {list(df.columns)}')
    logging.info(f'DataFrame shape: {df.shape}')

    expected_columns = {
                        'InvoiceNo','StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice',
                        'CustomerID', 'Country', 'IsCancelled', 'TotalPrice'
                        }
    
    assert isinstance(df, pd.DataFrame),  'Object is not DataFrame'
    assert not df.empty, 'Dataframe empty'
    assert expected_columns.issubset(set(df.columns)) , f'Columns missing : {expected_columns - set(df.columns)}'
    assert df['CustomerID'].dtype == 'object' , 'CustomerID should be string'
    assert df['InvoiceDate'].dtype == 'datetime64[ns]' , 'InvoiceDate should be datetime'


def test_group_daily_dates():

    df, _ = load_sample_data()
    df_grouped = TransformData().group_daily_dates(df)

    logging.info(f'Data Frame clumns: {list(df_grouped.columns)}')
    logging.info(f'DataFrame shape: {df_grouped.shape}')

    expected_columns = {
                    'CustomerID','InvoiceDate', 'Country', 'Invoices', 'Invoices_canceled', 'Unique_products_buy',
                    'Unique_products_return', 'Items_buy', 'Items_return', 'Items_net', 'Value_buy', 'Items_return',
                    'Items_net', 'Value_buy', 'Value_return', 'Total_value'
                    }

    assert isinstance(df_grouped, pd.DataFrame),  'Object is not DataFrame'
    assert not df_grouped.empty, 'Dataframe empty'
    assert expected_columns.issubset(set(df_grouped.columns)) , f'Columns missing : {expected_columns - set(df_grouped.columns)}'
    assert (df_grouped['Items_net'] != 0).all(), 'Items_net contains 0 after filtering'
    assert (df_grouped['Total_value'] != 0).all(), 'Total_value contains 0 after filtering'

def test_transform_data():

    df, _ = load_sample_data()
    df_grouped = TransformData().group_daily_dates(df)
    df_transform = TransformData().transform_data(df_grouped, churn_treshold = 0.2)

    logging.info(f'Data Frame clumns: {list(df_transform.columns)}')
    logging.info(f'DataFrame shape: {df_transform.shape}')

    expected_columns = {
                    'CustomerID', 'window_id', 'total_products', 'total_products_buys',
                    'value_buys', 'products_unique_buys', 'avg_invoice_buy', 'invoices_buy',
                    'total_products_return', 'value_return', 'products_unique_return',
                    'avg_invoice_return', 'invoices_return', 'total_value_obs',
                    'total_value_std', 'return_rate', 'return_value_rate', 'recency_days',
                    'customer_longevity', 'avg_days_between_purchases', 'month_frecuency',
                    'churn'
                    }

    assert isinstance(df_transform, pd.DataFrame),  'Object is not DataFrame'
    assert not df_transform.empty, 'Dataframe empty'
    assert expected_columns.issubset(set(df_transform.columns)) , f'Columns missing : {expected_columns - set(df_transform.columns)}'
    assert df_transform['window_id'].min() >= 1, 'Window IDs should start at 1'

    assert set(df_transform['churn'].unique()).issubset({0,1}), 'Churn must be binary'

def test_handle_outliers():

    df, _ = load_sample_data()
    df_grouped = TransformData().group_daily_dates(df)
    df_transform = TransformData().transform_data(df_grouped, churn_treshold = 0.2)

    cols = df_transform.drop(columns=['CustomerID', 'window_id', 'churn']).columns.tolist()
    df_cleaned = TransformData().handle_outliers(df_transform, cols)

    logging.info(f'Data Frame clumns: {list(df_cleaned.columns)}')
    logging.info(f'DataFrame shape: {df_cleaned.shape}')

    assert not df_transform.empty, 'Dataframe empty'
    assert df_cleaned.shape == df_transform.shape, 'Shape changed after winsorizing'

def test_load_data_clean():

    df, transformer = load_sample_data()
    df_grouped = transformer.group_daily_dates(df)
    df_transform = transformer.transform_data(df_grouped, churn_treshold = 0.2)

    cols = df_transform.drop(columns=['CustomerID', 'window_id', 'churn']).columns.tolist()
    df_cleaned = transformer.handle_outliers(df_transform, cols)

    path_clean = minio_path_data_clean()

    path_output  = transformer.load_data_clean(df_cleaned, path_clean, storage_options = storage_options())

    assert path_output.endswith(".parquet"), 'The file does not have a parquet extension.'
    assert file_exists_minio(path_output), f'File {path_output} was not uploaded to MinIO'