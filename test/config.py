import os
from dotenv import load_dotenv
import fsspec
from src.pipeline.transform import TransformData

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