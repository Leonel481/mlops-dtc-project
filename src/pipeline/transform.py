import pandas as pd
import numpy as np
import pickle
import fsspec
from typing import Tuple, Any
from sklearn.preprocessing import OneHotEncoder

def dump_pickle(obj: Any, path: str, **kwargs):
    """
    Save an object as a pickle file to local or remote storage (e.g., GCS).

    Args:
        obj (Any): Object to be pickled.
        path (str): Full path to save the pickle file (supports GCS, S3, local).
    """
    fs, _, paths = fsspec.get_fs_token_paths(path , **kwargs)
    with fs.open(paths[0], "wb") as f_out:
        pickle.dump(obj, f_out)
    

class TransformData():
    """
    Class to transform data for churn prediction.
    """
    def __init__(self, months_window_obs: int = 3, months_window_churn: int = 3):
        """
        Initialize the TransformData class.

        Args:
            window_obs (int): Number of months for the observation window.
            window_churn (int): Number of months for the churn window.
        """
        if months_window_obs <= 0 or months_window_churn <= 0:
            raise ValueError("Observation and churn windows must be positive integers.")
        self.months_window_obs = months_window_obs  # months for observation window
        self.months_window_churn = months_window_churn  # months for churn window

    def load_data(self, gcs_path: str , **kwargs) -> pd.DataFrame:
        """
        Load data raw from Google Clous Storage and return DataFrame.

        Args:
            gcs_path (str): Path to the CSV file in Google Cloud Storage.

        Returns:
            DataFrame: Transformed DataFrame with features for churn prediction.
        """
        df = pd.read_csv(gcs_path, **kwargs)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()
        df = df.dropna(subset=['CustomerID'])
        df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
        df = df[~df['Description'].isin(['Manual','Discount','This is a test product.'])]

        self.data_start = df['InvoiceDate'].min().strftime('%Y%m')
        self.data_end = df['InvoiceDate'].max().strftime('%Y%m')

        return df
    
    def group_daily_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group the data by daily dates.

        Args:
            df (DataFrame): DataFrame from load_data function.
        
        Returns:
            DataFrame: DataFrame with daily grouped data.
        """

        df_diario = df.groupby(['CustomerID','InvoiceDate','Country']).agg(
                    Invoices=('InvoiceNo', lambda x: x[~x.str.startswith('C')].nunique()),
                    Invoices_canceled=('InvoiceNo', lambda x: x[x.str.startswith('C')].nunique()),
                    Unique_products_buy=('StockCode', lambda x: x[df.loc[x.index, 'Quantity'] > 0].nunique()),
                    Unique_products_return=('StockCode', lambda x: x[df.loc[x.index, 'Quantity'] < 0].nunique()),
                    Items_buy=('Quantity', lambda x: x[x > 0].sum()),
                    Items_return=('Quantity', lambda x: x[x < 0].sum()),
                    Items_net=('Quantity','sum'),
                    Value_buy=('TotalPrice', lambda x: x[x > 0].sum()),
                    Value_return=('TotalPrice', lambda x: x[x < 0].sum()),
                    Total_value=('TotalPrice', 'sum'),
                    ).reset_index()
        
        df_grouped = df_diario[
                        (df_diario['Items_net'] != 0) &
                        (df_diario['Total_value'] != 0)
                        ]
        
        return df_grouped

    def transform_data(self, df: pd.DataFrame, churn_treshold: int = 0.2) -> pd.DataFrame:
        """
        Transform the data to create features for churn prediction.

        Args:
            df (DataFrame): DataFrame from group_daily_dates function.
        
        Returns:
            DataFrame: Transformed DataFrame with features for churn prediction.
        """

        # Ventasas churn
        meses_obs = self.months_window_obs
        meses_churn = self.months_window_churn
        obs_ini = df['InvoiceDate'].min() # 2010-12-01
        churn_treshold = churn_treshold
        window = []
        window_id = 1

        while True:

            obs_end = obs_ini + pd.DateOffset(months=meses_obs) - pd.DateOffset(days=1)
            churn_ini = obs_end + pd.DateOffset(days=1)
            churn_end = churn_ini + pd.DateOffset(months=meses_churn) - pd.DateOffset(days=1)

            if churn_end > df['InvoiceDate'].max():
                break

            df_obs = df[(df['InvoiceDate'] >= obs_ini) & (df['InvoiceDate'] <= obs_end)]
            df_churn = df[(df['InvoiceDate'] >= churn_ini) & (df['InvoiceDate'] <= churn_end) & (df['Items_net'] > 0)]

            customers_obs = df_obs['CustomerID'].dropna().unique()
            customer_churn = df_churn['CustomerID'].dropna().unique()

            for customer in customers_obs:

                df_customer = df_obs[df_obs['CustomerID'] == customer].copy()

                # Buys
                total_products_buys = df_customer['Items_buy'].sum()
                value_buys = df_customer['Value_buy'].sum()
                invoices_buy = df_customer['Invoices'].sum()
                avg_invoice_buy = value_buys / invoices_buy if invoices_buy > 0 else 0
                products_unique_buys = df_customer['Unique_products_buy'].nunique()

                # Returns
                total_products_return = df_customer['Items_return'].nunique()
                value_return = df_customer['Value_return'].sum()
                invoices_return = df_customer['Invoices_canceled'].sum()
                avg_invoice_return = value_return / invoices_return if invoices_return > 0 else 0
                products_unique_return = df_customer['Unique_products_return'].nunique()

                # Month frequency
                month_frecuency = df_customer['InvoiceDate'].dt.month.mode()

                # Total
                total_products = df_customer['Items_net'].sum()
                total_value_obs = df_customer['Total_value'].sum()
                total_value_std = df_customer['Total_value'].std(ddof=0)

                # Time variables
                customer_longevity = (df_churn['InvoiceDate'].max() - df_customer['InvoiceDate'].min()).days
                recency_days = (df_obs['InvoiceDate'].max() - df_customer['InvoiceDate'].max()).days
                dates = df_customer['InvoiceDate'].drop_duplicates().sort_values()
                if len(dates) > 1:
                    diffs = dates.diff().dropna()
                    avg_days_between_purchases = diffs.mean().days
                else:
                    avg_days_between_purchases = (df_obs['InvoiceDate'].max() - df_customer['InvoiceDate'].max()).days

                # Ratio
                return_rate = total_products_return / total_products_buys if total_products_buys > 0 else np.nan
                return_value_rate = value_return / value_buys if value_buys > 0 else np.nan
                
                # Value in window churn
                value_churn = df_churn[df_churn['CustomerID'] == customer]['Total_value'].sum()

                # Churn
                churn = 1 if (customer not in customer_churn or value_churn <= churn_treshold * total_value_obs) else 0

                # Filter so that the window does not have only returns
                if total_products_buys == 0 and total_products_return > 0:
                    continue

                window.append({
                    'CustomerID': customer,
                    'window_id': window_id,
                    'total_products': total_products,
                    'total_products_buys': total_products_buys,
                    'value_buys': value_buys,
                    'products_unique_buys': products_unique_buys,
                    'avg_invoice_buy': avg_invoice_buy,
                    'invoices_buy': invoices_buy,
                    'total_products_return': total_products_return,
                    'value_return': value_return,
                    'products_unique_return': products_unique_return,
                    'avg_invoice_return': avg_invoice_return,
                    'invoices_return': invoices_return,
                    'total_value_obs':total_value_obs,
                    'total_value_std': total_value_std,
                    'return_rate': return_rate,
                    'return_value_rate': return_value_rate,
                    'recency_days': recency_days,
                    'customer_longevity': customer_longevity,
                    'avg_days_between_purchases': avg_days_between_purchases,
                    'month_frecuency': month_frecuency.iloc[0] if not month_frecuency.empty else np.nan,
                    'churn': churn
                })

            obs_ini = obs_ini + pd.DateOffset(months=meses_obs)
            window_id += 1

        df_transformed  = pd.DataFrame(window)

        return df_transformed 

    def handle_outliers(self, df: pd.DataFrame, cols: list, group_col: str ='window_id') -> pd.DataFrame:
        """
        Handle outliers using winsorizing (clip to Tukey bounds) for each group.

        Args:
            df (DataFrame): Transformed DataFrame.
            cols (list): List of columns for treatment outliers
            group_col: Column for agrouped DataFrame
        
        Returns:
            DataFrame: Cleaned DataFrame with features for churn prediction.
        """
        
        df = df.copy()

        for col in cols:
        # Make sure the column is float to avoid errors when assigning Q1/Q3
            if not pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype(float)

            # Limits for group
            grouped = df.groupby(group_col)[col]
            Q1 = grouped.transform(lambda x: x.quantile(0.25))
            Q3 = grouped.transform(lambda x: x.quantile(0.75))
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            # Clip (winsorize)
            df[col] = df[col].clip(lower, upper)

        return df 

    def features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, OneHotEncoder] :
        """
        Prepare features and train, test datasets.

        Args:
            df (pd.DataFrame): Dataframe process with handle outliers.

        Returns:
            Dataframe: Dataframe with encoders
            encoder: Enconder of the features month_frecuency
        """

        # OneHotEncode for month_frecuency
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        month_encoded  = encoder.fit_transform(df[['month_frecuency']])
        month_encoded_df = pd.DataFrame(month_encoded, columns=encoder.get_feature_names_out(['month_frecuency']), index=df.index)
    
        df_num = df.drop(columns=['CustomerID', 'month_frecuency'])
        df_final = pd.concat([df_num, month_encoded_df], axis=1)

        return df_final, encoder


    def load_data_clean(self, df: pd.DataFrame, gcs_path: str, encoder: OneHotEncoder, engine: str = 'pyarrow', **kwargs) -> str:
        """
        Save the transformed data (parquet) and encoder (pkl) to Google Cloud Storage.

        Args:
            df (DataFrame): Cleaned DataFrame.
            gcs_path (str): Path to save the Parquet file in Google Cloud Storage.
            encoder: Encoder
            engine: pyarrow

        Returns:
            str: Path to the save Parquet file.
        """
        start_date = self.data_start
        end_date = self.data_end

        filename = f'data_{start_date}_{end_date}'
        full_path = f'{gcs_path.rstrip('/')}/{filename}.parquet'
        encoder_path = f"{gcs_path.rstrip('/')}/encoder_{filename}.pkl"

        df.to_parquet(full_path, engine = engine,index=False, **kwargs)
        dump_pickle(encoder, encoder_path)

        print(f"Data save to {gcs_path}")

        return str(gcs_path)
    
       # def features(self, df: pd.DataFrame, window_target : int = None) -> tuple :
    #     """
    #     Prepare features and train, test datasets.

    #     Args:
    #         df (pd.DataFrame): Dataframe process with handle outliers.
    #         window_target (int): Window to use as a test (last by default).

    #     Returns:
    #         tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_encoded)
    #     """

    #     # Target window
    #     window_target = window_target if window_target is not None else df_f['window_id'].max()

    #     # Split data and target
    #     train_df = df[df['window_id'] < window_target - 1]
    #     valid_df = df[df['window_id'] == window_target - 1]
    #     test_df  = df[df['window_id'] == window_target]

    #     y_train = train_df['churn']
    #     y_valid = valid_df['churn']
    #     y_test = test_df['churn']

    #     # Separate Features
    #     X_train_num = train_df.drop(columns=['CustomerID', 'window_id', 'churn', 'month_frecuency'])
    #     X_valid_num = valid_df.drop(columns=['CustomerID', 'window_id', 'churn', 'month_frecuency'])
    #     X_test_num = test_df.drop(columns=['CustomerID', 'window_id', 'churn', 'month_frecuency'])

    #     # OneHotEncode for month_frecuency
    #     encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    #     X_train_cat = encoder.fit_transform(train_df[['month_frecuency']])
    #     X_valid_cat = encoder.fit_transform(valid_df[['month_frecuency']])
    #     X_test_cat = encoder.transform(test_df[['month_frecuency']])

    #     # Combine numeric and categorical
    #     X_train_full = np.hstack([X_train_num.values, X_train_cat])
    #     X_valid_full = np.hstack([X_valid_num.values, X_valid_cat])
    #     X_test_full = np.hstack([X_test_num.values, X_test_cat])

    #     # Scale
    #     scaler = StandardScaler()
    #     X_train_scaled = scaler.fit_transform(X_train_full)
    #     X_valid_scaled = scaler.fit_transform(X_valid_full)
    #     X_test_scaled = scaler.transform(X_test_full)

    #     return X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid,  y_test, scaler, encoder


    
    # def save_artifacts(self, gcs_path: str, X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoder) -> Tuple[str, str, str, str]:
    #     """
    #     Save model artifacts (datasets, scaler, encoder) as pickles.

    #     Args:
    #         gcs_path (str): Base GCS path (e.g., "gs://bucket/artifacts").
    #         X_train_scaled (np.ndarray): Scaled training features.
    #         X_test_scaled (np.ndarray): Scaled test features.
    #         y_train (np.ndarray): Training labels.
    #         y_test (np.ndarray): Test labels.
    #         scaler (Any): Fitted scaler object (e.g., StandardScaler).
    #         encoder (Any): Fitted encoder object (e.g., OneHotEncoder).

    #     Returns:
    #         Tuple[str, str, str, str]: Paths to the saved pickle files.
    #     """
    #     start_date = self.data_start
    #     end_date = self.data_end
    #     filename = f'data_{start_date}_{end_date}.parquet'


    #     train_path = f"{gcs_path.rstrip('/')}/train_{filename}.pkl"
    #     test_path = f"{gcs_path.rstrip('/')}/test_{filename}.pkl"
    #     scaler_path = f"{gcs_path.rstrip('/')}/scaler_{filename}.pkl"
    #     encoder_path = f"{gcs_path.rstrip('/')}/encoder_{filename}.pkl"

    #     dump_pickle((X_train_scaled, y_train), train_path)
    #     dump_pickle((X_test_scaled, y_test), test_path)
    #     dump_pickle(scaler, scaler_path)
    #     dump_pickle(encoder, encoder_path)

    #     print(f"Artifacts save to {gcs_path}")

    #     return train_path, test_path, scaler_path, encoder_path