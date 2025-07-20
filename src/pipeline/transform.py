import pandas as pd
import numpy as np
from pathlib import Path

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

    def load_data(self, gcs_path: str) -> pd.DataFrame:
        """
        Run the transformation process.
        Args:
            gcs_path (str): Path to the CSV file in Google Cloud Storage.
        Returns:
            DataFrame: Transformed DataFrame with features for churn prediction.
        """
        df = pd.read_csv(gcs_path)
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

        df_ventanas = pd.DataFrame(window)

        return df_ventanas

    def export_data(self, df: pd.DataFrame, gcs_path: str) -> str:
        """
        Export the transformed data to Google Cloud Storage.

        Args:
            df (DataFrame): Transformed DataFrame.
            gcs_path (str): Path to save the Parquet file in Google Cloud Storage.

        Retrurns:
            str: Path to the exported Parquet file.
        """
        start_date = self.data_start
        end_date = self.data_end
        num_windows = df['window_id'].nunique()

        filename = f'data_{start_date}-{end_date}_{num_windows}window.parquet'
        full_path = Path(gcs_path) / filename

        df.to_parquet(full_path, index=False)
        print(f"Data exported to {full_path}")
        return df