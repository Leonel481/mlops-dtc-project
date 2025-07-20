import pandas as pd
import numpy as np


def load_data(gcs_path: str) -> pd.DataFrame:    
    """
    Load Data raw, filter CustomerId nan, filter valid Description.

    Args:
        gcs_path (str): Path to the CSV file in Google Cloud Storage.

    Returns:
        Dataframe: DataFrame with the raw data.
    """
    df = pd.read_csv(gcs_path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    df = df[~df['Description'].isin(['Manual','Discount','This is a test product.'])]
    return df


def group_daily_dates(df: pd.DataFrame) -> pd.DataFrame:
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
    
    df_final = df_diario[
                    (df_diario['Items_net'] != 0) &
                    (df_diario['Total_value'] != 0)
                    ]
    return df_final

def transform_data(df_final: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the data to create features for churn prediction.

    Args:
        df (DataFrame): DataFrame from group_daily_dates function.
    
    Returns:
        DataFrame: Transformed DataFrame with features for churn prediction.
    """

    # Ventasas churn
    meses_obs = 3
    meses_churn = 3
    obs_ini = df_final['InvoiceDate'].min() # 2010-12-01

    window = []
    window_id = 1

    while True:
        obs_end = obs_ini + pd.DateOffset(months=meses_obs) - pd.DateOffset(days=1)
        churn_ini = obs_end + pd.DateOffset(days=1)
        churn_end = churn_ini + pd.DateOffset(months=meses_churn) - pd.DateOffset(days=1)

        if churn_end > df_final['InvoiceDate'].max():
            break

        df_obs = df_final[(df_final['InvoiceDate'] >= obs_ini) & (df_final['InvoiceDate'] <= obs_end)]
        df_churn = df_final[(df_final['InvoiceDate'] >= churn_ini) & (df_final['InvoiceDate'] <= churn_end) & (df_final['Items_net'] > 0)]

        customers_obs = df_obs['CustomerID'].dropna().unique()
        customer_churn = df_churn['CustomerID'].dropna().unique()

        for customer in customers_obs:

            df_cliente = df_obs[df_obs['CustomerID'] == customer].copy()
            # df_cliente['quantity_buy'] = np.where(df_cliente['Quantity'] > 0, df_cliente['Quantity'], 0)
            # df_cliente['quantity_return'] = np.where(df_cliente['Quantity'] < 0, -df_cliente['Quantity'],0)

            # Compras
            total_products_buys = df_cliente['Items_buy'].sum()
            value_buys = df_cliente['Value_buy'].sum()
            invoices_buy = df_cliente['Invoices'].sum()
            avg_invoice_buy = value_buys / invoices_buy if invoices_buy > 0 else 0
            # promedio_monto_producto = total_gasto / total_productos_comprados if total_productos_comprados > 0 else 0
            products_unique_buys = df_cliente['Unique_products_buy'].nunique()

            # Devoluciones
            total_products_return = df_cliente['Items_return'].nunique()
            value_return = df_cliente['Value_return'].sum()
            invoices_return = df_cliente['Invoices_canceled'].sum()
            avg_invoice_return = value_return / invoices_return if invoices_return > 0 else 0
            products_unique_return = df_cliente['Unique_products_return'].nunique()

            # mes frecuente
            mes_frecuente = df_cliente['InvoiceDate'].dt.month.mode()

            # Total
            total_products = df_cliente['Items_net'].sum()
            total_value_obs = df_cliente['Total_value'].sum()
            total_value_std = df_cliente['Total_value'].std(ddof=0)

            # Variables of time
            customer_longevity = (df_churn['InvoiceDate'].max() - df_cliente['InvoiceDate'].min()).days
            recency_days = (df_obs['InvoiceDate'].max() - df_cliente['InvoiceDate'].max()).days
            fechas = df_cliente['InvoiceDate'].drop_duplicates().sort_values()
            if len(fechas) > 1:
                diffs = fechas.diff().dropna()
                promedio_dias_entre_compras = diffs.mean().days
            else:
                promedio_dias_entre_compras = (df_obs['InvoiceDate'].max() - df_cliente['InvoiceDate'].max()).days

            # Ratio
            return_rate = total_products_return / total_products_buys if total_products_buys > 0 else np.nan
            return_value_rate = value_return / value_buys if value_buys > 0 else np.nan
            
            # Value in window churn
            value_churn = df_churn[df_churn['CustomerID'] == customer]['Total_value'].sum()

            # Churn
            churn = 1 if (customer not in customer_churn or value_churn <= 0.2 * total_value_obs) else 0

            # Filtrar que la ventana no tenga solo devoluciones

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
                'days_between_purchases': promedio_dias_entre_compras,
                'month_frecuency': mes_frecuente.iloc[0] if not mes_frecuente.empty else np.nan,
                'churn': churn
            })

        obs_ini = obs_ini + pd.DateOffset(months=meses_obs)
        window_id += 1

    df_ventanas = pd.DataFrame(window)
    return df_ventanas
