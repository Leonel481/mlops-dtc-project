from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd

class ModelTrain():
    """
    Class for training model churn
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


    def load_data(self, df: pd.DataFrame, window_target) -> list:
        
        df_ventanas_f = pd.get_dummies(df, columns=['month_frecuency'], prefix='mes', dtype=int).sort_values(['window_id']).reset_index(drop=True)

        train_df = df_ventanas_f[df_ventanas_f['window_id'] < window_target]
        test_df  = df_ventanas_f[df_ventanas_f['window_id'] == window_target]

        X_train = train_df.drop(columns=['CustomerID', 'window_id', 'churn'])
        y_train = train_df['churn']

        X_test = test_df.drop(columns=['CustomerID', 'window_id', 'churn'])
        y_test = test_df['churn']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def model_train():

        # Diccionario de modelos (con mínimos ajustes para evitar warnings)
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            "LightGBM": LGBMClassifier(random_state=42)
        }

        results = {}

        for name, model in models.items():
            print(f"\nEntrenando {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            results[name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "ROC AUC": roc_auc_score(y_test, y_proba)
            }

        # Mostrar resultados ordenados por ROC AUC
        results_df = pd.DataFrame(results).T.sort_values(by="ROC AUC", ascending=False)
        print("\nResultados comparativos:")
        print(results_df)

        # Mostrar matriz de confusión del mejor modelo
        best_model_name = results_df.index[0]
        best_model = models[best_model_name]
        print(f"\nMejor modelo: {best_model_name}")

        y_pred_best = best_model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred_best)
