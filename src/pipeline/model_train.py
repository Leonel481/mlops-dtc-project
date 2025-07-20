from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd


df_ventanas_f = pd.get_dummies(df_ventanas_f, columns=['month_frecuency'], prefix='mes', dtype=int).sort_values(['window_id']).reset_index(drop=True)

train_df = df_ventanas_f[df_ventanas_f['window_id'] < 7]
test_df  = df_ventanas_f[df_ventanas_f['window_id'] == 7]

X_train = train_df.drop(columns=['CustomerID', 'window_id', 'churn'])
y_train = train_df['churn']

X_test = test_df.drop(columns=['CustomerID', 'window_id', 'churn'])
y_test = test_df['churn']

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
