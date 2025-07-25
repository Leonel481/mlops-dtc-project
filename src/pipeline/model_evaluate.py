import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

class ModelEvaluate():

    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)


    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, save_metrics_path: str = None):
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

        # Guardar métricas si se desea
        if save_metrics_path:
            with open(save_metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

        return metrics
    
class Metrics():

    def __init__(self):
        self.model = 'model'
