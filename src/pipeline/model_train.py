from google.cloud import aiplatform
import pickle, fsspec
from typing import Any, Dict
from sklearn.metrics import roc_auc_score


class ModelTrain():
    """
    Class for training model churn
    """
    def __init__(self, project: str, location: str, bucket: str):
        """
        Initialize the ModelTrain class.

        Args:
            project (str): GCP Project ID.
            location (str): Region (e.g., "us-central1").
            bucket (str): GCS bucket for artifacts (optional, only for manual saves).
        """
        self.project = project
        self.location = location
        self.bucket = bucket
        self.best_model = None
        self.best_model_name = None
        aiplatform.init(project=project, location=location)
   

    def train_base_models(self, models: Dict, X_train, y_train, X_val, y_val) -> Dict :
        """
        Train multiple models locally and return their ROC-AUC.

        Args:
            models (Dict): Dict with model names and sklearn-compatible estimators.
            X_train: Features for training.
            y_train: Labels for training.
            X_eval: Features for evaluation.
            y_eval: Labels for evaluation.

        Returns:
            Dict[str, float]: Model names and their ROC-AUC scores. 
        """

        scores = {}

        for name, model in models.items():
            print(f"\nEntrenando {name}...")
            model.fit(X_train, y_train)
            score_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            scores[name] = score_auc
            print(f"{name} ROC-AUC: {score_auc:.4f}")
        self.best_model_name = max(scores, key=scores.get)
        self.best_model = models[self.best_model_name]
        return scores

    def register_model_vertex(self, model_name: str)-> aiplatform.Model:




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
