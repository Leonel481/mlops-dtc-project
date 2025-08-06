import pandas as pd
from datetime import datetime
from google.cloud import aiplatform
import pickle, fsspec
from typing import Any, Dict, List, Tuple,  Generator
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import json
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

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

class ModelTrain():
    """
    Class for training model churn
    """
    def __init__(self, project: str, location: str, bucket: str, serving_container_image_uri: str = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest", experiment_name: str = "Churn_Prediction_Experiment"):
        """
        Initialize the ModelTrain class.

        Args:
            project (str): GCP Project ID.
            location (str): Region (e.g., "us-central1").
            bucket (str): GCS bucket for artifacts and as staging_bucket for Vertex AI.
            serving_container_image_uri (str): URI for the pre-built serving container.
        """
        self.project = project
        self.location = location
        self.bucket = bucket
        self.serving_container_image_uri = serving_container_image_uri

        self.experiment_name = experiment_name

        self.full_df = None 
        self.fitted_scaler = None
        self.windows_for_forward_chaining_tuning = None

        self.best_model_name = None
        self.run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        aiplatform.init(project=project, location=location)
   
    def start_main_run(self, job_id: str = None) -> None:
        """
        Starts the main Vertex AI Experiment run for this entire workflow.

        Args:
            job_id (str): Job_id main for experiment run.
        """

        if job_id is None:
            job_id = f'Pipeline-Churn-{self.run_timestamp}'

        self.main_experiment_run = aiplatform.start_run(experiment = self.experiment_name, job_id = job_id)
        self.main_experiment_run.__enter__()

    def end_main_run(self) -> None:
        """
        Ends the main Vertex AI Experiment run.
        """
        if self.main_experiment_run:
            self.main_experiment_run.__exit__(None, None, None)
            self.main_experiment_run = None

    def load_and_prepare_initial_splits(self, gcs_path: str, test_window_size: int = 1 , **kwargs) -> Tuple:
        """
        Load data procesed from Google Clous Storage only for training base models.

        Args:
            gcs_path (str): Path to the CSV file in Google Cloud Storage.

        Returns:
            tuple: (X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test, scaler)
        """
        self.full_df = pd.read_parquet(gcs_path, **kwargs)

        unique_windows = sorted(self.full_df['window_id'].unique())
        final_test_windows_ids = unique_windows[-test_window_size:]
        self.windows_for_forward_chaining_tuning = unique_windows[:-test_window_size]

        # Split data and target
        train_df = self.full_df[self.full_df['window_id'].isin(self.windows_for_forward_chaining_tuning[:-1])]
        valid_df = self.full_df[self.full_df['window_id'] == self.windows_for_forward_chaining_tuning[-1]]
        test_df  = self.full_df[self.full_df['window_id'].isin(final_test_windows_ids)]

        y_train, y_valid, y_test = train_df['churn'], valid_df['churn'], test_df['churn']
        X_train = train_df.drop(columns=['CustomerID', 'window_id', 'churn'])
        X_valid = valid_df.drop(columns=['CustomerID', 'window_id', 'churn'])
        X_test = test_df.drop(columns=['CustomerID', 'window_id', 'churn'])

        self.fitted_scaler = StandardScaler()
        X_train_scaled = self.fitted_scaler.fit_transform(X_train)
        X_valid_scaled = self.fitted_scaler.transform(X_valid)
        X_test_scaled = self.fitted_scaler.transform(X_test)

        return X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid,  y_test
    
    def save_artifacts(self, path_destiny: str, start_date: str, end_date: str , X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test) -> Tuple :
        """
        Save model artifacts (datasets, scaler) as pickles.

        Args:
            gcs_path (str): Base GCS path (e.g., "gs://bucket/artifacts").
            X_train_scaled (np.ndarray): Scaled training features.
            X_valid_scaled (np.ndarray): Scaled validation features.
            X_test_scaled (np.ndarray): Scaled test features.
            y_train (np.ndarray): Training labels.
            y_valid (np.ndarray): Validation labels.
            y_test (np.ndarray): Test labels.

        Returns:
            Tuple[str, str, str, str]: Paths to the saved pickle files.
        """
        filename = f'data_{start_date}_{end_date}.parquet'


        train_path = f"{path_destiny.rstrip('/')}/train_{filename}.pkl"
        valid_path = f"{path_destiny.rstrip('/')}/valid_{filename}.pkl"
        test_path = f"{path_destiny.rstrip('/')}/test_{filename}.pkl"
        scaler_path = f"{path_destiny.rstrip('/')}/scaler_{filename}.pkl"

        dump_pickle((X_train_scaled, y_train), train_path)
        dump_pickle((X_valid_scaled, y_valid), valid_path)
        dump_pickle((X_test_scaled, y_test), test_path)
        dump_pickle(self.fitted_scaler, scaler_path)

        print(f"Artifacts save to {path_destiny}")
        
        return train_path, test_path, scaler_path

    def train_base_models(self, models: Dict, gcs_model_path : str, data_source: str, X_train, y_train, X_val, y_val) -> List :
        """
        Train multiple models and return their ROC-AUC.

        Args:
            models (Dict): Dict with model names and sklearn-compatible estimators.
            gcs_model_path (str): Base GCS path to save initial models.
            X_train: Features for training.
            y_train: Labels for training.
            X_eval: Features for evaluation.
            y_eval: Labels for evaluation.

        Returns:
            results:
            best_model_path: 
        """

        results = []
        best_roc_auc_val = -1

        if self.main_experiment_run is None:
            raise RuntimeError('Main experiment run must be active before training base models.')

        for name, model_instance in models.items():
            self.model_class_map[name] = model_instance.__class__
        
        self.main_experiment_run.log_params({'dataset_source':data_source})

        for name, model_instance in models.items():
            
            current_run_id = f'{name}-training-{self.run_timestamp}'
            
            # Start a sub-run linked to the main experiment run implicitly by `experiment` and `job_id`
            with aiplatform.start_run(experiment=self.experiment_name, job_id=current_run_id, resume=True) as run:
                
                run.log_params({"model_type": name, "split_type": "initial_fixed_split", **model_instance.get_params()})

                model_instance.fit(X_train, y_train)
                y_pred_proba = model_instance.predict_proba(X_val)[:, 1]
                y_pred_class = (y_pred_proba > 0.5).astype(int)
                roc_auc = roc_auc_score(y_val, y_pred_proba)

                metrics = {
                    "roc_auc_score": roc_auc,
                    "precision_churn": precision_score(y_val, y_pred_class, pos_label=1, zero_division=0),
                    "recall_churn": recall_score(y_val, y_pred_class, pos_label=1, zero_division=0),
                    "f1_score_churn": f1_score(y_val, y_pred_class, pos_label=1, zero_division=0),
                    "accuracy": accuracy_score(y_val, y_pred_class),
                    "log_loss": log_loss(y_val, y_pred_proba)
                }

                run.log_metrics(metrics)

                model_path = f'{gcs_model_path.rstrip("/")}/{name}_base_model_{self.run_timestamp}.pkl'
                dump_pickle(model_instance, model_path)
                run.log_artifacts({"base_model_artifact_path": model_path})

                results.append({'name': name, 
                                'score': roc_auc,
                                'path': model_path})

        results = sorted(results, key=lambda x: x['score'], reverse=True)
        self.best_model_name = results[0]['name']
        return results
        
    def registry_best_model(self, results: dict = None )-> aiplatform.Model:
        """
        Register the model in the Vertex AI Model Registry.
        If the model already exists, register a new version.

        Args:
            model_name (str): Display name of the model in Vertex AI.
            gcs_path_model (str): Path for artifact model.
            results (Dict): Dictionary of results to include in the release description.
                                Example: {"roc_auc_score": 0.85, "precision_churn": 0.75, "recall_churn": 0.68}

        Returns:
            aiplatform.Model: Objeto del modelo registrado en Vertex AI.
        """


        version_description = f"Model {self.best_model_name} training at {self.run_timestamp}. ROC AUC (Test): {results[0]['score']:.4f}."

        try:
            existing_models = aiplatform.Model.list(filter=f'display_name="{self.best_model_name}"')
            parent_model_resource_name = existing_models[0].resource_name if existing_models else None

            # Si el modelo existe se crea nueva version
            if parent_model_resource_name:
                model_resource= aiplatform.Model.upload(
                    display_name = self.best_model_name,
                    artifact_uri = results[0]['path'],
                    serving_container_image_uri=self.serving_container_image_uri,
                    parent_model = parent_model_resource_name,
                    version_description = version_description
                )

            # Si el modelo no existe se crea uno
            else:
                model_resource = aiplatform.Model.upload(
                    display_name = self.best_model_name,
                    artifact_uri = results[0]['path'],
                    serving_container_image_uri=self.serving_container_image_uri,
                    is_default_version = True,
                    version_description = version_description
                )
            
            return model_resource

        except Exception as e:
            raise

    def generate_forward_chaining_splits(self, min_train_windows: int = 2, validation_window_size: int = 1) -> Generator[Tuple[Any, Any, Any, Any], None, None] :
        """
        Generates (X_train_scaled, y_train, X_val_scaled, y_val) pairs using forward chaining.

        Args:
            gcs_path (str): Path to the CSV file in Google Cloud Storage.
            min_train_windows (int): Minimum number of initial windows to use for the training set.
            validation_window_size (int): Number of windows to use for the validation set in each step.
            

        Returns:
            tuple: (X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test, scaler)
        """

        unique_windows_for_tuning = self.windows_for_forward_chaining_tuning

        # Iterate through possible validation window starts
        for i in range(min_train_windows, len(unique_windows_for_tuning) - validation_window_size + 1):
            train_windows_ids = unique_windows_for_tuning[:i]
            val_windows_ids = unique_windows_for_tuning[i : i + validation_window_size]

            train_df_fold = self.full_df[self.full_df['window_id'].isin(train_windows_ids)]
            val_df_fold = self.full_df[self.full_df['window_id'].isin(val_windows_ids)]

            y_train_fold, y_val_fold = train_df_fold['churn'], val_df_fold['churn']
            X_train_fold = train_df_fold.drop(columns=['CustomerID', 'window_id', 'churn'])
            X_val_fold = val_df_fold.drop(columns=['CustomerID', 'window_id', 'churn'])
            
            # Use the already fitted scaler
            X_train_scaled_fold = self.fitted_scaler.transform(X_train_fold)
            X_val_scaled_fold = self.fitted_scaler.transform(X_val_fold)
            
            yield X_train_scaled_fold, y_train_fold, X_val_scaled_fold, y_val_fold

    def tune_model(self, gcs_model_tuned_path: str, param_space: Dict, min_train_windows_fc: int = 2 , validation_window_size_fc: int = 1, max_evals: int = 50, **kwargs) -> str:
        """
        Performs hyperparameter tuning on the best base model using Hyperopt and forward chaining validation.
        
        Args:
            gcs_model_tuned_path (str): GCS path to best model base for tune.
            param_space (Dict): Dictionary defining the hyperparameter search space using hyperopt.hp.
            min_train_windows_fc (int): Minimum number of initial windows for the training set in forward chaining.
            validation_window_size_fc (int): Number of windows for the validation set in each forward chaining step.
            max_evals (int): Maximum number of hyperparameter combinations to try.

        Return:
            final_model_gcs_path (str): GCS path final tuned model
        """

        model_class = self.model_class_map[self.best_model_name]

        tuning_run_id = f"tuning-{self.best_model_name}-{self.run_timestamp}"

        with aiplatform.start_run(experiment = self.experiment_name, job_id = tuning_run_id, resume = True) as run:
            run.log_params({'tuning_method': 'HyperoptForwardChaining', 
                            'base_model': self.best_model_name, 
                            'max_evaluations': max_evals})

            trials = Trials()

            # Objective function for Hyperot
            def objective_function(hyperparams):
                """
                Objective function for Hyperopt. It trains the model with the given hyperparams
                across forward chaining folds and returns the negative average ROC AUC.
                Hyperopt minimizes the objective function.
                """

                fold_roc_aucs = []

                for fold_idx, (X_train_fold, y_train_fold, X_val_fold, y_val_fold) in enumerate(self.generate_forward_chaining_splits(min_train_windows_fc, validation_window_size_fc)):

                    try:
                        print(f"DEBUG: Try hyperparams={hyperparams}")
                        print(f"DEBUG: X_train_fold shape: {X_train_fold.shape}")
                        fold_model = model_class(**hyperparams)
                        print(f'Training fold {fold_idx + 1} with hyperparams: {hyperparams}')
                        fold_model.fit(X_train_fold, y_train_fold)
                        
                        y_pred_proba_val_fold = fold_model.predict_proba(X_val_fold)[:, 1]
                        roc_auc_fold = roc_auc_score(y_val_fold, y_pred_proba_val_fold)
                        fold_roc_aucs.append(roc_auc_fold)
                        print(f'Fold {fold_idx + 1} ROC AUC: {roc_auc_fold:.4f}')

                    except Exception as e:
                        print(f'Error during fold {fold_idx} for combination {hyperparams}: {e}')
                        # Hyperopt will try to avoid this
                        return {'loss': float('inf'), 'status': STATUS_OK}
                    
                if fold_roc_aucs:
                    avg_roc_auc = sum(fold_roc_aucs) / len(fold_roc_aucs)

                    run.log_params({f"trial_{len(trials.trials)+1}_hyperparams": json.dumps(hyperparams)})
                    run.log_metrics({f"trial_{len(trials.trials)+1}_avg_roc_auc": avg_roc_auc})

                    # Hyperopt minimizes, so negate AUC
                    return {'loss': -avg_roc_auc, 'status': STATUS_OK}
                else:
                    # Indicate a failed or unviable combination
                    return {'loss': float('inf'), 'status': STATUS_OK} 

            best_hyperopt_result = fmin(
                fn=objective_function,
                space=param_space,
                algo=tpe.suggest, # Tree-structured Parzen Estimator algorithm
                max_evals=max_evals,
                trials=trials,
                rstate=None # Use a fixed seed for reproducibility, e.g., np.random.default_rng(42)
            )

            best_trial = trials.best_trial
            best_hyperparams = best_trial['misc']['vals']
            best_hyperparams = {k: v[0] if isinstance(v, list) else v for k, v in best_hyperparams.items()}
            best_avg_roc_auc = -best_trial['result']['loss']

            print(f'Tuning complete. Best hyperparameters found: {best_hyperparams}')
            print(f'Best average ROC AUC: {best_avg_roc_auc:.4f}')

            self.best_model_metrics = {
                "avg_roc_auc_tuning": best_avg_roc_auc,
                "best_hyperparameters": best_hyperparams
            }

            # Retrain the final 'best' model

            final_train_val_df = self.full_df[self.full_df['window_id'].isin(self.windows_for_forward_chaining_tuning)]
            X_final_train_val = final_train_val_df.drop(columns=['CustomerID', 'window_id', 'churn'])
            y_final_train_val = final_train_val_df['churn']
            X_final_train_val_scaled = self.fitted_scaler.transform(X_final_train_val)

            self.best_model = model_class(**best_hyperparams) # Use the stored class with best params
            self.best_model_name = f"tuned_{self.best_model_name}" # Update name to reflect tuning
            
            # Fit this final model
            self.best_model.fit(X_final_train_val_scaled, y_final_train_val)


            final_model_gcs_path = f'{gcs_model_tuned_path.rstrip("/")}/{self.best_model_name}_final_tuned_model_{self.run_timestamp}.pkl'
            dump_pickle(self.best_model, final_model_gcs_path, **kwargs)

            # Log the final best parameters and metrics for the entire tuning run
            run.log_params({"final_best_hyperparameters": json.dumps(best_hyperparams)})
            run.log_metrics({"final_best_avg_roc_auc_tuning": best_avg_roc_auc})
            aiplatform.log_artifacts({"final_tuned_model_path": final_model_gcs_path})
        
        return final_model_gcs_path

    def evaluate_model(self, gcs_eval_path: str, X_test_scaled, y_test) -> str:
        """
        Evaluates the final best model on the unseen test set.
        Logs metrics to Vertex AI Experiment.

        Args:
            gcs_eval_path (str): 
            X_test_scaled
            y_test

        Return:
            test_results_path
        """

        y_pred_proba_test = self.best_model.predict_proba(X_test_scaled)[:, 1]
        y_pred_class_test = (y_pred_proba_test > 0.5).astype(int)

        metrics_test = {
            "roc_auc_test": roc_auc_score(y_test, y_pred_proba_test),
            "precision_churn_test": precision_score(y_test, y_pred_class_test, pos_label=1, zero_division=0),
            "recall_churn_test": recall_score(y_test, y_pred_class_test, pos_label=1, zero_division=0),
            "f1_score_churn_test": f1_score(y_test, y_pred_class_test, pos_label=1, zero_division=0),
            "accuracy_test": accuracy_score(y_test, y_pred_class_test),
            "log_loss_test": log_loss(y_test, y_pred_proba_test)
        }

        self.final_test_metrics = metrics_test
        self.main_experiment_run.log_metrics(self.final_test_metrics)

        test_results_path = f"{gcs_eval_path.rstrip('/')}/test_results_{self.run_timestamp}.pkl"
        dump_pickle({"y_true": y_test, "y_pred_proba": y_pred_proba_test}, test_results_path)
        self.main_experiment_run.log_artifacts({"final_test_results_path": test_results_path})

        return test_results_path
    
    def register_final_model(self, model_display_name: str, final_model_gcs_path: str, y_test) -> aiplatform.Model:
        """
        Registers the final tuned model in Vertex AI Model Registry.
        This function now also calls _import_model_evaluation to link test metrics.

        Args:
            model_display_name (str): The display name for the model in Vertex AI Model Registry.

        Returns:
            aiplatform.Model: The registered Vertex AI Model object.
        """

        model_display_name_with_ts = f'{model_display_name}-{self.run_timestamp}'
        artifact_dir_uri = final_model_gcs_path.rsplit('/', 1)[0]

        parent_model_resource_name = None
        try:
            existing_models = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
            if existing_models:
                parent_model_resource_name = existing_models[0].resource_name
                print(f'Existing model "{model_display_name}" found. Registering as new version under: {parent_model_resource_name}')
            else:
                print(f'No existing model "{model_display_name}" found. Creating a new model.')
        except Exception as e:
            print(f'Failed to check for existing models: {e}. Attempting to upload a new model.')


        model = aiplatform.Model.upload(
            display_name=model_display_name_with_ts if not parent_model_resource_name else model_display_name, # Use original display name if parenting
            artifact_uri=artifact_dir_uri,
            serving_container_image_uri=self.serving_container_image_uri, 
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            serving_container_ports=[8080],
            parent_model=parent_model_resource_name,
            is_default_version= (not parent_model_resource_name), # Set as default only if creating new model
            version_description=f"Model {self.best_model_name} trained at {self.run_timestamp}. ROC AUC (Test): {self.final_test_metrics.get('roc_auc_test', 'N/A'):.4f}.",
            sync=True
        )
        print(f'Model "{model.display_name}" registered with ID: {model.resource_name}')
        self.registered_model = model
        self._import_model_evaluation(self.main_experiment_run.name, model_display_name_with_ts, y_test)

        return model

    def _import_model_evaluation(self, experiment_run_resource_name: str, model_name: str, y_test) -> str:
        """
        Imports model evaluation metrics into the Vertex AI Model resource, linking it to the experiment run.
        This method assumes the evaluation metrics (self.final_test_metrics) are already computed
        and the model (self.registered_model) is already registered.
        It does NOT start a new experiment run.
        """

        try:
            # Prepare metrics for ModelEvaluation
            classification_metrics = ClassificationMetrics(
                au_roc=self.final_test_metrics.get("roc_auc_test"),
                precision=self.final_test_metrics.get("precision_churn_test"),
                recall=self.final_test_metrics.get("recall_churn_test"),
                f1_score=self.final_test_metrics.get("f1_score_churn_test"),
                accuracy=self.final_test_metrics.get("accuracy_test"),
                log_loss=self.final_test_metrics.get("log_loss_test"),
            )
            
            model_metrics = ModelEvaluationMetrics(
                classification_metrics=classification_metrics
            )

            # GAPICModelEvaluation takes proto types
            model_evaluation = GAPICModelEvaluation(
                metrics=model_metrics,
            )

            # Get the ModelService client
            client_options = {"api_endpoint": f"{self.location}-aiplatform.googleapis.com"}
            model_client = ModelServiceClient(client_options=client_options)

            # Construct the parent for the ModelEvaluation (the registered model's resource name)
            parent_model_resource = self.registered_model.resource_name

            # Request to import model evaluation
            response = model_client.import_model_evaluation(
                parent=parent_model_resource,
                model_evaluation=model_evaluation,
                external_evaluation_dataset={
                    "data_item_count": len(y_test), # Use the stored y_test
                    # "gcs_source": {"uris": [your_test_data_gcs_uri]} # Add GCS URI if available
                },
                metadata={"experimentRun": experiment_run_resource_name}, # Link to the main experiment run
            )
            print(f"Model evaluation imported successfully: {response.name}")

        except Exception as e:
            print(f"Failed to import model evaluation for model '{model_display_name}': {e}", exc_info=True)
            # Log failure to the main experiment run
            if self.main_experiment_run:
                self.main_experiment_run.log_params({"model_evaluation_import_status": "FAILED", "error_message": str(e)})







 