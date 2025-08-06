import pytest
from hyperopt import hp
from hyperopt.pyll import scope
from hyperopt import Trials
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import logging
from src.pipeline.model_train import ModelTrain
from test.config import model_train_instance, mock_df, mock_scalar_data, DummyModelForTest, minio_path_data_clean, storage_options

logging.basicConfig(level=logging.INFO)

@patch('src.pipeline.model_train.pd.read_parquet')
@patch('src.pipeline.model_train.StandardScaler')
def test_load_and_prepare_initial_splits(mock_scaler, mock_read_parquet, model_train_instance, mock_df):
    mock_read_parquet.return_value = mock_df
    mock_scaler.return_value = MagicMock()
    mock_scaler.return_value.fit_transform.return_value = np.random.rand(6, 2)
    mock_scaler.return_value.transform.side_effect = [np.random.rand(2, 2), np.random.rand(2, 2)]

    X_train, X_valid, X_test, y_train, y_valid, y_test = model_train_instance.load_and_prepare_initial_splits(gcs_path="gs://dummy-path")
    
    assert isinstance(X_train, np.ndarray)
    assert len(X_train) == 6
    assert len(X_valid) == 2
    assert len(X_test) == 2
    assert isinstance(model_train_instance.fitted_scaler, MagicMock)
    assert 'window_id' in model_train_instance.full_df.columns
    mock_read_parquet.assert_called_once()
    mock_scaler.return_value.fit_transform.assert_called_once()
    mock_scaler.return_value.transform.call_count == 2

@patch('src.pipeline.model_train.dump_pickle')
@patch('src.pipeline.model_train.aiplatform.start_run')
def test_train_base_models(mock_start_run, mock_dump_pickle, model_train_instance):
    mock_run_instance = MagicMock()
    mock_start_run.return_value.__enter__.return_value = mock_run_instance
    mock_model1 = MagicMock()
    mock_model1.get_params.return_value = {'param': 'value1'}
    mock_model1.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2]]) # Simulates high ROC AUC
    mock_model2 = MagicMock()
    mock_model2.get_params.return_value = {'param': 'value2'}
    mock_model2.predict_proba.return_value = np.array([[0.6, 0.4], [0.7, 0.3]]) # Simulates low ROC AUC
    
    models = {'Model1': mock_model1, 'Model2': mock_model2}
    
    # Setup the internal state of the ModelTrain instance
    model_train_instance.main_experiment_run = MagicMock()
    model_train_instance.model_class_map = {name: MagicMock for name in models.keys()}

    # Run the method under test
    results = model_train_instance.train_base_models(models, "gs://model-path", "source", np.random.rand(10, 2), pd.Series([1,0,1,0,1,0,1,0,1,0]), np.random.rand(2, 2), pd.Series([1,0]))
    
    # Assertions
    assert len(results) == 2
    assert model_train_instance.best_model_name == 'Model1'
    assert mock_model1.fit.called
    assert mock_model2.fit.called
    assert mock_dump_pickle.call_count == 2
    assert mock_start_run.call_count == 2

@patch('src.pipeline.model_train.dump_pickle')
@patch('src.pipeline.model_train.fmin')
@patch('src.pipeline.model_train.ModelTrain.generate_forward_chaining_splits')
@patch('src.pipeline.model_train.aiplatform.start_run')
@patch('src.pipeline.model_train.Trials')
@patch('src.pipeline.model_train.aiplatform')
def test_tune_model(mock_aiplatform, mock_trials_cls, mock_start_run, mock_generator, mock_fmin, mock_dump_pickle, model_train_instance, mock_df):
    """Tests the hyperparameter tuning process with Hyperopt."""
    
    # Mock
    mock_dummy_model = MagicMock()
    mock_instance_of_model = MagicMock()
    mock_dummy_model.return_value = mock_instance_of_model

    # Aiplatform mock
    mock_run = MagicMock()
    mock_aiplatform.start_run.return_value.__enter__.return_value = mock_run
   
    # Mock Hyperopt to return a specific best result instantly
    mock_trials = MagicMock()
    mock_trials.best_trial = {
        'misc': {'vals': {'max_depth': [10], 'learning_rate': [0.1]}}, 
        'result': {'loss': -0.95, 'status': 'ok'}
    }
    mock_trials_cls.return_value = mock_trials
    mock_generator.return_value = iter([(np.random.rand(5, 2), pd.Series([0,1,0,1,0]), np.random.rand(2, 2), pd.Series([1,0]))])
    mock_fmin.return_value = {'max_depth': 10, 'learning_rate': 0.1}

    # Call Objective function for Hyperot
    def fmin_side_effect(fn, space, algo, max_evals, trials, rstate=None):
        fn({'max_depth': 10, 'learning_rate': 0.1})
        return {'max_depth': 10, 'learning_rate': 0.1}
    
    mock_fmin.side_effect = fmin_side_effect
    
    # Setup the ModelTrain instance's state
    model_train_instance.best_model_name = 'DummyModel'
    # model_train_instance.model_class_map = {'DummyModel': DummyModelForTest} 
    model_train_instance.model_class_map = {'DummyModel': mock_dummy_model}
    model_train_instance.main_experiment_run = MagicMock()
    model_train_instance.full_df = mock_df
    model_train_instance.windows_for_forward_chaining_tuning = mock_df['window_id'].unique().tolist()[:-1]

    mock_fitted_scaler = MagicMock()
    mock_fitted_scaler.transform.return_value = np.random.rand(len(mock_df), 2)
    model_train_instance.fitted_scaler = mock_fitted_scaler

    param_space = {
        'max_depth': hp.quniform('max_depth', 4, 15, 1),
        'learning_rate': hp.loguniform('learning_rate', -3, 0)
    }

    final_path = model_train_instance.tune_model(minio_path_data_clean(), param_space, max_evals=1, storage_options = storage_options())
    
    # Assertions
    mock_fmin.assert_called_once()
    assert 'tuned_DummyModel_final_tuned_model' in final_path
    assert model_train_instance.best_model_name == 'tuned_DummyModel'
    # mock_instance_of_model.fit.assert_called_twice()
    assert mock_instance_of_model.fit.call_count == 2

    mock_dump_pickle.assert_called_once_with(
        mock_instance_of_model,
        final_path,
        storage_options=storage_options()
    )

@patch('src.pipeline.model_train.dump_pickle')
@patch('src.pipeline.model_train.roc_auc_score')
def test_evaluate_model(mock_roc_auc_score, mock_dump_pickle, model_train_instance):
    """Tests the final model evaluation on the test set."""
    
    # Setup mocks
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9], [0.9, 0.1], [0.8, 0.2]])
    mock_model.predict.return_value = np.array([1, 0, 0])
    mock_roc_auc_score.return_value = 0.85
    
    # Setup the ModelTrain instance's state
    model_train_instance.best_model = mock_model
    model_train_instance.main_experiment_run = MagicMock()
    model_train_instance.y_test_final = pd.Series([1, 0, 0]) # Mocking this for the test
    
    # Run the method under test
    test_results_path = model_train_instance.evaluate_model("gs://eval-path", np.random.rand(3, 2), pd.Series([1, 0, 0]))

    # Assertions
    assert 'roc_auc_test' in model_train_instance.final_test_metrics
    assert model_train_instance.final_test_metrics['roc_auc_test'] == 0.85
    model_train_instance.best_model.predict_proba.assert_called_once()
    assert 'test_results' in test_results_path
    mock_dump_pickle.assert_called_once()
    model_train_instance.main_experiment_run.log_metrics.assert_called_once()
    