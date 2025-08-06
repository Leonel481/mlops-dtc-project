import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from google.cloud import bigquery
from typing import Dict, List

# Importamos la clase que vamos a testear.
# Asumimos que esta clase está en un archivo llamado 'model_deploy.py'.
# Si el nombre del archivo es diferente, ajusta la importación.
from src.pipeline.model_deploy import ModelDeploy

class TestModelDeploy(unittest.TestCase):
    """
    Tests unitarios para la clase ModelDeploy, enfocados en la lógica de
    generación de informes y la inserción de datos en BigQuery.
    """
    def setUp(self):
        """
        Configura el entorno de prueba antes de cada test.
        Se usa patch para mockear bigquery.Client antes de instanciar ModelDeploy.
        """
        self.project_id = "test-project"
        self.table_id = "test-dataset.test-table"

        # Mockeamos el cliente de BigQuery antes de crear la instancia de ModelDeploy
        self.mock_bq_client_patcher = patch('src.pipeline.model_deploy.bigquery.Client')
        self.mock_bq_client = self.mock_bq_client_patcher.start()
        
        # Ahora creamos la instancia de ModelDeploy, que usará el mock
        self.model_deploy = ModelDeploy(project=self.project_id, bigquery_table_id=self.table_id)

        # Se crean datos de mock para los tests
        self.mock_reference_data = pd.DataFrame({'feature_a': [1, 2], 'churn': [0, 1]})
        self.mock_current_data = pd.DataFrame({'feature_a': [3, 4], 'churn': [1, 0], 'prediction_proba': [0.9, 0.1]})
        self.mock_rows_to_insert = [
            {"timestamp": "2023-10-27T10:00:00", "metric_type": "accuracy_score", "feature_name": None, "metric_value": 0.95},
            {"timestamp": "2023-10-27T10:00:00", "metric_type": "drift_pvalue", "feature_name": "feature_a", "metric_value": 0.05}
        ]

    def tearDown(self):
        """
        Limpia el entorno de prueba después de cada test.
        """
        self.mock_bq_client_patcher.stop()

    @patch('src.pipeline.model_deploy.ModelDeploy._flatten_report_to_bq_rows')
    @patch('src.pipeline.model_deploy.Report')
    @patch('pandas.Timestamp')
    def test_generate_and_save_report_success(self, mock_timestamp, mock_report, mock_flatten_report):
        """
        Verifica que el método guarde las métricas correctamente en BigQuery
        cuando la inserción es exitosa.
        """
        # Configuración de los mocks
        mock_timestamp.now.return_value.isoformat.return_value = "2023-10-27T10:00:00"
        mock_report_instance = MagicMock()
        mock_report_instance.as_dict.return_value = {"metrics": []}
        mock_report.return_value = mock_report_instance
        mock_flatten_report.return_value = self.mock_rows_to_insert

        # Mockeamos el cliente de BigQuery para que la inserción sea exitosa
        self.mock_bq_client.return_value.insert_rows_json.return_value = []  # Lista vacía indica éxito

        # Ejecutamos el método a testear
        self.model_deploy.generate_and_save_report(self.mock_reference_data, self.mock_current_data)

        # Verificaciones (Assertions)
        mock_flatten_report.assert_called_once()
        self.mock_bq_client.return_value.insert_rows_json.assert_called_once_with(
            self.table_id, self.mock_rows_to_insert
        )

    @patch('src.pipeline.model_deploy.ModelDeploy._flatten_report_to_bq_rows')
    @patch('src.pipeline.model_deploy.Report')
    @patch('pandas.Timestamp')
    def test_generate_and_save_report_failure(self, mock_timestamp, mock_report, mock_flatten_report):
        """
        Verifica que el método maneje errores correctamente cuando la inserción
        en BigQuery falla.
        """
        # Configuración de los mocks
        mock_timestamp.now.return_value.isoformat.return_value = "2023-10-27T10:00:00"
        mock_report_instance = MagicMock()
        mock_report_instance.as_dict.return_value = {"metrics": []}
        mock_report.return_value = mock_report_instance
        mock_flatten_report.return_value = self.mock_rows_to_insert

        # Mockeamos el cliente de BigQuery para simular un error en la inserción
        self.mock_bq_client.return_value.insert_rows_json.return_value = ["error_details"]

        # Ejecutamos el método a testear
        self.model_deploy.generate_and_save_report(self.mock_reference_data, self.mock_current_data)
        
        # Verificaciones (Assertions)
        mock_flatten_report.assert_called_once()
        self.mock_bq_client.return_value.insert_rows_json.assert_called_once_with(
            self.table_id, self.mock_rows_to_insert
        )

    @patch('src.pipeline.model_deploy.ModelDeploy._flatten_report_to_bq_rows')
    @patch('src.pipeline.model_deploy.Report')
    def test_generate_and_save_report_no_metrics(self, mock_report, mock_flatten_report):
        """
        Verifica que el método no intente insertar en BigQuery si no hay métricas para insertar.
        """
        # Mockeamos la función que extrae las métricas para que devuelva una lista vacía
        mock_flatten_report.return_value = []
        mock_report_instance = MagicMock()
        mock_report_instance.as_dict.return_value = {"metrics": []}
        mock_report.return_value = mock_report_instance
        
        # Ejecutamos el método a testear
        self.model_deploy.generate_and_save_report(self.mock_reference_data, self.mock_current_data)

        # Verificaciones (Assertions)
        # Verificar que el método de inserción en BigQuery NO fue llamado
        self.mock_bq_client.return_value.insert_rows_json.assert_not_called()