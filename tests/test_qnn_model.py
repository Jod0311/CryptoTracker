"""Unit tests for the train_qnn_model function in the qnn_models module."""

import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pennylane as qml

# Add the parent directory to sys.path to import the qnn_model module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qnn_models.qnn_model import train_qnn_model, QNNModel


class TestTrainQNNModel(unittest.TestCase):
    """Test suite for the train_qnn_model function."""

    def setUp(self):
        """Set up mock data for use in tests."""
        # Create a DataFrame with timestamps and price data
        dates = [datetime.now() - timedelta(hours=i) for i in range(30)]
        self.mock_df = pd.DataFrame({
            'symbol': ['BTC'] * 20 + ['ETH'] * 10,
            'current_price': np.random.uniform(30000, 40000, 30),
            'last_updated': dates
        })

    @patch('qnn_models.qnn_model.QNNModel')
    def test_train_qnn_model_success(self, mock_qnn_model):
        """Test that training with valid BTC data returns a price prediction and no error."""
        # Mock the QNN model to avoid actual quantum computation
        mock_instance = MagicMock()
        mock_instance.predict.return_value = np.array([0.5])
        mock_qnn_model.return_value = mock_instance

        predicted_price, error = train_qnn_model(self.mock_df, 'BTC')

        self.assertIsNotNone(predicted_price)
        self.assertIsNone(error)
        self.assertIsInstance(predicted_price, float)
        self.assertTrue(mock_instance.train.called)
        self.assertTrue(mock_instance.predict.called)

    def test_train_qnn_model_insufficient_data(self):
        """Test that training fails gracefully with too little data."""
        # Create a small DataFrame with only a few records
        dates = [datetime.now() - timedelta(hours=i) for i in range(5)]
        small_df = pd.DataFrame({
            'symbol': ['BTC'] * 5,
            'current_price': np.random.uniform(30000, 40000, 5),
            'last_updated': dates
        })

        predicted_price, error = train_qnn_model(small_df, 'BTC')
        self.assertIsNone(predicted_price)
        self.assertEqual(error, "Not enough data to train QNN")

    def test_train_qnn_model_wrong_symbol(self):
        """Test that training fails when the symbol doesn't exist in the DataFrame."""
        predicted_price, error = train_qnn_model(self.mock_df, 'XRP')
        self.assertIsNone(predicted_price)
        self.assertEqual(error, "Not enough data to train QNN")

    @patch('qnn_models.qnn_model.QNNModel')
    def test_model_exception_handling(self, mock_qnn_model):
        """Test that exceptions are properly caught and handled."""
        # Make the model raise an exception when trained
        mock_instance = MagicMock()
        mock_instance.train.side_effect = Exception("Test exception")
        mock_qnn_model.return_value = mock_instance

        predicted_price, error = train_qnn_model(self.mock_df, 'BTC')
        self.assertIsNone(predicted_price)
        self.assertTrue(error.startswith("Error in QNN:"))
        self.assertIn("Test exception", error)

    def test_qnn_model_class(self):
        """Test the QNNModel class directly."""
        # This is a more basic test that just verifies the class instantiates
        model = QNNModel(num_qubits=2, num_layers=1)
        self.assertEqual(model.num_qubits, 2)
        self.assertEqual(model.num_layers, 1)
        self.assertIsNotNone(model.weights)
        self.assertEqual(model.weights.shape, (1, 2))

    @patch('pennylane.qnode')
    def test_qnn_circuit(self, mock_qnode):
        """Test the quantum circuit definition."""
        # This is more of a smoke test to ensure the quantum circuit code runs
        mock_qnode.return_value = lambda weights, x: 0.5  # Mock the quantum circuit
        
        # Create simplified test data
        X_test = np.array([[0.1, 0.2, 0.3]])
        y_test = np.array([0.4])
        
        # Create a model with small epoch count to speed up test
        with patch('qnn_models.qnn_model.qnn', return_value=0.5):
            model = QNNModel(num_qubits=2, num_layers=1, learning_rate=0.1)
            # Just make sure these don't crash
            model.train(X_test, y_test, num_epochs=1)
            pred = model.predict(X_test)
            self.assertIsInstance(pred, np.ndarray)


if __name__ == '__main__':
    unittest.main()