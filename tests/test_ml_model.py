"""Unit tests for the train_model function in the ml_models module."""

import os
import sys
from datetime import datetime, timedelta
import unittest

import numpy as np
import pandas as pd

# Add the parent directory to sys.path to import the ml_model module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_models.ml_model import train_model


class TestTrainModel(unittest.TestCase):
    """Test suite for the train_model function."""

    def setUp(self):
        """Set up mock data for use in tests."""
        dates = [datetime.now() - timedelta(days=i) for i in range(20)]
        self.mock_df = pd.DataFrame({
            'symbol': ['BTC'] * 10 + ['ETH'] * 10,
            'current_price': np.random.uniform(30000, 40000, 20)
        }, index=dates)

    def test_train_model_success(self):
        """Test training with valid BTC data returns a model and timestamp."""
        model, last_timestamp = train_model(self.mock_df, 'BTC')

        self.assertIsNotNone(model)
        self.assertIsNotNone(last_timestamp)
        self.assertTrue(hasattr(model, 'coef_'))
        self.assertTrue(hasattr(model, 'intercept_'))

    def test_train_model_insufficient_data(self):
        """Test that training fails gracefully with too little data."""
        dates = [datetime.now() - timedelta(days=i) for i in range(3)]
        small_df = pd.DataFrame({
            'symbol': ['BTC'] * 3,
            'current_price': np.random.uniform(30000, 40000, 3)
        }, index=dates)

        model, last_timestamp = train_model(small_df, 'BTC')
        self.assertIsNone(model)
        self.assertIsNone(last_timestamp)

    def test_train_model_wrong_symbol(self):
        """Test that training fails when the symbol doesn't exist in the DataFrame."""
        model, last_timestamp = train_model(self.mock_df, 'XRP')
        self.assertIsNone(model)
        self.assertIsNone(last_timestamp)

    def test_model_predictions(self):
        """Test that the trained model can make predictions."""
        model, _ = train_model(self.mock_df, 'BTC')
        test_timestamp = np.array([[self.mock_df.index[0].timestamp()]])
        prediction = model.predict(test_timestamp)
        self.assertIsInstance(prediction[0], (int, float, np.floating))


if __name__ == '__main__':
    unittest.main()
