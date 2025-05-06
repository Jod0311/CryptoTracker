"""Unit tests for the train_lstm_model function in the dl_models module."""

import os
import sys
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential

# Add the parent directory to sys.path to import the lstm_model module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dl_models.lstm_model import train_lstm_model


class TestTrainLSTMModel(unittest.TestCase):
    """Test suite for the train_lstm_model function."""

    def setUp(self):
        """Set up mock data for use in tests."""
        # Create a DataFrame with timestamps and price data
        dates = [datetime.now() - timedelta(hours=i) for i in range(30)]
        self.mock_df = pd.DataFrame({
            'symbol': ['BTC'] * 20 + ['ETH'] * 10,
            'current_price': np.random.uniform(30000, 40000, 30),
            'last_updated': dates
        })

    def test_train_lstm_model_success(self):
        """Test that training with valid BTC data returns a price prediction and no error."""
        predicted_price, error = train_lstm_model(self.mock_df, 'BTC')

        self.assertIsNotNone(predicted_price)
        self.assertIsNone(error)
        self.assertIsInstance(predicted_price, float)
        # The prediction should be in a reasonable range for BTC prices
        self.assertTrue(20000 < predicted_price < 50000)

    def test_train_lstm_model_insufficient_data(self):
        """Test that training fails gracefully with too little data."""
        # Create a small DataFrame with only a few records
        dates = [datetime.now() - timedelta(hours=i) for i in range(5)]
        small_df = pd.DataFrame({
            'symbol': ['BTC'] * 5,
            'current_price': np.random.uniform(30000, 40000, 5),
            'last_updated': dates
        })

        predicted_price, error = train_lstm_model(small_df, 'BTC')
        self.assertIsNone(predicted_price)
        self.assertEqual(error, "Not enough data to train LSTM")

    def test_train_lstm_model_wrong_symbol(self):
        """Test that training fails when the symbol doesn't exist in the DataFrame."""
        predicted_price, error = train_lstm_model(self.mock_df, 'XRP')
        self.assertIsNone(predicted_price)
        self.assertEqual(error, "Not enough data to train LSTM")

    def test_model_structure(self):
        """Test internal model structure by mocking and inspecting the model."""
        # Create a test environment where we can inspect the model
        # This is a more advanced test that checks model structure
        original_fit = Sequential.fit
        original_predict = Sequential.predict
        
        model_structure_correct = [False]
        
        def mock_fit(self, *args, **kwargs):
            # Verify model structure
            if len(self.layers) == 2 and 'lstm' in str(self.layers[0]).lower() and 'dense' in str(self.layers[1]).lower():
                model_structure_correct[0] = True
            return original_fit(self, *args, **kwargs)
        
        def mock_predict(self, *args, **kwargs):
            # Return a predictable value for testing
            return np.array([[0.5]])
        
        try:
            # Apply the mock
            Sequential.fit = mock_fit
            Sequential.predict = mock_predict
            
            # Call the function
            train_lstm_model(self.mock_df, 'BTC')
            
            # Assert that model structure was correct
            self.assertTrue(model_structure_correct[0])
        
        finally:
            # Restore original methods
            Sequential.fit = original_fit
            Sequential.predict = original_predict


if __name__ == '__main__':
    unittest.main()