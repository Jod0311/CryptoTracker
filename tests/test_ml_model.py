import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import patch

# Add the parent directory to sys.path to import the ml_model module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_models.ml_model import train_model

class TestTrainModel(unittest.TestCase):
    
    def setUp(self):
        # Create a mock DataFrame for testing
        dates = [datetime.now() - timedelta(days=i) for i in range(20)]
        self.mock_df = pd.DataFrame({
            'symbol': ['BTC'] * 10 + ['ETH'] * 10,
            'current_price': np.random.uniform(30000, 40000, 20)
        }, index=dates)
    
    def test_train_model_success(self):
        # Test with sufficient data
        model, last_timestamp = train_model(self.mock_df, 'BTC')
        
        # Check if model is created
        self.assertIsNotNone(model)
        # Check if last_timestamp is returned
        self.assertIsNotNone(last_timestamp)
        # Verify model has expected attributes
        self.assertTrue(hasattr(model, 'coef_'))
        self.assertTrue(hasattr(model, 'intercept_'))
    
    def test_train_model_insufficient_data(self):
        # Create a small DataFrame with less than 5 rows
        dates = [datetime.now() - timedelta(days=i) for i in range(3)]
        small_df = pd.DataFrame({
            'symbol': ['BTC'] * 3,
            'current_price': np.random.uniform(30000, 40000, 3)
        }, index=dates)
        
        # Test with insufficient data
        model, last_timestamp = train_model(small_df, 'BTC')
        
        # Both should be None
        self.assertIsNone(model)
        self.assertIsNone(last_timestamp)
    
    def test_train_model_wrong_symbol(self):
        # Test with symbol not in DataFrame
        model, last_timestamp = train_model(self.mock_df, 'XRP')
        
        # Should return None for both as there's no data for XRP
        self.assertIsNone(model)
        self.assertIsNone(last_timestamp)
    
    def test_model_predictions(self):
        # Test that the model can make predictions
        model, _ = train_model(self.mock_df, 'BTC')
        
        # Create a test timestamp
        test_timestamp = np.array([[self.mock_df.index[0].timestamp()]])
        
        # Make a prediction
        prediction = model.predict(test_timestamp)
        
        # Check prediction is a number
        self.assertTrue(isinstance(prediction[0], (int, float)))

if __name__ == '__main__':
    unittest.main()