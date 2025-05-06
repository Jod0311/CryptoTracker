"""LSTM model implementation for cryptocurrency price prediction."""

# pylint: disable=import-error,no-name-in-module
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def train_lstm_model(df, coin_symbol):
    """
    Train an LSTM model to predict cryptocurrency prices.
    
    Args:
        df (pandas.DataFrame): DataFrame containing cryptocurrency data
        coin_symbol (str): Symbol of the cryptocurrency to train on (e.g., 'BTC')
        
    Returns:
        tuple: (predicted_price, error_message)
            - predicted_price (float): The predicted price or None if error
            - error_message (str): Error message or None if successful
    """
    coin_df = df[df['symbol'] == coin_symbol].copy()
    coin_df.sort_values('last_updated', inplace=True)

    if len(coin_df) < 10:
        return None, "Not enough data to train LSTM"

    prices = coin_df['current_price'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)

    x_data = []
    y_data = []
    time_steps = 3
    for i in range(time_steps, len(prices_scaled)):
        x_data.append(prices_scaled[i - time_steps:i])
        y_data.append(prices_scaled[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(x_data.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_data, y_data, epochs=50, verbose=0)

    last_sequence = prices_scaled[-time_steps:].reshape(1, time_steps, 1)
    predicted_scaled = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(predicted_scaled)

    return float(predicted_price[0][0]), None
