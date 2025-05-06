"""
Machine learning model for cryptocurrency price prediction.
This module provides functionality to train linear regression models
for predicting cryptocurrency prices based on historical data.
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def train_model(df, coin_symbol):

    coin_df = df[df['symbol'] == coin_symbol].copy()
    coin_df = coin_df.sort_index()
    coin_df['timestamp'] = coin_df.index.astype('int64') // 10**9  # convert to seconds

    features = coin_df[['timestamp']]
    target = coin_df['current_price']

    if len(features) < 5:
        return None, None

    features_train, features_test, target_train, _ = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(features_train, target_train)

    return model, features_test.iloc[-1].values[0]
