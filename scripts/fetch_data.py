"""Fetch and store cryptocurrency market data from the CoinGecko API."""

from datetime import datetime, timedelta

import pandas as pd
import requests
import sqlalchemy as db


def fetch_data():
    """Fetches cryptocurrency market data from the CoinGecko API and simulates hourly data."""
    try:
        url = 'https://api.coingecko.com/api/v3/coins/markets'
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 10,
            'page': 1,
            'sparkline': False,
            'price_change_percentage': '24h'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        df = pd.DataFrame(data)

        columns_to_keep = [
            'id', 'symbol', 'name', 'image', 'current_price', 'market_cap',
            'total_volume', 'high_24h', 'low_24h', 'price_change_percentage_24h',
            'circulating_supply', 'total_supply', 'last_updated'
        ]
        df = df[columns_to_keep]

        current_time = datetime.now()
        records = []
        for i in range(12):
            time_point = current_time - timedelta(hours=i)
            temp_df = df.copy()
            temp_df['last_updated'] = time_point
            temp_df['current_price'] *= (1 + (i % 5) / 100)
            records.append(temp_df)

        df_combined = pd.concat(records, ignore_index=True)
        print(f"Fetched {len(df_combined)} records with timestamps.")
        return df_combined

    except requests.exceptions.RequestException as req_err:
        print(f"HTTP request failed: {req_err}")
    except Exception as e:
        print(f"Unexpected error occurred while fetching data: {e}")
    return None


def store_data(df):
    """Stores the given DataFrame into a SQLite database."""
    try:
        engine = db.create_engine('sqlite:///data/database.db')
        df.to_sql('cryptocurrency_data', con=engine, if_exists='replace', index=False)
        print("Data stored in database successfully.")
    except Exception as e:
        print(f"Failed to store data in database: {e}")


def main():
    """Main function to orchestrate data fetching and storing."""
    df = fetch_data()
    if df is not None and not df.empty:
        store_data(df)
    else:
        print("No valid data to store in the database.")


if __name__ == "__main__":
    main()

