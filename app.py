"""Streamlit Cryptocurrency Tracker App with ML & Gemini AI Suggestions."""

import os
import hashlib
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from ml_models.ml_model import train_model
from dl_models.lstm_model import train_lstm_model
from qnn_models.qnn_model import train_qnn_model


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#test1234

def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user_table():
    """Create the user table in SQLite if not exists."""
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, password TEXT)')
    conn.commit()
    conn.close()

def add_user(username, password):
    """Add a new user with hashed password."""
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute('INSERT INTO users(username, password) VALUES (?, ?)',
              (username, hash_password(password)))
    conn.commit()
    conn.close()

def login_user(username, password):
    """Authenticate a user."""
    conn = sqlite3.connect('data/users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?',
              (username, hash_password(password)))
    data = c.fetchone()
    conn.close()
    return data

def fetch_data():
    """Fetch cryptocurrency data from SQLite DB."""
    try:
        conn = sqlite3.connect('data/database.db')
        query = "SELECT * FROM cryptocurrency_data"
        df = pd.read_sql(query, conn)
        conn.close()
        df['last_updated'] = pd.to_datetime(df['last_updated'])
        df.set_index('last_updated', inplace=True)
        return df
    except sqlite3.DatabaseError as e:
        st.error(f"Failed to fetch data from database: {e}")
        return None


def show_basic_info(df):
    st.title("Cryptocurrency Dashboard")
    unique_coins_df = df.drop_duplicates(subset=['symbol'])
    suggestions = []

    for _, row in unique_coins_df.iterrows():
        with st.container():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(row['image'], width=64)
                st.header(f"{row['name']} ({row['symbol'].upper()})")
                st.subheader(f"Price: ${row['current_price']:.2f}")
                st.markdown(f"Market Cap: ${row['market_cap']:,}")
                st.markdown(f"24H Change: {row['price_change_percentage_24h']:.2f}%")
                st.subheader("Market Stats")
                st.markdown(f"Volume (24H): ${row['total_volume']:,}")
                st.markdown(f"High (24H): ${row['high_24h']:.2f}")
                st.markdown(f"Low (24H): ${row['low_24h']:.2f}")

                coin_df = df[df['symbol'] == row['symbol']].copy()

                # **ML Prediction**
                model, latest_time = train_model(df, row['symbol'])
                if model:
                    future_time = [[latest_time + 3600]]  # 1 hour later
                    ml_predicted_price = model.predict(future_time)[0]
                    st.markdown(f"🔍 **ML Predicted Price (1h later):** ${ml_predicted_price:.2f}")
                    suggestions.append((row['name'], ml_predicted_price))

                # **LSTM Prediction**
                with st.spinner("Training LSTM..."):
                    lstm_pred, error = train_lstm_model(df.reset_index(), row['symbol'])
                if error:
                    st.markdown("🤖 **DL (LSTM) Prediction:** Not available")
                else:
                    st.markdown(f"🤖 **DL (LSTM) Predicted Price:** ${lstm_pred:.2f}")

                # **QNN Prediction**
                with st.spinner("Training QNN..."):
                    qnn_pred, qnn_error = train_qnn_model(df, row['symbol'])
                if qnn_error:
                    st.markdown(f"🔮 **QNN Prediction:** {qnn_error}")
                else:
                    st.markdown(f"🔮 **QNN Predicted Price (1h later):** ${qnn_pred:.2f}")
                    suggestions.append((row['name'], qnn_pred))

            with col2:
                st.subheader("Historical Price Chart")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(coin_df.index, coin_df['current_price'], label="Price", color="green")
                ax.axhline(row['high_24h'], color="red", linestyle="--", label="High (24H)")
                ax.axhline(row['low_24h'], color="blue", linestyle="--", label="Low (24H)")
                ax.axhline(row['current_price'], color="yellow", linestyle="--", label="Current Price")
                max_price = max(coin_df['current_price'].max(), row['high_24h'])
                ax.set_ylim(0, max_price * 1.1)
                ax.set_xlabel("Time")
                ax.set_ylabel("Price (USD)")
                ax.set_title(f"Price Trend: {row['name']}")
                ax.legend()
                st.pyplot(fig)

        st.markdown("---")

    show_gemini_suggestions(suggestions)

def show_gemini_suggestions(suggestions):
    """Display AI-generated investment suggestions from Gemini."""
    if suggestions:
        suggestions.sort(key=lambda x: x[1], reverse=True)
        top_3 = suggestions[:3]
        coin_list = [f"{coin} (Predicted: ${pred:.2f})" for coin, pred in top_3]
        prompt = (
            "Based on the predicted prices below, which cryptocurrency should a user consider buying?\n"
            + "\n".join(coin_list)
        )
        st.subheader("Gemini Crypto Suggestions")
        if st.button("Get AI Suggestions"):
            try:
                response = genai.GenerativeModel("gemini-2.0-flash").generate_content(
                    contents=[{"parts": [{"text": prompt}]}])
                st.info(response.parts[0].text)
            except Exception as e:
                st.error(f"Error fetching AI suggestions: {e}")

def cumulative_graph(df):
    """Plot cumulative price trends of all cryptocurrencies."""
    st.title("Cumulative Cryptocurrency Price Chart")
    cumulative_df = df.groupby(['last_updated', 'name'])['current_price'].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 6))
    for coin in cumulative_df.columns:
        ax.plot(cumulative_df.index, cumulative_df[coin], label=coin)
    ax.set_title("Cumulative Cryptocurrency Price Trends")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")
    ax.legend(title="Cryptocurrencies")
    st.pyplot(fig)

def login_register():
    """Handle login and registration via sidebar."""
    create_user_table()
    st.sidebar.subheader("User Login / Registration")
    option = st.sidebar.radio("Action", ["Login", "Register"])

    if option == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success(f"Welcome {username}!")
            else:
                st.error("Incorrect username or password")

    elif option == "Register":
        new_user = st.sidebar.text_input("New Username")
        new_pass = st.sidebar.text_input("New Password", type="password")
        if st.sidebar.button("Register"):
            try:
                add_user(new_user, new_pass)
                st.success("Registration successful! You can now login.")
            except sqlite3.IntegrityError:
                st.error("Username already exists.")

    if "logged_in" in st.session_state and st.session_state["logged_in"]:
        st.sidebar.success(f"Logged in as {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()

def main():
    """Main function to launch the Streamlit dashboard."""
    login_register()
    if st.session_state.get("logged_in"):
        df = fetch_data()
        if df is not None and not df.empty:
            show_basic_info(df)
            cumulative_graph(df)
        else:
            st.error("No data available.")
    else:
        st.warning("Please log in to view the dashboard.")

if __name__ == "__main__":
    main()
#hello