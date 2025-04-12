import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime

st.set_page_config(page_title="Prediksi Saham LSTM", layout="centered")
st.title("📊 Prediksi Saham Indonesia (LSTM)")
st.write("Masukkan kode saham seperti `BBCA.JK`, `TLKM.JK`, `GOTO.JK`")

ticker = st.text_input("Kode Saham", value="BBCA.JK")

def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length - 7):  # 7 hari ke depan
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + 7])  # prediksi 7 hari
    return np.array(x), np.array(y)

if st.button("Prediksi"):
    try:
        data = yf.download(ticker, start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))
        close_prices = data[['Close']].dropna()

        if close_prices.empty:
            st.error("Data tidak tersedia.")
        else:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(close_prices)

            SEQ_LEN = 60
            X, y = create_sequences(scaled, SEQ_LEN)

            X = X.reshape((X.shape[0], X.shape[1], 1))

            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
                LSTM(50),
                Dense(7)  # prediksi 7 hari
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)

            # Prediksi 7 hari ke depan
            last_sequence = scaled[-SEQ_LEN:]
            input_seq = last_sequence.reshape((1, SEQ_LEN, 1))
            prediction = model.predict(input_seq)
            prediction = scaler.inverse_transform(prediction).flatten()

            st.subheader("Prediksi Harga 7 Hari Kedepan")
            future_dates = pd.date_range(start=close_prices.index[-1] + pd.Timedelta(days=1), periods=7)
            pred_df = pd.DataFrame({"Tanggal": future_dates, "Prediksi Harga": prediction})
            st.dataframe(pred_df.set_index("Tanggal"))

            # Visualisasi
            st.subheader("Grafik Prediksi")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(close_prices.index[-100:], close_prices['Close'][-100:], label='Harga Historis')
            ax.plot(future_dates, prediction, label='Prediksi', marker='o')
            ax.set_title(f"Prediksi Saham {ticker}")
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Harga (IDR)")
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")