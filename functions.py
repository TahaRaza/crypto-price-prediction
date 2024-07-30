# functions.py
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import datetime
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import io
import base64
import tensorflow as tf

api_key = "abcd8f872aab474db4528eb0581da2f4"

def get_data(symbol="BTC/USD"):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "apikey": api_key,
        "interval": "1day",
        "format": "JSON",
        "symbol": symbol,
        "start_date": "2010-01-01",
        "end_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "type": "none"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        json_response = response.json()
        data = json_response.get('values')
        if data:
            df = pd.DataFrame(data)
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric, errors='coerce')
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values(by='datetime').reset_index(drop=True)
            return df
    return None

def train_model(df):
    df['tomorrow_close'] = df['close'].shift(-1)
    df = df.dropna()
    features = ['open', 'high', 'low', 'close']
    target_columns = ['tomorrow_close']

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_features = feature_scaler.fit_transform(df[features])
    scaled_targets = target_scaler.fit_transform(df[target_columns])

    X, y = [], []
    sequence_length = 10

    for i in range(len(df) - sequence_length):
        X.append(scaled_features[i:i + sequence_length])
        y.append(scaled_targets[i + sequence_length])

    X, y = np.array(X), np.array(y)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

    predictions = model.predict(X_test)
    predictions_rescaled = target_scaler.inverse_transform(predictions)
    y_test_rescaled = target_scaler.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))

    # Predict next 7 days
    last_sequence = scaled_features[-sequence_length:]
    future_predictions = []
    for _ in range(7):
        prediction = model.predict(last_sequence[np.newaxis, :, :])
        future_predictions.append(prediction[0][0])
        last_sequence = np.append(last_sequence[1:], [[prediction[0][0], last_sequence[-1][1], last_sequence[-1][2], last_sequence[-1][3]]], axis=0)

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions_rescaled = target_scaler.inverse_transform(future_predictions)

    accuracy = 100 - (np.mean(np.abs(predictions_rescaled - y_test_rescaled) / y_test_rescaled) * 100)

    # Save the model
    model.save('btc_price_prediction_model.h5')

    # Generate plots
    actual_vs_predicted_plot, future_predictions_plot = generate_plots(
        y_test_rescaled.flatten(), predictions_rescaled.flatten(), future_predictions_rescaled.flatten(), accuracy, rmse, df['datetime'], pd.date_range(df['datetime'].iloc[-1] + pd.Timedelta(days=1), periods=7))

    return df, y_test_rescaled, predictions_rescaled, future_predictions_rescaled, accuracy, rmse, actual_vs_predicted_plot, future_predictions_plot

def generate_plots(actual, predicted, future_predicted, accuracy, rmse, dates, future_dates):
    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.plot(dates[-len(actual):], actual, label='Actual', color='blue')
    plt.plot(dates[-len(predicted):], predicted, label='Predicted', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    actual_vs_predicted_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    # Plot Future Predictions
    plt.figure(figsize=(10, 6))
    plt.plot(future_dates, future_predicted, label='Future Predicted', color='green')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Future Predictions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    future_predictions_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return actual_vs_predicted_plot, future_predictions_plot

def load_model():
    model = tf.keras.models.load_model('btc_price_prediction_model.h5')
    return model
