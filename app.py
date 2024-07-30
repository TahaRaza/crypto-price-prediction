import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, jsonify
import datetime
import pandas as pd
import numpy as np
from functions import get_data, train_model, load_model, mean_squared_error, save_plots, load_plots
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symbol = data.get('symbol')
    model_option = data.get('model_option')

    df = get_data(symbol)

    if df is not None:
        if model_option == 'train':
            df, actual, predicted, future_predicted, accuracy, rmse, actual_vs_predicted_plot, future_predictions_plot = train_model(df)
            
            # Save plots for later use with the pre-trained model
            save_plots(actual_vs_predicted_plot, future_predictions_plot)
            
            # Filter data from 2024-01-01 onwards
            filtered_df = df[df['datetime'] >= pd.to_datetime('2024-01-01')]
            future_dates = pd.date_range(df['datetime'].iloc[-1] + pd.Timedelta(days=1), periods=7).astype(str).tolist()

            return jsonify({
                'datetime': filtered_df['datetime'].astype(str).tolist(),
                'actual': actual.flatten().tolist(),
                'predicted': predicted.flatten().tolist(),
                'future_dates': future_dates,
                'future_predicted': future_predicted.flatten().tolist(),
                'accuracy': round(accuracy, 3),
                'rmse': round(rmse, 3),
                'actual_vs_predicted_plot': 'data:image/png;base64,' + actual_vs_predicted_plot,
                'future_predictions_plot': 'data:image/png;base64,' + future_predictions_plot
            })
        elif model_option == 'use_trained':
            # Load saved plots
            actual_vs_predicted_plot, future_predictions_plot = load_plots()

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

            model = load_model()

            predictions = model.predict(X_test)
            predictions_rescaled = target_scaler.inverse_transform(predictions)
            y_test_rescaled = target_scaler.inverse_transform(y_test)

            rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))

            accuracy = 100 - (np.mean(np.abs(predictions_rescaled - y_test_rescaled) / y_test_rescaled) * 100)

            # Predict next 7 days
            last_sequence = scaled_features[-sequence_length:]
            future_predictions = []
            for _ in range(7):
                prediction = model.predict(last_sequence[np.newaxis, :, :])
                future_predictions.append(prediction[0][0])
                last_sequence = np.append(last_sequence[1:], [[prediction[0][0], last_sequence[-1][1], last_sequence[-1][2], last_sequence[-1][3]]], axis=0)

            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions_rescaled = target_scaler.inverse_transform(future_predictions)

            return jsonify({
                'actual': y_test_rescaled.flatten().tolist(),
                'predicted': predictions_rescaled.flatten().tolist(),
                'future_predicted': future_predictions_rescaled.flatten().tolist(),
                'accuracy': round(accuracy, 3),
                'rmse': round(rmse, 3),
                'actual_vs_predicted_plot': 'data:image/png;base64,' + actual_vs_predicted_plot,
                'future_predictions_plot': 'data:image/png;base64,' + future_predictions_plot
            })
    return jsonify({'error': 'No data found'})

if __name__ == '__main__':
    app.run(debug=True)
