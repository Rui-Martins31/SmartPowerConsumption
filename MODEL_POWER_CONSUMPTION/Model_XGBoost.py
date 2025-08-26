import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import joblib

DAYS_LOOKBACK = 7
HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
LOOKBACK_HOURS = DAYS_LOOKBACK * HOURS_PER_DAY
MODEL_PARAMS = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'learning_rate': 0.02,
                'max_depth': 8,
                'subsample': 0.4,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.8,
                'seed': 42
            }


def create_features(X_hourly, Y_hourly, Y_scaled, lookback_hours=LOOKBACK_HOURS):
    """
    Creates feature set for the model.
    X_hourly: Averaged hourly time features.
    Y_scaled: Scaled hourly power consumption values.
    lookback_hours: How many past hours of consumption to use as features.
    """
    features = []
    targets = []
    for i in range(lookback_hours, len(Y_scaled)):
        # Reconstruct date/time features from normalized hourly averages
        month_norm, day_norm, hour_norm, _ = X_hourly[i]
        
        month = int(np.round(month_norm * 11 + 1))
        day = int(np.round(day_norm * 30 + 1))
        hour = int(np.round(hour_norm * 23))

        month = np.clip(month, 1, 12)
        day = np.clip(day, 1, 31)
        hour = np.clip(hour, 0, 23)

        try:
            date = datetime(2024, month, day, hour) # Check if date and time are valid
        except ValueError:
            continue

        day_of_week = date.weekday() / 6.0
        
        # Cyclical features for the hour of the day
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        
        # Lagged features (power consumption from previous hours)
        lagged_y = Y_scaled[i-lookback_hours:i].tolist()
        
        # Combine all features
        current_features = [
            month / 12.0, 
            day / 31.0, 
            hour / 24.0, 
            day_of_week, 
            hour_sin, 
            hour_cos
        ] + lagged_y
        
        features.append(current_features)
        targets.append(Y_hourly[i])
        
    return np.array(features), np.array(targets)

def load_and_preprocess_data():
    """
    Loads and preprocesses the data, returning hourly aggregated features and targets.
    
    Returns:
        tuple: (X_hourly, Y_hourly, Y_scaled, scaler)
    """
    try:
        X = np.load('DATASET/X.npy')  # Shape: (n_samples, 4), normalized [0, 1]
        Y = np.load('DATASET/Y.npy')  # Shape: (n_samples,), raw kW values
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data files not found in DATASET directory: {e}")

    # Aggregate to hourly resolution
    total_minutes = len(Y)
    n_full_hours = total_minutes // MINUTES_PER_HOUR
    
    if total_minutes % MINUTES_PER_HOUR != 0:
        print(f"Warning: Dataset has {total_minutes % MINUTES_PER_HOUR} extra minutes. Trimming to {n_full_hours} hours.")
        Y = Y[:n_full_hours * MINUTES_PER_HOUR]
        X = X[:n_full_hours * MINUTES_PER_HOUR]

        # Reshape and average over each 60-minute block
    Y_hourly = Y.reshape(n_full_hours, MINUTES_PER_HOUR).mean(axis=1)
    X_hourly = X.reshape(n_full_hours, MINUTES_PER_HOUR, 4).mean(axis=1)

    # Normalize lagged values using MinMaxScaler
    scaler = MinMaxScaler()
    Y_scaled = scaler.fit_transform(Y_hourly.reshape(-1, 1)).flatten()

    # Validate data
    if len(X_hourly) != len(Y_hourly):
        raise ValueError(f"X_hourly and Y_hourly lengths mismatch: {len(X_hourly)} vs {len(Y_hourly)}")
    if len(Y_hourly) < LOOKBACK_HOURS + 1:
        raise ValueError(f"Dataset too small: need at least {LOOKBACK_HOURS + 1} hours of data.")
    
    return X_hourly, Y_hourly, Y_scaled, scaler

def evaluate_model(model, X_test, y_test, y_pred):
    """
    Evaluates the model performance and creates visualizations.
    """
    # Calculate metrics
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f'\nTest MAE: {test_mae:.4f} kW')
    print(f'Test RMSE: {test_rmse:.4f} kW')
    
    """
    # Plot predictions
    n_days_to_plot = 3
    for i in range(min(n_days_to_plot, len(y_test) // HOURS_PER_DAY)):
        start_idx = i * HOURS_PER_DAY
        end_idx = (i + 1) * HOURS_PER_DAY
        plt.figure(figsize=(15, 5))
        plt.plot(y_test[start_idx:end_idx], label='Actual', marker='.')
        plt.plot(y_pred[start_idx:end_idx], label='Predicted', marker='.')
        plt.title(f'Test Day {i+1} Power Consumption (Hourly)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Power Consumption (kW)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()

    # Feature importance
    xgb.plot_importance(model, max_num_features=20)
    plt.title("Feature Importance")
    plt.show()
    """
    
    return test_mae, test_rmse

def train_model():
    """
    Loads data, preprocesses it, trains the XGBoost model, and saves it.
    """
    print("Loading and preprocessing data...")
    X_hourly, Y_hourly, Y_scaled, scaler = load_and_preprocess_data()
    
    # Save the scaler for later use in prediction
    joblib.dump(scaler, 'power_consumption_scaler.pkl')
    print("Scaler saved to 'power_consumption_scaler.pkl'")

    # Create features and split data
    print("Creating features...")
    X_features, y = create_features(X_hourly, Y_hourly, Y_scaled, lookback_hours=LOOKBACK_HOURS)
    
    # Create feature names
    feature_names = get_feature_names()

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_features, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, shuffle=False)
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

    try:
        # Prepare data for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

        # Train XGBoost
        print("Training XGBoost model...")
        params = MODEL_PARAMS
        evals = [(dtrain, 'train'), (dval, 'eval')]
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=1000,
            evals=evals, 
            early_stopping_rounds=200,
            verbose_eval=50
        )

        # Predict and evaluate
        y_pred_test = model.predict(dtest)
        test_mae, test_rmse = evaluate_model(model, X_test, y_test, y_pred_test)

        # Save model
        model.save_model('xgboost_power_model.json')
        print("Model saved to 'xgboost_power_model.json'")
        
        return model, test_mae, test_rmse
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return None, None, None


def get_feature_names() -> list[str]:
    """
    Returns a list of feature names used by the model.
    """
    time_feature_names = ['month', 'day', 'hour', 'day_of_week', 'hour_sin', 'hour_cos']
    lag_feature_names = [f'lag_{i+1}h' for i in range(LOOKBACK_HOURS)]
    return time_feature_names + lag_feature_names

def predict_next_hour(prediction_dt: datetime, recent_hourly_consumption_kw: list[float]) -> float:
    """
    Predicts power consumption for the next single hour given a datetime and recent consumption data.
    prediction_dt: The datetime object for the hour you want to predict.
    recent_hourly_consumption_kw: A list of the last LOOKBACK_HOURS (168) power consumption values in kW.
    """
    # Load Model
    try:
        model = xgb.Booster()
        model.load_model('xgboost_power_model.json')
        scaler = joblib.load('power_consumption_scaler.pkl')
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        print("Please run train_model() first.")
        return -1.0

    if len(recent_hourly_consumption_kw) != LOOKBACK_HOURS:
        raise ValueError(f"Input `recent_hourly_consumption_kw` must contain exactly {LOOKBACK_HOURS} values.")

    # Create features
    month = prediction_dt.month / 12.0
    day = prediction_dt.day / 31.0
    hour = prediction_dt.hour / 24.0
    day_of_week = prediction_dt.weekday() / 6.0
    hour_sin = np.sin(2 * np.pi * prediction_dt.hour / 24.0)
    hour_cos = np.cos(2 * np.pi * prediction_dt.hour / 24.0)
    
    scaled_lags = scaler.transform(np.array(recent_hourly_consumption_kw).reshape(-1, 1))
    scaled_lags = scaled_lags.flatten().tolist()

    feature_vector = [month, day, hour, day_of_week, hour_sin, hour_cos] + scaled_lags
    feature_names = get_feature_names()

    # Predict
    dpredict = xgb.DMatrix([feature_vector], feature_names=feature_names)
    prediction = model.predict(dpredict)

    return float(prediction[0])

def predict_next_24_hours(start_dt: datetime, recent_hourly_consumption_kw: list[float]) -> list[float]:
    """
    Predicts power consumption for the next 24 hours using a recursive strategy.
    start_dt: The datetime for the beginning of the 24-hour forecast period.
    recent_hourly_consumption_kw: A list of the last LOOKBACK_HOURS (168) known power consumption values in kW.
    """
    # Load Model
    try:
        model = xgb.Booster()
        model.load_model('xgboost_power_model.json')
        scaler = joblib.load('power_consumption_scaler.pkl')
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        print("Please ensure train_model() has been run successfully.")
        return []

    if len(recent_hourly_consumption_kw) != LOOKBACK_HOURS:
        raise ValueError(f"Input `recent_hourly_consumption_kw` must contain exactly {LOOKBACK_HOURS} values.")

    predictions_24_hours = []
    current_lags = list(recent_hourly_consumption_kw)

    for hour_ahead in range(24):
        prediction_dt = start_dt + timedelta(hours=hour_ahead)

        # Create features
        month = prediction_dt.month / 12.0
        day = prediction_dt.day / 31.0
        hour = prediction_dt.hour / 24.0
        day_of_week = prediction_dt.weekday() / 6.0
        hour_sin = np.sin(2 * np.pi * prediction_dt.hour / 24.0)
        hour_cos = np.cos(2 * np.pi * prediction_dt.hour / 24.0)

        # Scale the current set of lagged values
        scaled_lags = scaler.transform(np.array(current_lags).reshape(-1, 1))
        scaled_lags = scaled_lags.flatten().tolist()

        # Combine all features
        feature_vector = [month, day, hour, day_of_week, hour_sin, hour_cos] + scaled_lags
        feature_names = get_feature_names()
        
        # Predict
        dpredict = xgb.DMatrix([feature_vector], feature_names=feature_names)
        new_prediction = model.predict(dpredict)[0]
        new_prediction = max(0.0, float(new_prediction))

        # Store result
        predictions_24_hours.append(new_prediction)

        current_lags.pop(0)
        current_lags.append(new_prediction)

    return predictions_24_hours



"""
'learning_rate': 0.02,
'max_depth': 8,
'subsample': 0.4,
'colsample_bytree': 0.9,
'reg_alpha': 0.8,
'seed': 42
"""

if __name__ == '__main__':
    print("Train model:")
    best_test_mae: float = float("inf")
    params_to_use: list[float] = [ 0 for _ in range(6)]
    for lr in np.arange(0.01, 0.3, 0.05):
        for max_d in range(3, 8):
            for subsmp in np.arange(0.5, 1.0, 0.1):
                for colsmp in np.arange(0.5, 1.0, 0.1):
                    for reg_al in np.arange(0.1, 1.0, 0.2):
                        for seed in range(42, 43):
                            
                            MODEL_PARAMS['learning_rate'] = lr
                            MODEL_PARAMS['max_depth'] = max_d
                            MODEL_PARAMS['subsample'] = round(subsmp, 2)
                            MODEL_PARAMS['colsample_bytree'] = round(colsmp, 2)
                            MODEL_PARAMS['reg_alpha'] = round(reg_al, 2)
                            MODEL_PARAMS['seed'] = seed
                            
                            # print(f"Testing params: {MODEL_PARAMS}") # Helpful to see progress
                            _, test_mae, _ = train_model()

                            if test_mae is not None and test_mae < best_test_mae:
                                best_test_mae = test_mae
                                best_params = MODEL_PARAMS.copy()


    print(f"Best parameters found: {best_params}")
    print(f"Best test MAE: {best_test_mae}")

    """
    print("\nTesting 24-Hour Prediction Function:")
    start_prediction_dt = datetime(2025, 8, 23, 13, 0, 0)

    dataset_values = np.load('DATASET/Y.npy')[-LOOKBACK_HOURS:]
    previous_values_list: list[float] = [ float(value) for value in dataset_values ]
    print(f"{previous_values_list = },\n {type(previous_values_list)}")

    try:
        forecast_values = predict_next_24_hours(start_prediction_dt, previous_values_list)

        if forecast_values:
            print(f"\nSuccessfully generated 24-hour forecast starting from {start_prediction_dt}:")
            for i, val in enumerate(forecast_values):
                timestamp = start_prediction_dt + timedelta(hours=i)
                print(f"  - {timestamp.strftime('%Y-%m-%d %H:%M')}: {val:.4f} kW")

            # Visualize the forecast
            plt.figure(figsize=(15, 6))
            forecast_hours = [start_prediction_dt + timedelta(hours=i) for i in range(24)]
            plt.plot(forecast_hours, forecast_values, marker='o', linestyle='-', label='24-Hour Forecast')
            
            plt.title(f'Power Consumption Forecast for the Next 24 Hours\n(Starting {start_prediction_dt.strftime("%Y-%m-%d %H:%M")})')
            plt.xlabel('Date and Time')
            plt.ylabel('Predicted Power Consumption (kW)')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")
    
    """