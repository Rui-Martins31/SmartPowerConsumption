import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os

HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
LOOKBACK_DAYS = 7
# LOOKBACK_HOURS = LOOKBACK_DAYS * HOURS_PER_DAY
LOOKBACK_HOURS = 24
MODEL_PARAMS = {
    'objective': 'reg:squarederror', 
    'eval_metric': 'rmse', 
    'learning_rate': 0.01, 
    'max_depth': 7, 
    'subsample': 0.8, 
    'colsample_bytree': 0.7, 
    'reg_alpha': 0.9, 
    'seed': 42
}

# Folder
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(MODEL_DIR, 'DATASET/dataset_power_visualizer.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_power_model_2.json')
SCALER_PATH = os.path.join(MODEL_DIR, 'power_consumption_scaler.pkl')


def clean_dataset(df: pd.DataFrame, required_columns: list[str] = ['Date', 'Global_active_power']):
    """
    Cleans the dataset by renaming columns and verifying/converting data types.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.rename(columns={'Date': 'ds', 'Global_active_power': 'y'})

    # Convert 'ds' to datetime
    try:
        df['ds'] = pd.to_datetime(df['ds'], errors='raise')
    except Exception as e:
        raise ValueError(f"Failed to convert 'ds' column to datetime: {str(e)}")

    # Convert 'y' to numeric (float)
    try:
        df['y'] = pd.to_numeric(df['y'], errors='raise')
    except Exception as e:
        raise ValueError(f"Failed to convert 'y' column to numeric: {str(e)}")

    # Verify data types
    if not pd.api.types.is_datetime64_any_dtype(df['ds']):
        raise ValueError("'ds' column is not in datetime format")
    if not pd.api.types.is_numeric_dtype(df['y']):
        raise ValueError("'y' column is not numeric")

    return df

def create_df_from_date_to_predict(date_to_predict: datetime.datetime, prev_values: list[float]) -> pd.DataFrame:
    """
    Creates a DataFrame for prediction starting from a specified date using previous power consumption values.
    """
    # Validate inputs
    if not isinstance(date_to_predict, datetime.datetime):
        raise ValueError("date_to_predict must be a datetime object")
    if len(prev_values) < LOOKBACK_HOURS:
        raise ValueError(f"prev_values must contain at least {LOOKBACK_HOURS} values for lookback period")
    
    try:
        prev_values = [float(x) for x in prev_values]
    except (ValueError, TypeError):
        raise ValueError("prev_values must contain numeric values")
    
    # Create timestamps for lookback period
    start_date = date_to_predict - pd.Timedelta(hours=LOOKBACK_HOURS)
    timestamps = pd.date_range(start=start_date, end=date_to_predict - pd.Timedelta(hours=1), freq='h')
    
    # Create DataFrame
    df = pd.DataFrame({
        'ds': timestamps,
        'y': prev_values[-LOOKBACK_HOURS:]
    })
    
    # Verify data types
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='raise')
    
    return df

def create_features(df: pd.DataFrame, n_lags=24):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df['ds'].dt.hour
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['quarter'] = df['ds'].dt.quarter
    df['month'] = df['ds'].dt.month
    # df['year'] = df['ds'].dt.year
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['dayofmonth'] = df['ds'].dt.day
    df['weekofyear'] = df['ds'].dt.isocalendar().week.astype(int)

    # Add multiple lagged features using a loop
    for i in range(1, n_lags + 1):
        df[f'y_lag{i}'] = df['y'].shift(i)

    # Add rolling window features (example: rolling mean of window size 3)
    df['rolling_mean_3'] = df['y'].rolling(window=3).mean()

    return df

def get_feature_names(df: pd.DataFrame) -> list[str]:
    """
    Returns a list of feature names used by the model.
    """
    features = [col for col in df.columns if col not in ['ds', 'y']]
    return features

def train_model(df: pd.DataFrame, debug: bool = False, save_model: bool = False):
    """
    Trains the XGBoost model.
    """
    features = get_feature_names(df)
    X = df[features]
    y = df['y']
    ds = df['ds']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    ds_test = ds.iloc[-len(y_test):]

    # Train
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=1000, 
        learning_rate=0.05, 
        random_state=42, 
        eval_metric="rmse",
        n_jobs=-1
    )
    xgb_model.fit(
        X_train,           
        y_train, 
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50
    )

    if debug:
        # Predict
        y_pred = xgb_model.predict(X_test)

        # Evaluate (RMSE)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"XGBoost RMSE: {rmse}")

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(ds_test, y_test, label='Actual (y_test)', marker='x')
        plt.plot(ds_test, y_pred, label='Predicted (y_pred)', marker='o')
        plt.title('Power Consumption: Actual vs Predicted (Test Set)')
        plt.xlabel('Time')
        plt.ylabel('Power Consumption')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Feature importance
        xgb.plot_importance(xgb_model)
        plt.show()

    if save_model:
        xgb_model.save_model(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    return xgb_model

def predict_future(df: pd.DataFrame, 
                   model_path: str = MODEL_PATH, 
                   forecast_hours: int = 24):
    """
    Predicts power consumption for the next specified hours and compares with actual values if available.
    """
    # Model
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    # Get dataframe
    df_copy: pd.DataFrame = df.tail(LOOKBACK_HOURS+1).copy()
    df_copy: pd.DataFrame = create_features(df=df_copy, n_lags=LOOKBACK_HOURS)
    list_features: list[str] = get_feature_names(df=df_copy)
    # df: pd.DataFrame = df[list_features]
    # print(f"{df_copy = }")

    predictions: list[float] = []

    # Loop
    for i in range(forecast_hours):
        X_pred: pd.DataFrame = df_copy[list_features].tail(1)

        # Predict
        pred = float(model.predict(X_pred)[0])
        predictions.append(pred)  

        # Update dataframe
        df_copy.loc[df_copy.index[-1], 'y'] = pred

        # Add new row
        new_time = df_copy["ds"].iloc[-1] + pd.Timedelta(hours=1)
        new_row = {col: 0 for col in df_copy.columns if col != "ds"}
        new_row["ds"] = new_time

        df_copy: pd.DataFrame = pd.concat([df_copy, pd.DataFrame([new_row])], ignore_index=True)

        # Create features
        df_copy: pd.DataFrame = create_features(df=df_copy, n_lags=LOOKBACK_HOURS)


    # print(f"{pred = }")
    print(f"{df_copy = }")   

    return predictions


#-------------------------
if __name__ == '__main__':
    print("-> Train model:")
    df = pd.read_csv(DATASET_PATH)
    df = clean_dataset(df)
    # print(df.head())

    df_features = create_features(df=df, n_lags=24)
    # print(df_features)

    df_features = df_features.dropna()
    model = train_model(df=df_features, debug=True, save_model=True)
    
    print("\n-> Predicting next 24 hours:")
    predictions = predict_future(df=df, forecast_hours=24)
    print(f"{predictions = }")



# Deprecated -------------
def predict_next_24_hours(start_dt: datetime.datetime, recent_hourly_consumption_kw: list[float]) -> list[float]:
    """
    Predicts power consumption for the next 24 hours using a recursive strategy.
    start_dt: The datetime for the beginning of the 24-hour forecast period.
    recent_hourly_consumption_kw: A list of the last LOOKBACK_HOURS (168) known power consumption values in kW.
    """
    # print("The method 'predic_next_24_hours' will be updated in the future.")
    # Load Model
    try:
        model = xgb.Booster()
        model.load_model(os.path.join(MODEL_DIR, 'xgboost_power_model.json'))
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        print("Please ensure train_model() has been run successfully.")
        return []

    if len(recent_hourly_consumption_kw) != LOOKBACK_DAYS * HOURS_PER_DAY:
        raise ValueError(f"Input `recent_hourly_consumption_kw` must contain exactly {LOOKBACK_DAYS * HOURS_PER_DAY} values.")

    predictions_24_hours = []
    current_lags = list(recent_hourly_consumption_kw)

    for hour_ahead in range(24):
        prediction_dt = start_dt + datetime.timedelta(hours=hour_ahead)

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
        time_feature_names = ['month', 'day', 'hour', 'day_of_week', 'hour_sin', 'hour_cos']
        lag_feature_names = [f'lag_{i+1}h' for i in range(LOOKBACK_DAYS * HOURS_PER_DAY)]
        feature_names = time_feature_names + lag_feature_names
        
        # Predict
        dpredict = xgb.DMatrix([feature_vector], feature_names=feature_names)
        new_prediction = model.predict(dpredict)[0]
        new_prediction = max(0.0, float(new_prediction))

        # Store result
        predictions_24_hours.append(new_prediction)

        current_lags.pop(0)
        current_lags.append(new_prediction)

    return predictions_24_hours


def get_pow_con_from_db(day: datetime.datetime) -> list[float]:
    """
    Temporary method that get power consumption values from the dataset
    """
    df: pd.DataFrame = pd.read_csv(
        DATASET_PATH,
        parse_dates=["Date"],
        dtype={"Global_active_power": "float"}
    )

    # print(f"{df = }")

    try:
        target_day = day.replace(year=2007) # Replace the year to be compatible with dataset's timeline
        daily_data = df[df["Date"].dt.date == target_day.date()]
    except:
        target_day = day.replace(year=2007, day=day.day-1)
        daily_data = df[df["Date"].dt.date == target_day.date()]

    # print(f"{target_day = }")
    # print(f"{daily_data = }")

    return daily_data["Global_active_power"].tolist()
    