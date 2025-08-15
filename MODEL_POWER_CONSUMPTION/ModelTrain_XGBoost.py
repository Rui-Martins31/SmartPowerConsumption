import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Load data
X = np.load('DATASET/X.npy')  # Shape: (n_samples, 4), normalized [0, 1]
Y = np.load('DATASET/Y.npy')  # Shape: (n_samples,), raw kW values

# Aggregate to hourly data
minutes_per_day = 1440
hours_per_day = 24
days_lookback = 7
total_minutes = len(Y)
n_full_days = total_minutes // minutes_per_day
if total_minutes % minutes_per_day != 0:
    print(f"Warning: Dataset has {total_minutes % minutes_per_day} extra minutes. Trimming to {n_full_days} days.")
    Y = Y[:n_full_days * minutes_per_day]
    X = X[:n_full_days * minutes_per_day]

n_samples = len(Y) // minutes_per_day
Y_hourly = Y.reshape(n_samples, minutes_per_day).mean(axis=1)  # Average kW per hour
X_hourly = X.reshape(n_samples, minutes_per_day, 4).mean(axis=1)  # Average time features per hour

# Normalize lagged values
scaler = MinMaxScaler()
Y_scaled = scaler.fit_transform(Y_hourly.reshape(-1, 1)).flatten()

# Validate data
if len(X_hourly) != len(Y_hourly):
    raise ValueError(f"X_hourly and Y_hourly lengths mismatch: {len(X_hourly)} vs {len(Y_hourly)}")
if len(Y_hourly) < (days_lookback + 1) * hours_per_day:
    raise ValueError(f"Dataset too small: need at least {(days_lookback + 1) * hours_per_day} hours")

# Feature engineering
def create_features(X, Y_scaled, lookback=days_lookback * hours_per_day):
    features = []
    targets = []
    for i in range(lookback, len(Y_scaled)):
        month = np.round(X[i, 0] * 11 + 1).astype(int)
        day = np.round(X[i, 1] * 30 + 1).astype(int)
        hour = np.round(X[i, 2] * 23).astype(int)
        minute = np.round(X[i, 3] * 59).astype(int)
        
        month = np.clip(month, 1, 12)
        day = np.clip(day, 1, 31)
        hour = np.clip(hour, 0, 23)
        
        date = datetime(2024, month, day, hour)
        day_of_week = date.weekday() / 6.0
        
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        
        lagged_y = Y_scaled[i-lookback:i].tolist()
        
        features.append([month / 12.0, day / 31.0, hour / 24.0, day_of_week, hour_sin, hour_cos] + lagged_y)
        targets.append(Y_hourly[i])
    return np.array(features), np.array(targets)

X_features, y = create_features(X_hourly, Y_scaled)
X_train, X_temp, y_train, y_temp = train_test_split(X_features, y, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, shuffle=False)

# Prepare data for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train XGBoost with native API and early stopping
params = {
    'objective': 'reg:squaredlogerror',
    'learning_rate': 0.05,
    'max_depth': 4,
    'subsample': 0.6,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.8,
    'seed': 42
}
evals = [(dtrain, 'train'), (dval, 'eval')]
model = xgb.train(params, dtrain, num_boost_round=200, evals=evals, early_stopping_rounds=10, verbose_eval=False)

# Predict
y_pred_train = model.predict(dtrain)
y_pred_val = model.predict(dval)
y_pred_test = model.predict(dtest)

# Evaluate
train_mae = mean_absolute_error(y_train, y_pred_train)
val_mae = mean_absolute_error(y_val, y_pred_val)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f'Train MAE: {train_mae:.4f} kW, Val MAE: {val_mae:.4f} kW, Test MAE: {test_mae:.4f} kW')
print(f'Train RMSE: {train_rmse:.4f} kW, Test RMSE: {test_rmse:.4f} kW')

# Save model
model.save_model('xgboost_power_model.json')

# Visualize predictions for a few test days
import matplotlib.pyplot as plt
n_days = 3
hours_per_test_day = len(y_test) // n_days if len(y_test) >= n_days else len(y_test)
for i in range(min(n_days, len(y_test) // hours_per_day)):
    start_idx = i * hours_per_day
    end_idx = (i + 1) * hours_per_day
    plt.figure(figsize=(12, 4))
    plt.plot(y_pred_test[start_idx:end_idx], label='Predicted', alpha=0.7)
    plt.plot(y_test[start_idx:end_idx], label='Actual', alpha=0.7)
    plt.title(f'Test Day {i+1} Power Consumption (Hourly)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Power Consumption (kW)')
    plt.legend()
    plt.show()

# Feature importance
xgb.plot_importance(model)
plt.show()