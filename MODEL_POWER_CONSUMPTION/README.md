# MODEL TO PREDICT POWER CONSUMPTION BASED ON DATE AND TIME

## 1st Model

The first model uses Adam as the optimizer and is using lr=0.001
Epoch [101/200], Train Loss: 0.7867, Val Loss: 0.7779


## 2nd Model

The second model uses SGD as the optimizer and is using lr=0.01
Epoch [100/200], Train Loss: 0.8045, Val Loss: 0.7995


## 3rd Model

The third model uses LSTM to predict output values.
Epoch [200/200], Train Loss: 0.7122


## 4th Model

The forth model uses XGBoost to predict the output values.
params = {
    'objective': 'reg:squaredlogerror',
    'learning_rate': 0.05,
    'max_depth': 4,
    'subsample': 0.6,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.8,
    'seed': 42
}
Train MAE: 0.1548 kW, Val MAE: 0.1801 kW, Test MAE: 0.1664 kW
Train RMSE: 0.2262 kW, Test RMSE: 0.2254 kW

**Input (6):**
    - Month;
    - Day;
    - Hour;
    - Weekday;
    - Sin(Hour);
    - Cos(Hour);
    - List of Previous Power Values (168 -> 7 days * 24 hours);