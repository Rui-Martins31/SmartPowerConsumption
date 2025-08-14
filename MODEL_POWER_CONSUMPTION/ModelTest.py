"""
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
#from ModelTrain import PowerConsumptionModel 

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PowerConsumptionModel(nn.Module):
    def __init__(self, input_size):
        super(PowerConsumptionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

# Load the saved model
model = PowerConsumptionModel(input_size=4).to(device)
model.load_state_dict(torch.load('model_power_consumption.pth'))
model.eval()

# Recreate MinMaxScaler parameters for input normalization
# Based on preprocessing: Month [1,12], Day [1,31], Hour [0,23], Minute [0,59]
scaler = MinMaxScaler()
scaler.data_min_ = np.array([1, 1, 0, 0])  # Min values for Month, Day, Hour, Minute
scaler.data_max_ = np.array([12, 31, 23, 59])  # Max values
scaler.min_ = -scaler.data_min_ / (scaler.data_max_ - scaler.data_min_)
scaler.scale_ = 1.0 / (scaler.data_max_ - scaler.data_min_)

# Example new inputs
new_inputs = np.array([
    [6, 15, 12, 30],  # June 15, 12:30
    [12, 1, 0, 0],    # December 1, 00:00
    [3, 25, 18, 45],  # March 25, 18:45
    [12, 16, 17, 24],
    [12, 16, 17, 25],
    [12, 16, 17, 26]
])

# Normalize new inputs
new_inputs_normalized = scaler.transform(new_inputs)

# Convert to tensor and predict
new_inputs_tensor = torch.FloatTensor(new_inputs_normalized).to(device)
with torch.no_grad():
    new_predictions = model(new_inputs_tensor)
    print("\nPredictions for new inputs (Global_active_power):")
    for i, (input_vals, pred) in enumerate(zip(new_inputs, new_predictions.cpu().numpy())):
        print(f"Input: Month={input_vals[0]}, Day={input_vals[1]}, Hour={input_vals[2]}, Minute={input_vals[3]} -> Predicted Power: {pred[0]:.4f}")
"""




import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.model_selection import train_test_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the LSTM model (same as training)
class PowerConsumptionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(PowerConsumptionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]  # Take last timestep
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Load the saved model
model = PowerConsumptionLSTM(input_size=1, hidden_size=64, num_layers=1).to(device)
model.load_state_dict(torch.load('lstm_time_model.pth'))
model.eval()

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Example inputs: [Month, Day, Hour, Minute]
new_inputs = np.array([
    [6, 15, 12, 30],  # June 15, 12:30
    [12, 1, 0, 0],    # December 1, 00:00
    [3, 25, 18, 45],  # March 25, 18:45
    [8, 14, 9, 28]    # August 14, 09:28 (today, 09:28 AM WEST, 2025)
])

# Apply cyclical encoding to Hour and Minute
hour = new_inputs[:, 2]
minute = new_inputs[:, 3]
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
minute_sin = np.sin(2 * np.pi * minute / 60)
minute_cos = np.cos(2 * np.pi * minute / 60)
new_inputs_cyclical = np.column_stack([new_inputs[:, :2], hour_sin, hour_cos, minute_sin, minute_cos])

# Normalize inputs (update scaler ranges for cyclical features)
scaler_cyclical = pickle.load(open('scaler.pkl', 'rb'))
scaler_cyclical.data_min_ = np.array([1, 1, -1, -1, -1, -1])  # Month, Day, sin, cos
scaler_cyclical.data_max_ = np.array([12, 31, 1, 1, 1, 1])
scaler_cyclical.min_ = -scaler_cyclical.data_min_ / (scaler_cyclical.data_max_ - scaler_cyclical.data_min_)
scaler_cyclical.scale_ = 1.0 / (scaler_cyclical.data_max_ - scaler_cyclical.data_min_)
new_inputs_normalized = scaler_cyclical.transform(new_inputs_cyclical)
new_inputs_tensor = torch.FloatTensor(new_inputs_normalized).reshape(-1, 6, 1).to(device)

# Predict
with torch.no_grad():
    predictions = model(new_inputs_tensor)
    print("\nPredictions for new inputs with cyclical encoding (Global_active_power):")
    for i, (input_vals, pred) in enumerate(zip(new_inputs, predictions.cpu().numpy())):
        print(f"Input: Month={input_vals[0]}, Day={input_vals[1]}, Hour={input_vals[2]}, Minute={input_vals[3]} -> Predicted Power: {pred[0]:.4f}")

# Optional: Test on the test set (requires X.npy and Y.npy)
try:
    X = np.load('DATASET/X.npy')
    Y = np.load('DATASET/Y.npy')

    # Recreate test set (same split as training: 70% train, 20% val, 10% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3333, random_state=42)

    # Convert to tensors and reshape for LSTM
    X_test = torch.FloatTensor(X_test).reshape(-1, 4, 1).to(device)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1).to(device)

    # Evaluate on test set in batches
    batch_size = 128
    test_batch_indices = range(0, len(X_test), batch_size)
    test_loss_total = 0
    test_mse_total = 0
    with torch.no_grad():
        for i in test_batch_indices:
            batch_X_test = X_test[i:i+batch_size]
            batch_y_test = y_test[i:i+batch_size]
            test_outputs = model(batch_X_test)
            test_loss = torch.nn.HuberLoss()(test_outputs, batch_y_test)
            test_mse = ((test_outputs - batch_y_test) ** 2).mean().item()
            test_loss_total += test_loss.item()
            test_mse_total += test_mse
    
    avg_test_loss = test_loss_total / len(test_batch_indices)
    avg_test_mse = test_mse_total / len(test_batch_indices)
    print(f'\nTest Loss (Huber): {avg_test_loss:.4f}')
    print(f'Test Loss (MSE): {avg_test_mse:.4f}')
except FileNotFoundError:
    print("\nTest set evaluation skipped: DATASET/X.npy or DATASET/Y.npy not found")
