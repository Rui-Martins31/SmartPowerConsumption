import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
import os

# Set environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
days_lookback = 14
minutes_per_day = 1440
input_size = 5  # 4 time features + 1 power consumption
sequence_length = days_lookback * minutes_per_day  # 10,080
output_size = minutes_per_day  # 1440
hidden_size = 64
num_layers = 1
batch_size = 16

# Load data
X = np.load('DATASET/X.npy')  # Shape: (n_samples, 4)
Y = np.load('DATASET/Y.npy')  # Shape: (n_samples,)

# Validate data
if len(X) != len(Y):
    raise ValueError(f"X and Y lengths mismatch: {len(X)} vs {len(Y)}")
if len(Y) < (days_lookback + 1) * minutes_per_day:
    raise ValueError(f"Dataset too small: need at least {(days_lookback + 1) * minutes_per_day} measurements")

# Create input-output pairs with daily sliding window (step size = 1440)
X_lagged = []
Y_target = []
for i in range(0, len(Y) - (days_lookback + 1) * minutes_per_day + 1, minutes_per_day):
    sequence_X = X[i:i + days_lookback * minutes_per_day]
    sequence_Y = Y[i:i + days_lookback * minutes_per_day].reshape(-1, 1)  # Raw kW values
    sequence = np.hstack((sequence_X, sequence_Y))
    X_lagged.append(sequence)
    Y_target.append(Y[i + days_lookback * minutes_per_day:i + (days_lookback + 1) * minutes_per_day])  # Raw kW values

X_lagged = np.array(X_lagged)  # Shape: (samples, 10080, 5)
Y_target = np.array(Y_target)  # Shape: (samples, 1440)

# Chronological split
n = len(Y_target)
train_end = int(0.7 * n)
val_end = int(0.9 * n)

X_train = X_lagged[:train_end]
y_train = Y_target[:train_end]
X_val = X_lagged[train_end:val_end]
y_val = Y_target[train_end:val_end]
X_test = X_lagged[val_end:]
y_test = Y_target[val_end:]

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

# Convert to tensors and move to device
X_train = torch.FloatTensor(X_train).to(device)
X_val = torch.FloatTensor(X_val).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.FloatTensor(y_train).to(device)
y_val = torch.FloatTensor(y_val).to(device)
y_test = torch.FloatTensor(y_test).to(device)

# Define the model (LSTM-based)
class PowerConsumptionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PowerConsumptionModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)  # No ReLU to allow any kW value
        # self.relu = nn.ReLU()  # Commented out if negatives are possible
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Initialize model, loss, and optimizer
model = PowerConsumptionModel(input_size=input_size, hidden_size=hidden_size, 
                             num_layers=num_layers, output_size=output_size).to(device)
criterion = nn.L1Loss()  # Use MAE for better kW interpretation
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for raw values

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Training loop with early stopping
num_epochs = 1000
batch_indices = range(0, len(X_train), batch_size)
best_val_loss = float('inf')
patience = 40
patience_counter = 0
best_model_state = None

for epoch in tqdm(range(num_epochs), desc="Epochs"):
    model.train()
    batch_bar = tqdm(batch_indices, desc="Batches", leave=False)
    total_loss = 0
    for i in batch_bar:
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    # Average training loss
    avg_train_loss = total_loss / len(batch_indices)
    
    # Evaluate on validation set in smaller chunks
    model.eval()
    val_loss = 0
    val_batches = range(0, X_val.shape[0], batch_size)
    with torch.no_grad():
        for i in val_batches:
            batch_X_val = X_val[i:i+batch_size]
            batch_y_val = y_val[i:i+batch_size]
            val_outputs = model(batch_X_val)
            val_loss += criterion(val_outputs, batch_y_val).item() * len(batch_X_val)
    val_loss /= len(X_val)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f} kW, Val Loss: {val_loss:.4f} kW')
    
    # Step the scheduler
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Load best model
model.load_state_dict(best_model_state)

# Evaluate on test set in smaller chunks
model.eval()
test_loss = 0
test_batches = range(0, len(X_test), batch_size)
test_outputs_np = []
with torch.no_grad():
    for i in test_batches:
        batch_X_test = X_test[i:i+batch_size]
        batch_y_test = y_test[i:i+batch_size]
        batch_outputs = model(batch_X_test)
        test_loss += criterion(batch_outputs, batch_y_test).item() * len(batch_X_test)
        test_outputs_np.append(batch_outputs.cpu().numpy())
test_loss /= len(X_test)

# Concatenate test outputs
test_outputs_np = np.concatenate(test_outputs_np, axis=0)
test_predictions = test_outputs_np  # No inverse transform needed
y_test_np = y_test.cpu().numpy()
y_test_original = y_test_np  # No inverse transform needed

print(f'\nTest Loss: {test_loss:.4f} kW')

# Save the best model
torch.save(best_model_state, 'model_power_previous_days.pth')

# Visualize predictions for a few test days
import matplotlib.pyplot as plt
for i in range(min(3, len(test_predictions))):
    plt.figure(figsize=(12, 4))
    plt.plot(test_predictions[i], label='Predicted', alpha=0.7)
    plt.plot(y_test_original[i], label='Actual', alpha=0.7)
    plt.title(f'Test Day {i+1} Power Consumption')
    plt.xlabel('Minute of Day')
    plt.ylabel('Power Consumption (kW)')
    plt.legend()
    plt.show()

# Additional metrics
mae = np.mean(np.abs(test_predictions - y_test_original))
rmse = np.sqrt(np.mean((test_predictions - y_test_original) ** 2))
print(f'Test MAE: {mae:.4f} kW, Test RMSE: {rmse:.4f} kW')