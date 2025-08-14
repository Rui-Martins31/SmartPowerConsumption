"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
X = np.load('DATASET/X.npy')
X = X[:, [2, 3]]    # Hour and Minute
Y = np.load('DATASET/Y.npy')

# Check data ranges before processing
print("X - Min:", X.min(axis=0), 
      "Max:", X.max(axis=0), 
      "Mean:", X.mean(axis=0), 
      "Std:", X.std(axis=0))
print("Y - Min:", Y.min(), 
      "Max:", Y.max(), 
      "Mean:", Y.mean(), 
      "Std:", Y.std())

# Convert to PyTorch tensors and move to device
X = torch.FloatTensor(X).to(device)
Y = torch.FloatTensor(Y).reshape(-1, 1).to(device)

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X.cpu().numpy(), Y.cpu().numpy(), test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3333, random_state=42)  # 0.3 * 0.3333 ≈ 0.1 test

# Convert to tensors and move to device
X_train = torch.FloatTensor(X_train).to(device)
X_val = torch.FloatTensor(X_val).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
y_val = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
y_test = torch.FloatTensor(y_test).reshape(-1, 1).to(device)

# Define the model with increased capacity
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

# Initialize model, move to device, loss, and optimizer
#model = PowerConsumptionModel(input_size=4).to(device)
model = PowerConsumptionModel(input_size=2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
#optimizer = optim.SGD(model.parameters(), lr=0.01)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Training loop with early stopping
num_epochs = 200
batch_size = 128
batch_indices = range(0, len(X_train), batch_size)
best_val_loss = float('inf')
patience = 20
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
    
    # Average training loss for the epoch
    avg_train_loss = total_loss / len(batch_indices)
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}')
    
    # Step the scheduler based on validation loss
    scheduler.step(val_loss)
    
    # Early stopping based on validation loss
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

# Evaluate the model on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'\nTest Loss: {test_loss.item():.4f}')

# Save the best model
torch.save(best_model_state, 'model_power_consumption.pth')
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import copy
import os
import pickle

# Set CUDA memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
X = np.load('DATASET/X.npy')
Y = np.load('DATASET/Y.npy')

# Check data ranges before processing
print("X - Min:", X.min(axis=0), 
      "Max:", X.max(axis=0), 
      "Mean:", X.mean(axis=0), 
      "Std:", X.std(axis=0))
print("Y - Min:", Y.min(), 
      "Max:", Y.max(), 
      "Mean:", Y.mean(), 
      "Std:", Y.std())

# Save the scaler for testing
scaler = MinMaxScaler()
scaler.data_min_ = np.array([1, 1, 0, 0])  # Month, Day, Hour, Minute
scaler.data_max_ = np.array([12, 31, 23, 59])
scaler.min_ = -scaler.data_min_ / (scaler.data_max_ - scaler.data_min_)
scaler.scale_ = 1.0 / (scaler.data_max_ - scaler.data_min_)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Convert to PyTorch tensors and move to device
X = torch.FloatTensor(X).to(device)
Y = torch.FloatTensor(Y).reshape(-1, 1).to(device)

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X.cpu().numpy(), Y.cpu().numpy(), test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3333, random_state=42)

# Convert to tensors and reshape for LSTM (batch, seq_len=4, features=1)
X_train = torch.FloatTensor(X_train).reshape(-1, 4, 1).to(device)
X_val = torch.FloatTensor(X_val).reshape(-1, 4, 1).to(device)
X_test = torch.FloatTensor(X_test).reshape(-1, 4, 1).to(device)
y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
y_val = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
y_test = torch.FloatTensor(y_test).reshape(-1, 1).to(device)

# Define the LSTM model
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

# Initialize model, move to device, loss, and optimizer
model = PowerConsumptionLSTM(input_size=1, hidden_size=64, num_layers=1).to(device)
criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Training loop with early stopping
num_epochs = 50
batch_size = 128
batch_indices = range(0, len(X_train), batch_size)
val_batch_indices = range(0, len(X_val), batch_size)
best_val_loss = float('inf')
patience = 20
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
    
    # Average training loss for the epoch
    avg_train_loss = total_loss / len(batch_indices)
    
    # Evaluate on validation set in batches
    model.eval()
    val_loss_total = 0
    with torch.no_grad():
        for i in val_batch_indices:
            batch_X_val = X_val[i:i+batch_size]
            batch_y_val = y_val[i:i+batch_size]
            val_outputs = model(batch_X_val)
            val_loss = criterion(val_outputs, batch_y_val)
            val_loss_total += val_loss.item()
    
    avg_val_loss = val_loss_total / len(val_batch_indices)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # Step the scheduler based on validation loss
    scheduler.step(avg_val_loss)
    
    # Early stopping based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Load best model
model.load_state_dict(best_model_state)

# Evaluate the model on test set
model.eval()
test_loss_total = 0
test_mse_total = 0
with torch.no_grad():
    for i in val_batch_indices:
        batch_X_test = X_test[i:i+batch_size]
        batch_y_test = y_test[i:i+batch_size]
        test_outputs = model(batch_X_test)
        test_loss = criterion(test_outputs, batch_y_test)
        test_loss_total += test_loss.item()
        test_mse = ((test_outputs - batch_y_test) ** 2).mean().item()
        test_mse_total += test_mse
    
avg_test_loss = test_loss_total / len(val_batch_indices)
avg_test_mse = test_mse_total / len(val_batch_indices)
print(f'\nTest Loss (Huber): {avg_test_loss:.4f}')
print(f'Test Loss (MSE): {avg_test_mse:.4f}')

# Save the best model
torch.save(best_model_state, 'lstm_time_model.pth')