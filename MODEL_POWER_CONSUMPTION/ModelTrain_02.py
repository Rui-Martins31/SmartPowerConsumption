import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
from sklearn.preprocessing import MinMaxScaler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
Y = np.load('DATASET/Y.npy')

# Create lagged features (previous 5 measurements)
lag = 5
X_lagged = np.array([Y[i:i + lag] for i in range(len(Y) - lag)])
Y_target = Y[lag:].reshape(-1, 1)

# Normalize X_lagged
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_lagged)

# Chronological split (70% train, 20% val, 10% test)
n = len(Y_target)
train_end = int(0.7 * n)
val_end = int(0.9 * n)

X_train = X_normalized[:train_end]
y_train = Y_target[:train_end]

X_val = X_normalized[train_end:val_end]
y_val = Y_target[train_end:val_end]

X_test = X_normalized[val_end:]
y_test = Y_target[val_end:]

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

# Convert to tensors and move to device
X_train = torch.FloatTensor(X_train).to(device)
X_val = torch.FloatTensor(X_val).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.FloatTensor(y_train).to(device)
y_val = torch.FloatTensor(y_val).to(device)
y_test = torch.FloatTensor(y_test).to(device)

# Define the model
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

# Initialize model, loss, and optimizer
model = PowerConsumptionModel(input_size=5).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
torch.save(best_model_state, 'model_power_previous.pth')