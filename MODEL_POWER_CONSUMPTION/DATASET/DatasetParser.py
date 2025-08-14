import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(
    'dataset_power_csv.csv',
    sep=',',
    parse_dates={'DateTime': ['Date', 'Time']},
    dayfirst=True,
    low_memory=False
)

df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
df.dropna(subset=['Global_active_power'], inplace=True)
print(df.head())

# Separate fields
df['Year'] = df['DateTime'].dt.year     #Won't use for now
df['Month'] = df['DateTime'].dt.month
df['Day'] = df['DateTime'].dt.day
df['Hour'] = df['DateTime'].dt.hour
df['Minute'] = df['DateTime'].dt.minute

#X = df[['Year', 'Month', 'Day', 'Hour', 'Minute']].to_numpy()
X = df[['Month', 'Day', 'Hour', 'Minute']].to_numpy()
Y = df['Global_active_power'].to_numpy()

# Normalize X values
scaler_X = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)

print(X_normalized[:5])
print(Y[:5])

# Save as .npy files
np.save('X.npy', X_normalized)
np.save('Y.npy', Y) 

print(f"Saved X_normalized.npy with shape {X_normalized.shape}")
print(f"Saved Y.npy with shape {Y.shape}")