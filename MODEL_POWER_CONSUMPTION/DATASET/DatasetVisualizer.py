import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Dataframe
df = pd.read_csv(
    './dataset_power_csv.csv',
    sep=',',
    dayfirst=True,
    low_memory=False
)

df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
df = df.set_index('DateTime')

columns_to_keep: list = ['Global_active_power']
df = df[columns_to_keep]
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')

print(df.head())

df['hourly_group'] = df.index.floor('h')
df = df.groupby('hourly_group').mean().round(3)
df.index.name = 'Date'

df.dropna(inplace=True)

print(df.head())

daily_counts = df.index.to_series().dt.date.value_counts()
complete_days = daily_counts[daily_counts == 24].index

df = df[df.index.to_series().dt.date.isin(complete_days)]

# Save dataframe
df.to_csv('./dataset_power_visualizer.csv')



# Plot
num_days = 8
start_date = datetime.datetime(year=2008, month=3, day=1)
curr_date = start_date

plt.figure(figsize=(20, 10))
plt.subplots_adjust(hspace=0.5)

for day in range(num_days):
    next_date = curr_date + datetime.timedelta(hours=24)
    df_snapshot = df.loc[ curr_date : next_date ]
    plt.subplot(2, 4, day + 1)
    plt.plot(df_snapshot.index.hour, df_snapshot['Global_active_power'])
    plt.title(curr_date.strftime("%Y/%m/%d"))
    plt.xlabel("Hours")
    plt.ylabel("kWh")

    curr_date = next_date

plt.show()