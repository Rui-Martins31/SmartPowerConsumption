# Fetch values from REN's Database and updates them to be used as inputs in the ML model

import collections
import os
import csv
import matplotlib.pyplot as plt

class Database:
    
    def __init__(self, num_days: int = 14):
        """
        Initializes the queue with a fixed size.
        """
        if not isinstance(num_days, int) or num_days <= 0:
            raise ValueError("Number of days must be a positive integer.")
        
        self.queue = collections.deque(maxlen=num_days)

    def initialize_from_csv(self, filepath):
        """
        Populates the queue by reading the most recent data from a CSV file.
        Each row in the CSV is expected to be a single hourly sample with a timestamp.
        """
        if not os.path.exists(filepath):
            print(f"Warning: File not found at '{filepath}'. Cannot initialize queue.")
            return

        print(f"\n--- Initializing queue from '{filepath}' ---")
        
        daily_data = collections.OrderedDict()
        
        with open(filepath, mode='r') as file:
            reader = csv.reader(file)
            next(reader, None)
            
            for row in reader:
                if len(row) < 2:
                    continue

                timestamp_str, power_str = row[0], row[1]
                
                try:
                    date_part = timestamp_str.split(' ')[0]
                    power_value = float(power_str)
                    
                    # Group data by date
                    if date_part not in daily_data:
                        daily_data[date_part] = []
                    daily_data[date_part].append(power_value)
                    
                except (ValueError, IndexError):
                    print(f"Skipping invalid row: {row}")
                    continue

        all_complete_days = [data for data in daily_data.values() if len(data) == 24]

        # Extract the amount of data that fills the queue
        data_to_load = all_complete_days[-self.queue.maxlen:]
        
        # Add the data to the queue
        for day_data in data_to_load:
            self.queue.append(day_data)
        
        print(f"Initialization complete. Queue now contains data for {len(self.queue)} days.")

    def get_all_data(self):
        """
        Returns all data currently in the queue.
        The data is ordered from oldest to newest.
        """
        return list(self.queue)

    def get_latest_data(self):
        """
        Returns the data for the most recent day.
        """
        if not self.queue:
            return None
        return self.queue[-1]

    def get_oldest_data(self):
        """
        Returns the data for the oldest day currently in the queue.
        """
        if not self.queue:
            return None
        return self.queue[0]

    def get_current_size(self):
        """
        Returns the current number of days stored in the queue.
        """
        return len(self.queue)



if __name__ == "__main__":
    db = Database(num_days = 14)
    db.initialize_from_csv('../MODEL_POWER_CONSUMPTION/DATASET/dataset_power_visualizer.csv')

    all_data: list[list[float]] = db.get_all_data()
    print(all_data)

    hours_list: list[int] = [ h for h in range(24 * 14)]
    temp_list: list[float] = []
    for data in all_data:
        for value in data:
            temp_list.append(value)

    plt.figure(figsize=(15,5))
    plt.plot(hours_list, temp_list)
    plt.show()