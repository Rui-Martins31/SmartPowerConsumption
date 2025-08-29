# Fetch values from REN's Database and updates them to be used as inputs in the ML model

import collections
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from supabase import create_client, Client

class Database:
    
    def __init__(self, num_days: int = 14, 
                 supabase_url: str = "",
                 supabase_key: str = "",
                 table_name: str = "power_consumption"):
        """
        Initializes the queue with a fixed size.
        """
        print("Initializing Database object... ", end="")

        # bypass method's args
        load_dotenv()
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')

        if not supabase_url or not supabase_key:
            raise ValueError("Supabase's URL or Key were not loaded correctly.")
        
        if not isinstance(num_days, int) or num_days <= 0:
            raise ValueError("Number of days must be a positive integer.")
        
        self.queue = collections.deque(maxlen=num_days)
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.table_name: str = table_name
        self.last_date: str | None = None

        print("Done")

        self.initialize()


    def initialize(self) -> None:
        """
        Get data from Supabase and populate local queue
        """
        print("Fetching data from Supabase... ", end="")
        HOURS_IN_DAY: int = 24
        NUM_DATA: int = self.queue.maxlen * HOURS_IN_DAY
        response = (
            self.supabase.table(self.table_name)
            .select(f"datetime, pow_con_value")
            .order("datetime", desc=True)
            .limit(NUM_DATA)
            .execute()
        )
        data = response.data
        if not data:
            raise ValueError(f"Couldn't fetch data from Supabase's table {self.table_name}")
        print("Done")

        # print(f"{data = }")
        # print(f"{len(data) = }")

        print("Processing data... ", end="")
        try:
            self.last_date: str = data[-1]['datetime']

            values_to_store: list[float] = [ data[idx]['pow_con_value'] for idx in range(0, NUM_DATA) ]

            if any([self.last_date == "", values_to_store == [], len(values_to_store) != NUM_DATA]):
                raise ValueError("Data is empty or is not complete.")
            
            for idx in range(1, self.queue.maxlen+1):
                if idx == 1:
                    self.queue.append(list(values_to_store[-idx*HOURS_IN_DAY:])[::-1]) # [::-1] reverses the list
                else:
                    self.queue.append(list(values_to_store[-idx*HOURS_IN_DAY:-(idx-1)*HOURS_IN_DAY])[::-1])

            if not self.queue or len(self.queue) < self.queue.maxlen:
                raise ValueError("Queue is empty or is not complete.")
        except:
            raise ValueError("Error while parsing data.")
        print("Done")

        # print(f"{self.queue = }")

        print("Database is now initialized and fully populated!")
        return None

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
    
    def set_new_day(self, pow_con_list: list[float], date: str) -> bool:
        """
        Adds a new day of power consumption data to the database.
        
        Args:
            pow_con_list (list[float]): List of 24 hourly power consumption values
            date (str): Date in ISO format (YYYY-MM-DD)
            
        Returns:
            bool: True if successful, False otherwise
        """
        HOURS_IN_DAY = 24
        
        # Validate input
        if not isinstance(pow_con_list, list) or len(pow_con_list) != HOURS_IN_DAY:
            print(f"Error: Power consumption list must contain exactly {HOURS_IN_DAY} values")
            return False
            
        if not all(isinstance(x, (int, float)) for x in pow_con_list):
            print("Error: All values must be numbers")
            return False
        
        self.queue.append(pow_con_list)
        self.last_date: str = f"{date} 23:00:00"
        
        return True
        # Eventually add to Supabase's database
        """
        try:
            # Add to Supabase
            data_to_insert = [
                {"datetime": f"{date} {hour:02d}:00:00", "pow_con_value": value}
                for hour, value in enumerate(pow_con_list)
            ]
            
            self.supabase.table(self.table_name).insert(data_to_insert).execute()
            
            # Update local queue
            self.queue.append(pow_con_list)  # Old day will be automatically removed due to maxlen
            self.last_date = f"{date} 23:00:00"  # Last hour of the day
            
            return True
        except Exception as e:
            print(f"Error adding new day to database: {e}")
            return False
        """



if __name__ == "__main__":
    db = Database(num_days = 14)
    all_data: list = db.get_all_data()
    # print(f"{all_data = }")
    
    hours_list: list[int] = [ h for h in range(24 * 14)]

    temp_list: list[float] = []
    for data in all_data:
        for value in data:
            temp_list.append(value)

    plt.figure(figsize=(15,5))
    plt.plot(hours_list, temp_list)
    plt.show()
    