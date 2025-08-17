# File to run the entire system

import matplotlib.pyplot as plt
import datetime
import numpy as np
from scipy.signal import argrelextrema

from REN_API.RenApiCall import get_electricity_price
from BATTERY.ClassBattery import Battery

#-----------------------------------------------------
# METHODS
#-----------------------------------------------------
def predict_power_consumption(hour: int) -> float:
    """
    Placeholder for the ML model. Returns a dummy consumption value in kWh.
    """

    # A simple pattern: higher consumption in the morning and evening
    if 7 <= hour <= 9 or 18 <= hour <= 21:
        return 1.5  # High consumption period
    return 0.5    # Low consumption period



def find_local_min(data_list: list[float], smooth_area: int = 0) -> list[float]:
    """
    Finds local minimums in a list.
    """

    arr: np.ndarray = np.array(data_list)
    minima_indices = argrelextrema(arr, np.less)[0]
    
    #return arr[minima_indices].tolist()
    return minima_indices, arr[minima_indices].tolist()



def start_day(curr_day:str):
    """
    Fetches price and consumption values.
    """

    # Get prices
    price_list: list = get_electricity_price(
        country = "pt-PT",
        date = curr_day
    )
    if isinstance(price_list, dict):    # To be modified later
        return [], []
    else:
        price_list: list[float] = [ value/1000 for value in price_list ]

    # Get consumption
    consump_list: list[float] = []

    for hour in range(24):
        pow_con: float = predict_power_consumption(hour=hour)
        consump_list.append(pow_con)

    return price_list, consump_list



#-----------------------------------------------------
# MAIN
#-----------------------------------------------------
def main() -> None:
    
    # Initialize Battery
    battery = Battery(
        battery_total_capacity = 10.0,
        battery_curr_capacity = 0.0
    )

    # Vars
    curr_date: datetime = datetime.datetime(year=2024, month=1, day=1)
    num_days_sim: int = 7
    threshold_price: float = 0.08   # To be removed later
    hours_list: list[int] = [ h for h in range(24 * num_days_sim) ]

    price_list_total: list[float] = []
    consump_list_total: list[float] = []

    for day in range(num_days_sim):
        # Fetch values
        curr_date: datetime = curr_date + datetime.timedelta(days=day)
        price_list, consump_list = start_day(curr_day = curr_date.strftime("%Y-%m-%d"))
        
        for value in price_list:
            price_list_total.append(value)

        for value in consump_list:
            consump_list_total.append(value)
            

        # print(f"Price list: {price_list}")
        # print(f"Consumption list: {consump_list}")
        print(f"Day {day}:\n    Current Battery Capacity: {battery.curr_capacity}\n")

    ind, price_min_list = find_local_min(price_list_total)
    is_local_min = [ hour for id, hour in enumerate(hours_list) if id in ind  ]

    plt.figure(figsize=(20,10))
    plt.subplot(211)
    plt.plot(hours_list, price_list_total, color="red")
    # plt.plot(hours_list[is_local_min], price_list_total[is_local_min], color="green")
    plt.scatter(is_local_min, price_min_list, color="green", marker="o")

    plt.subplot(212)
    plt.plot(hours_list, consump_list_total, color="blue")
    plt.show()
    



if __name__ == "__main__":
    main()