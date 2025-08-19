# File to run the entire system

import matplotlib.pyplot as plt
import datetime
import numpy as np
import math
from scipy.signal import argrelextrema

from REN_API.RenApiCall import get_electricity_price
from BATTERY.ClassBattery import Battery

#-----------------------------------------------------
# METHODS
#-----------------------------------------------------
def predict_power_consumption(hour: int) -> float:
    """
    Placeholder for the ML model. 
    Returns a dummy consumption value in kWh.
    """

    # A simple pattern: higher consumption in the morning and evening
    # Peak around 8 AM
    morning_peak = 1.5 * math.exp(-((hour - 8)**2) / 4)
    
    # Peak around 20 PM
    evening_peak = 1.8 * math.exp(-((hour - 20)**2) / 4)
    
    # Base consumption
    base_consumption = 0.5
    
    # Return the sum of base and peaks
    return base_consumption + morning_peak + evening_peak


def find_local_maxmin(data_list: list[float], maxmin: str = "max", smooth_area: int = 1, smooth_threshold: float = 0.01) -> (tuple[list[int], list[float]] | None):
    """
    Finds local maximums or minimums in a list.
    """

    arr: np.ndarray = np.array(data_list)
    if maxmin == "max":
        maxmin_indices = argrelextrema(arr, np.greater)[0]
    elif maxmin == "min": 
        maxmin_indices = argrelextrema(arr, np.less)[0]
    else:
        print("maxmin is not valid!")
        return None

    # Get neighbors
    neigh_indices: list[int] = []
    for idx in maxmin_indices:
        for incr in range(smooth_area, 0, -1):
            if (arr[idx - incr] <= arr[idx] * 1.0+smooth_threshold) and (arr[idx - incr] >= arr[idx] * 1.0-smooth_threshold) and (idx > 0):
                neigh_indices.append(idx - incr)
        
        neigh_indices.append(idx)

        for incr in range(1, smooth_area+1, 1):
            if (arr[idx + incr] <= arr[idx] * 1.0+smooth_threshold) and (arr[idx + incr] >= arr[idx] * 1.0-smooth_threshold) and (idx < len(arr) - 1):
                neigh_indices.append(idx + incr)

    # Non-repeatable
    neigh_indices: set[int] = set(neigh_indices)
    neigh_indices: list[int] = list(neigh_indices)
    
    #return arr[maxmin_indices].tolist()
    #return maxmin_indices, arr[maxmin_indices].tolist()
    return neigh_indices, arr[neigh_indices].tolist()



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
        battery_curr_capacity = 0.0,
        charger_rate = 2.2
    )

    # Vars
    curr_date: datetime = datetime.datetime(year=2024, month=1, day=1)
    num_days_sim: int = 14
    threshold_price: float = 0.08   # To be removed later
    hours_list: list[int] = [ h for h in range(24 * num_days_sim) ]

    price_list_total: list[float] = []
    consump_list_total: list[float] = []
    bat_cap_list_total: list[float] = []

    for day in range(num_days_sim):
        # Fetch values
        curr_date: datetime = curr_date + datetime.timedelta(days=day)
        price_list, consump_list = start_day(curr_day = curr_date.strftime("%Y-%m-%d"))
        if price_list == []:    # If connection wasn't established with REN's Database
            continue
        
        when_to_buy, _ = find_local_maxmin(price_list, maxmin="min", smooth_area=3)
        when_to_use, _ = find_local_maxmin(consump_list, maxmin="max", smooth_area=3)

        for hour in range(24):

            # Check if it's time to buy
            if hour in when_to_buy:
                battery.charge()

            # Check if it's time to use the battery
            if hour in when_to_use:
                battery.discharge(consump_list[hour])

            # Save data
            price_list_total.append(price_list[hour])
            consump_list_total.append(consump_list[hour])
            bat_cap_list_total.append(battery.curr_capacity)

        ## DEBUG
        print(f"Day {day}:\n    Current Battery Capacity: {battery.curr_capacity}\n")

    min_ind, price_min_list = find_local_maxmin(price_list_total, maxmin="min", smooth_area=5, smooth_threshold=0.01)
    max_ind, consump_max_list = find_local_maxmin(consump_list_total, maxmin="max", smooth_area=3)

    plt.figure(figsize=(20,15))

    plt.subplot(311)
    plt.plot(hours_list, price_list_total, color="red")
    plt.scatter(min_ind, price_min_list, color="green", marker="o")
    plt.title("Electrictity Price (kWh)")
    plt.xlabel("Hours")
    plt.ylabel("kWh")

    plt.subplot(312)
    plt.plot(hours_list, consump_list_total, color="blue")
    plt.scatter(max_ind, consump_max_list, color="green", marker="o")
    plt.title("Power Consumption (kWh)")
    plt.xlabel("Hours")
    plt.ylabel("kWh")
    
    plt.subplot(313)
    plt.plot(hours_list, bat_cap_list_total, color="pink")
    plt.title("Battery Current Capacity (kWh)")
    plt.xlabel("Hours")
    plt.ylabel("kWh")

    plt.show()
    



if __name__ == "__main__":
    main()