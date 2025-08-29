# File to run the entire system

import matplotlib.pyplot as plt
import datetime
import math
import pandas as pd #Temporarily

from REN_API.RenApiCall import get_electricity_price
from BATTERY.ClassBattery import Battery
from UTILS.UtilsGraph import find_local_maxmin
from DATABASE.Database import Database
from MODEL_POWER_CONSUMPTION.Model_XGBoost import predict_next_24_hours

#-----------------------------------------------------
# METHODS
#-----------------------------------------------------
def predict_power_consumption(day: datetime, database: Database) -> list[float]:
    """
    Uses the XGBoost model to predict power consumption for the next 24 hours
    """
    historical_data: list[float] = []
    all_data: list[list[float]] = database.get_all_data()
    
    for daily_data in all_data:
        historical_data.extend(daily_data)
    
    HOUR_IN_DAY = 24
    NUM_DAYS = 7
    recent_consumption = historical_data[-HOUR_IN_DAY*NUM_DAYS:]
    
    # Predict
    predictions = predict_next_24_hours(day, recent_consumption)
    if not predictions:
        print("Warning: Model prediction returned empty list")
        return [2.0] * 24  # Default consumption of 2 kWh per hour
    
    return predictions



def start_day(curr_day: datetime, database: Database) -> tuple[list, list]:
    """
    Fetch price and consumption values.
    """

    # Get prices
    price_list: list = get_electricity_price(
        country = "pt-PT",
        date = curr_day.strftime("%Y-%m-%d")
    )
    if isinstance(price_list, dict):    # To be modified later
        return [], []
    else:
        price_list: list[float] = [ value/1000 for value in price_list ]

    # Get consumption
    consump_list: list[float] = predict_power_consumption(curr_day, database)
    if not consump_list or len(consump_list) != 24:
        print("Warning: Invalid consumption predictions")
        return [], []

    return price_list, consump_list



#-----------------------------------------------------
# MAIN
#-----------------------------------------------------
def main() -> None:
    
    # Initialize Database
    db: Database = Database(num_days=14)

    # Initialize Battery
    battery = Battery(
        battery_total_capacity = 10.0,
        battery_curr_capacity = 0.0,
        charger_rate = 2.2
    )

    # Vars
    curr_date: datetime = datetime.datetime(year=2024, month=1, day=1)
    # curr_date: datetime = datetime.datetime.now()
    num_days_sim: int = 14
    threshold_price: float = 0.08   # To be removed later

    price_list_total: list[float] = []
    consump_list_total: list[float] = []
    bat_cap_list_total: list[float] = []

    for day in range(num_days_sim):
        # Fetch values
        curr_date: datetime = curr_date + datetime.timedelta(days=day)
        price_list, consump_list = start_day(curr_day=curr_date, database=db)
        if price_list == []:    # If connection wasn't established with REN's Database
            continue
        
        # print(f"{consump_list = }, \n{len(consump_list) = }")

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

        # Update db
        db.set_new_day(pow_con_list=consump_list, date=curr_date)

        ## DEBUG
        print(f"Day {day}:\n    Current Battery Capacity: {battery.curr_capacity}\n")

    min_ind, price_min_list = find_local_maxmin(price_list_total, maxmin="min", smooth_area=5, smooth_threshold=0.01)
    max_ind, consump_max_list = find_local_maxmin(consump_list_total, maxmin="max", smooth_area=3)

    hours_list: list[int] = list(range(len(price_list_total)))

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
    plt.ylim(0, battery.total_capacity)

    plt.show()
    



if __name__ == "__main__":
    main()