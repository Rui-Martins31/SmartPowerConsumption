# File to run the entire system

import matplotlib.pyplot as plt
import datetime
import math
import pandas as pd #Temporarily

from REN_API.RenApiCall import get_electricity_price
from BATTERY.ClassBattery import Battery
from UTILS.UtilsGraph import find_local_maxmin
from DATABASE.Database import Database
from MODEL_POWER_CONSUMPTION.Model_XGBoost import create_df_from_date_to_predict, predict_future
from MODEL_POWER_CONSUMPTION.Model_XGBoost import predict_next_24_hours, get_pow_con_from_db

#-----------------------------------------------------
# METHODS
#-----------------------------------------------------
def predict_power_consumption(day: datetime.datetime, database: Database) -> list[float]:
    """
    Uses the XGBoost model to predict power consumption for the next 24 hours
    """
    historical_data: list[float] = []
    all_data: list[list[float]] = database.get_all_data()
    
    for daily_data in all_data:
        historical_data.extend(daily_data)
    
    HOURS_IN_DAY = 24
    NUM_DAYS = 7
    recent_consumption: list[float] = historical_data[-HOURS_IN_DAY*NUM_DAYS:]
    
    # Predict
    # df: pd.DataFrame = create_df_from_date_to_predict(date_to_predict=day, prev_values=recent_consumption)
    # predictions: list[float] = predict_future(df=df, forecast_hours=HOURS_IN_DAY)
    predictions: list[float] = predict_next_24_hours(start_dt=day, recent_hourly_consumption_kw=recent_consumption)
    predictions: list[float] = get_pow_con_from_db(day=day)
    if not predictions:
        print("Warning: Model prediction returned empty list")
        return [2.0] * 24  # Default consumption of 2 kWh per hour
    
    return predictions



def start_day(curr_day: datetime, database: Database) -> tuple[list, list]:
    """
    Fetch price and consumption values.
    """
    # Const
    HOURS_IN_DAY: int = 24

    # Get prices
    price_list: list = get_electricity_price(
        country = "pt-PT",
        date = curr_day.strftime("%Y-%m-%d")
    )
    if isinstance(price_list, dict):
        print(f"{price_list['error']}.")
        return [], []
    elif price_list == []:
        print(f"Warning: REN's API outputed an empty list.")
        return [], []
    elif len(price_list) != HOURS_IN_DAY:
        print("Warning: Price list does not contain 24 values.") 
        print(f"For {curr_day.strftime("%Y-%m-%d")} the {price_list = }")
        if len(price_list) < HOURS_IN_DAY:
            for _ in range(HOURS_IN_DAY - len(price_list)):
                price_list.append(price_list[-1])
        else:
            price_list = price_list[:HOURS_IN_DAY]
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
        battery_total_capacity = 20.0,
        battery_curr_capacity = 0.0
        # charger_rate = 2.2
    )

    # Vars
    DAYS_TO_SIM: int = int(30 * 12)
    HOURS_IN_DAY: int = 24
    curr_date: datetime = datetime.datetime(year=2024, month=1, day=1)
    # curr_date: datetime = datetime.datetime.now()

    # Lists to track
    price_list_total: list[float] = []
    consump_list_total: list[float] = []
    bat_cap_list_total: list[float] = []

    list_total_cost_no_bat: list[float] = []
    list_total_cost_with_bat: list[float] = []

    # Debug lists
    min_ind: list[int] = []
    max_ind: list[int] = []

    # Simulation loop
    for day in range(DAYS_TO_SIM):
        # Fetch values
        curr_date: datetime = curr_date + datetime.timedelta(days=1)
        price_list, consump_list = start_day(curr_day=curr_date, database=db)
        if price_list == []:    # If connection wasn't established with REN's Database
            print(f"Didn't get any price list for day {day}. Skipping to the next day...\n")
            continue
        
        # print(f"{consump_list = }, \n{len(consump_list) = }")

        MOVINNG_AVG_WINDOW = HOURS_IN_DAY * 1   # Best results when only considering the previous day
        if len(price_list_total) < MOVINNG_AVG_WINDOW:
            moving_avg_list: list[float] = price_list_total
        else: 
            moving_avg_list: list[float] = price_list_total[-MOVINNG_AVG_WINDOW:]
        when_to_buy, _ = find_local_maxmin(price_list, maxmin="min", smooth_area=5, moving_average=moving_avg_list)
        when_to_use, _ = find_local_maxmin(consump_list, maxmin="max", smooth_area=3)

        for hour in range(HOURS_IN_DAY):
            # Costs
            list_total_cost_no_bat.append(price_list[hour] * consump_list[hour])
            list_total_cost_with_bat.append(0.0)  # initialize as 0€

            # Check if it's time to buy
            if hour in when_to_buy:
                amount_bought: float = battery.charge()
                list_total_cost_with_bat[-1] += amount_bought * price_list[hour]    # Save the price that we payed to store energy
                list_total_cost_with_bat[-1] += consump_list[hour] * price_list[hour] # Save the price that cost to fulfill consumption
            else:
                amount_left: float = battery.discharge(consump_list[hour])
                list_total_cost_with_bat[-1] += amount_left * price_list[hour]  # Save the price in case battery hadn't enough energy

            # Save data
            price_list_total.append(price_list[hour])
            consump_list_total.append(consump_list[hour])
            bat_cap_list_total.append(battery.curr_capacity)

        # Update db
        db.set_new_day(pow_con_list=consump_list, date=curr_date)

        ## DEBUG
        print(f"Day {day}:\n    Current Battery Capacity: {battery.curr_capacity}\n")
        min_ind.extend([idx + day*HOURS_IN_DAY for idx in when_to_buy])
        max_ind.extend([idx + day*HOURS_IN_DAY for idx in when_to_use])

    # Plot results
    price_min_list: list[float] = [ price_list_total[idx] for idx in min_ind ]
    consump_max_list: list [float] = [ consump_list_total[idx] for idx in max_ind ]

    hours_list: list[int] = list(range(len(price_list_total)))

    # Plot Data
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

    # Plot Cost
    plt.figure(figsize=(10,5))

    plt.plot(hours_list, list_total_cost_no_bat, color="red")
    plt.plot(hours_list, list_total_cost_with_bat, color="green")
    plt.title("Cost (€)")
    plt.xlabel("Hours")
    plt.ylabel("Cost")

    plt.show()

    ## DEBUG
    print(f"\nTotal cost without battery ({DAYS_TO_SIM} days): {round(sum(list_total_cost_no_bat), 2)}€")
    print(f"Total cost with battery ({DAYS_TO_SIM} days): {round(sum(list_total_cost_with_bat), 2)}€")
    print(f"Total amount saved (€): {round(sum(list_total_cost_no_bat) - sum(list_total_cost_with_bat), 2)}€")
    print(f"Total amount saved (%): {round((sum(list_total_cost_no_bat) - sum(list_total_cost_with_bat))/(sum(list_total_cost_no_bat))*100, 2)}%")
    



if __name__ == "__main__":
    main()