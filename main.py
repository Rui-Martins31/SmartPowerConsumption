# File to run the entire system

import matplotlib.pyplot as plt
import datetime

from REN_API.RenApiCall import get_electricity_price

#-----------------------------------------------------
# METHODS
#-----------------------------------------------------
def predict_power_consumption(hour: int) -> float:
    """Placeholder for the ML model. Returns a dummy consumption value in kWh."""
    # A simple pattern: higher consumption in the morning and evening
    if 7 <= hour <= 9 or 18 <= hour <= 21:
        return 1.5  # High consumption period
    return 0.5    # Low consumption period



#-----------------------------------------------------
# MAIN
#-----------------------------------------------------
def main() -> None:
    # Simulation Parameters
    num_days_to_simulate = 5
    battery_capacity_kwh = 10.0  # Total battery capacity
    battery_charge_rate_kw = 2.0  # Max charge rate
    battery_discharge_rate_kw = 2.0  # Max discharge rate
    battery_soc = 5.0  # Initial State of Charge (SoC)
    price_threshold = 0.08  # Price in €/kWh to trigger action
    
    # Lists to store simulation results
    hours_of_day = list(range(24))
    total_prices = []
    total_consumption = []
    total_battery_soc = []
    total_charged_power = []
    total_grid_power = []
    total_discharged_power = []

    current_date = datetime.datetime(year=2024, month=1, day=1)

    for day in range(num_days_to_simulate):
        current_date_str = (current_date + datetime.timedelta(days=day)).strftime("%Y-%m-%d")

        raw_prices = get_electricity_price(date=current_date_str)
        
        if "error" in raw_prices:
            print(f"Skipping day {current_date_str} due to an error: {raw_prices['error']}")
            continue
            
        prices = [val / 1000 for val in raw_prices]

        for hour in hours_of_day:
            consumption = predict_power_consumption(hour)
            charged_power = 0.0
            discharged_power = 0.0
            grid_power = 0.0
            
            # Decision logic based on price threshold
            if prices[hour] < price_threshold and battery_soc < battery_capacity_kwh:
                # Charge the battery
                charge_needed = battery_capacity_kwh - battery_soc
                charged_power = min(battery_charge_rate_kw, charge_needed, consumption) # Charge only what's consumed if consumption is low
                battery_soc += charged_power
                grid_power = consumption - charged_power
                
            elif prices[hour] > price_threshold and battery_soc > 0.0:
                # Discharge the battery
                discharge_possible = battery_soc
                discharged_power = min(battery_discharge_rate_kw, discharge_possible, consumption)
                battery_soc -= discharged_power
                grid_power = consumption - discharged_power
            
            else:
                # Use grid power directly
                grid_power = consumption

            # Append to history lists
            total_prices.append(prices[hour])
            total_consumption.append(consumption)
            total_battery_soc.append(battery_soc)
            total_charged_power.append(charged_power)
            total_discharged_power.append(discharged_power)
            total_grid_power.append(grid_power)

    # --- Plotting Results ---

    fig, axs = plt.subplots(4, 1, figsize=(15, 20))
    
    # Plot 1: Prices
    ax = axs[0]
    ax.set_title('Hourly Electricity Price')
    ax.set_ylabel('Price (€/kWh)')
    ax.grid(True)
    ax.plot(total_prices, color='blue')
    ax.axhline(y=price_threshold, color='red', linestyle='--', label='Threshold')
    ax.legend()
    
    # Plot 2: Consumption, Grid Use, and Battery Power
    ax = axs[1]
    ax.set_title('Power Flows')
    ax.set_ylabel('Power (kWh)')
    ax.grid(True)
    ax.plot(total_consumption, color='green', label='Total Consumption')
    ax.plot(total_grid_power, color='orange', label='Grid Power Use')
    ax.plot(total_charged_power, color='red', linestyle='--', label='Battery Charge')
    ax.plot(total_discharged_power, color='purple', linestyle='--', label='Battery Discharge')
    ax.legend()

    # Plot 3: Battery State of Charge
    ax = axs[2]
    ax.set_title('Battery State of Charge')
    ax.set_ylabel('SoC (kWh)')
    ax.grid(True)
    ax.plot(total_battery_soc, color='teal')
    ax.axhline(y=battery_capacity_kwh, color='gray', linestyle='--', label='Max Capacity')
    ax.legend()

    # Plot 4: Daily Costs
    total_cost_per_day = [0] * num_days_to_simulate
    hourly_costs = [p * c for p, c in zip(total_prices, total_grid_power)]

    for i in range(num_days_to_simulate):
        start_hour = i * 24
        end_hour = start_hour + 24
        total_cost_per_day[i] = sum(hourly_costs[start_hour:end_hour])

    ax = axs[3]
    ax.set_title('Daily Electricity Cost')
    ax.set_xlabel('Day')
    ax.set_ylabel('Cost (€)')
    ax.grid(True)
    ax.bar(range(num_days_to_simulate), total_cost_per_day, color='skyblue')

    plt.tight_layout()
    plt.show()
    



if __name__ == "__main__":
    main()