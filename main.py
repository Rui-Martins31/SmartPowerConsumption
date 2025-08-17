# File to run the entire system

import matplotlib.pyplot as plt

from REN_API.RenApiCall import get_electricity_price


def main() -> None:
    # Get energy prices
    hours: list = [ h for h in range(0, 24) ]
    price_list: list = [ val/1000 for val in get_electricity_price() ]
    price_threshold: float = 0.12

    price_above: list = [price if price > price_threshold else float('nan') for price in price_list]
    price_below: list = [price if price <= price_threshold else float('nan') for price in price_list]

    plt.figure(figsize=(20,10))
    plt.plot(hours, price_above, color='red', marker='o', linestyle='-', label='Price Above Threshold')
    plt.plot(hours, price_below, color='green', marker='o', linestyle='-', label='Price Below Threshold')
    plt.axhline(y=price_threshold, color='blue', linestyle='--', label=f'Threshold ({price_threshold} €/kWh)')
    plt.title('Hourly Electricity Price')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Price (€/kWh)')
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()

    plt.show()
    



if __name__ == "__main__":
    main()