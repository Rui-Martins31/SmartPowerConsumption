import requests
from datetime import datetime
from pprint import pprint
import os
from dotenv import load_dotenv

# API configuration
load_dotenv()

API_URL = os.getenv("API_URL")
CULTURE = os.getenv("CULTURE")
DATE = os.getenv("DATE")


def get_electricity_price(country:str = "pt-PT", date:datetime = datetime.now().strftime("%Y-%m-%d")) -> list:
    """
    Fetch daily electricity (MWh) market prices from the REN DataHub API.
    Returns a list with all the values.
    """
    
    params = {
        "culture": country,
        "date": date
    }
    
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()  # Raise an error for bad status codes

        response = response.json()
        return response["series"][0]["data"]
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch data: {str(e)}"}

def main():
    # Fetch market prices
    result: list = get_electricity_price(country=CULTURE, date=DATE)
    
    if "error" in result:
        print(result["error"])
    else:
        print(f"Electricity Market Prices for {CULTURE} on {DATE}:")
        if isinstance(result, list):
            # print(f"Price: {list( map(lambda x: round(x / 1000, 5), result["series"][0]["data"]) )} €/kWh")^
            #print(f"Price: { result["series"][0]["data"] } €/MWh")
            print(f"Price: { result } €/MWh")
        else:
            print("Data format not recognized.")
        
        #pprint(result)

if __name__ == "__main__":
    main()