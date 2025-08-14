import requests
from datetime import datetime
from pprint import pprint

# API configuration
API_URL = "https://servicebus.ren.pt/datahubapi/electricity/ElectricityMarketPricesDaily"
CULTURE = "pt-PT"  # Portuguese language and region
DATE = datetime.now().strftime("%Y-%m-%d")  # Current date

def get_electricity_price(country:str = CULTURE, date:datetime = DATE):
    """
    Fetch daily electricity market prices for Portugal from the REN DataHub API.
    
    Returns:
        dict: JSON response with price data or error message
    """
    params = {
        "culture": country,
        "date": date
    }
    
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch data: {str(e)}"}

def main():
    # Fetch today's electricity market prices for Portugal
    result = get_electricity_price()
    
    if "error" in result:
        print(result["error"])
    else:
        # Assuming the API returns a list or object with price data
        print(f"Electricity Market Prices for Portugal on {DATE}:")
        if isinstance(result, dict):
            print(f"Price: {result["series"][0]["data"]} €/MWh")
        else:
            print("Data format not recognized.")
        
        #pprint(result)

if __name__ == "__main__":
    main()