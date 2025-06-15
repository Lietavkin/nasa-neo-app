import requests
import datetime
import os
import json

# Set up dates
start_date = datetime.date.today()
end_date = start_date + datetime.timedelta(days=7)

# Format dates as YYYY-MM-DD
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# NASA NeoWs API endpoint
url = 'https://api.nasa.gov/neo/rest/v1/feed'
params = {
    'start_date': start_date_str,
    'end_date': end_date_str,
    'api_key': 'DEMO_KEY'  # Replace with your actual API key if you have one
}

# Fetch data
response = requests.get(url, params=params)
response.raise_for_status()  # Raise error if request failed

data = response.json()

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Save JSON response
with open('data/nasa_data.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"✅ Data saved to data/nasa_data.json for {start_date_str} to {end_date_str}")
