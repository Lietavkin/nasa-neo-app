import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import joblib

API_KEY = "FJjxhcMeY9EIuzW8pDnSgiDfUMyanfsBXKJ7UOJ8"
BASE_URL = "https://api.nasa.gov/neo/rest/v1/feed"

# Calculate date range for the last 7 days
end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=6)

params = {
    "start_date": start_date.strftime("%Y-%m-%d"),
    "end_date": end_date.strftime("%Y-%m-%d"),
    "api_key": API_KEY
}

response = requests.get(BASE_URL, params=params)
response.raise_for_status()
data = response.json()

asteroids = []
for date in data["near_earth_objects"]:
    for a in data["near_earth_objects"][date]:
        try:
            ca = a["close_approach_data"][0]
            asteroids.append({
                "name": a["name"],
                "absolute_magnitude_h": a["absolute_magnitude_h"],
                "estimated_diameter_min": a["estimated_diameter"]["kilometers"]["estimated_diameter_min"],
                "estimated_diameter_max": a["estimated_diameter"]["kilometers"]["estimated_diameter_max"],
                "velocity_kph": float(ca["relative_velocity"]["kilometers_per_hour"]),
                "velocity_kps": float(ca["relative_velocity"]["kilometers_per_second"]),
                "distance_km": float(ca["miss_distance"]["kilometers"]),
                "distance_lunar": float(ca["miss_distance"]["lunar"]),
                "distance_au": float(ca["miss_distance"]["astronomical"]),
                "is_potentially_hazardous_asteroid": a["is_potentially_hazardous_asteroid"]
            })
        except Exception as e:
            continue

# Create DataFrame and save to CSV
neo_df = pd.DataFrame(asteroids)
neo_df.to_csv("neo_full_data.csv", index=False)
print(f"Saved {len(neo_df)} asteroids to neo_full_data.csv")

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Save JSON response
with open('data/nasa_data.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"✅ Data saved to data/nasa_data.json for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

joblib.dump(model, "model.pkl")
