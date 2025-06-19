import requests
import pandas as pd
import joblib
from datetime import datetime
import os
import json

API_KEY = "FJjxhcMeY9EIuzW8pDnSgiDfUMyanfsBXKJ7UOJ8"

print("[INFO] Starting daily NEO pipeline...")

today = datetime.utcnow().date()
url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={today}&end_date={today}&api_key={API_KEY}"

try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    print(f"[INFO] Data fetched for {today}.")
except Exception as e:
    print(f"[ERROR] Failed to fetch data from NASA API: {e}")
    exit(1)

if "near_earth_objects" not in data or not data["near_earth_objects"]:
    print("[WARNING] No NEO data found for today.")
    exit(0)

neos = []
for date in data["near_earth_objects"]:
    for neo in data["near_earth_objects"][date]:
        if not neo["close_approach_data"]:
            continue
        est_diameter = neo["estimated_diameter"]["meters"]
        close = neo["close_approach_data"][0]
        try:
            neos.append({
                "name": neo["name"],
                "date": date,
                "absolute_magnitude_h": neo["absolute_magnitude_h"],
                "estimated_diameter_min": est_diameter["estimated_diameter_min"],
                "estimated_diameter_max": est_diameter["estimated_diameter_max"],
                "velocity_kph": float(close["relative_velocity"]["kilometers_per_hour"]),
                "velocity_kps": float(close["relative_velocity"]["kilometers_per_second"]),
                "distance_km": float(close["miss_distance"]["kilometers"]),
                "distance_lunar": float(close["miss_distance"]["lunar"]),
                "distance_au": float(close["miss_distance"]["astronomical"]),
                "is_potentially_hazardous_asteroid": neo["is_potentially_hazardous_asteroid"]
            })
        except Exception as e:
            print(f"[WARNING] Skipping NEO due to data error: {e}")
            continue

if not neos:
    print("[WARNING] No valid NEOs found for today after processing.")
    exit(0)

print(f"[INFO] {len(neos)} NEOs found and processed.")

df_today = pd.DataFrame(neos)

# Clean/process like neo_clean.csv
numeric_cols = [
    "absolute_magnitude_h",
    "estimated_diameter_min",
    "estimated_diameter_max",
    "velocity_kph",
    "velocity_kps",
    "distance_km",
    "distance_lunar",
    "distance_au"
]

df_today = df_today.dropna()
for col in numeric_cols:
    df_today[col] = pd.to_numeric(df_today[col], errors="coerce")
df_today = df_today.dropna(subset=numeric_cols)

if df_today.empty:
    print("[WARNING] No NEOs left after cleaning. Exiting.")
    exit(0)

print(f"[INFO] Cleaned data shape: {df_today.shape}")

# Load all models
model_keys = {
    "rf": "model_rf.pkl",
    "dt": "model_dt.pkl",
    "lr": "model_lr.pkl",
    "knn": "model_knn.pkl",
    "svm": "model_svm.pkl",
    "xgb": "model_xgb.pkl",
    "myclf": "model_myclf.pkl"
}
models = {}
for key, path in model_keys.items():
    if os.path.exists(path):
        try:
            models[key] = joblib.load(path)
            print(f"[INFO] Loaded {path}.")
        except Exception as e:
            print(f"[WARNING] Could not load {path}: {e}")

X_live = df_today[numeric_cols]

# For each model, predict and save
for key, model in models.items():
    try:
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict(X_live)
            y_proba = model.predict_proba(X_live)
            confidences = y_proba.max(axis=1)
        else:
            y_pred = model.predict(X_live)
            confidences = [None] * len(y_pred)
        df_pred = df_today.copy()
        df_pred["predicted_hazard"] = y_pred
        df_pred["predicted_hazard_label"] = df_pred["predicted_hazard"].map({1: "Hazardous", 0: "Safe"})
        df_pred["confidence"] = confidences
        # Save as JSON for frontend
        records = df_pred[["name", "date", "estimated_diameter_min", "estimated_diameter_max", "distance_km", "predicted_hazard_label", "confidence"]].to_dict(orient="records")
        with open(f"data/daily_predictions_{key}.json", "w") as f:
            json.dump(records, f, indent=2)
        print(f"✅ Saved daily_predictions_{key}.json")
        # Save hazardous distribution chart for this model
        from ml_model import save_hazardous_distribution_chart
        save_hazardous_distribution_chart(key)
    except Exception as e:
        print(f"[ERROR] Prediction failed for {key}: {e}")

# For backward compatibility, save the default CSV for myclf
if "myclf" in models:
    df_pred = df_today.copy()
    y_pred = models["myclf"].predict(X_live)
    df_pred["predicted_hazard"] = y_pred
    df_pred["predicted_hazard_label"] = df_pred["predicted_hazard"].map({1: "Hazardous", 0: "Safe"})
    df_pred.to_csv("neo_today_predictions.csv", index=False)
    print("✅ Predictions saved as neo_today_predictions.csv") 