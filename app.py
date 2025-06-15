from flask import Flask, render_template, jsonify, request, redirect, url_for, send_file
import os
import json
import joblib
import pandas as pd
from ml_model import train_model
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from apscheduler.schedulers.background import BackgroundScheduler
import requests
import datetime

app = Flask(__name__)
model = joblib.load("model.pkl")  # Load trained ML model

# Store last prediction for report download
last_prediction = {
    'diameter': None,
    'velocity': None,
    'distance': None,
    'prediction_label': None,
    'confidence': None
}

def load_feature_importances():
    path = os.path.join("data", "feature_importance.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

@app.route("/")
def index():
    data_file = os.path.join("data", "nasa_data.json")
    data = {}
    if os.path.exists(data_file):
        with open(data_file, "r") as f:
            data = json.load(f)
    importances = load_feature_importances()
    return render_template("index.html", data=data, prediction=None, importances=importances)

@app.route("/api/data")
def api_data():
    data_file = os.path.join("data", "nasa_data.json")
    if os.path.exists(data_file):
        with open(data_file, "r") as f:
            data = json.load(f)
    else:
        data = {"near_earth_objects": {}}
    return jsonify(data)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        diameter = float(request.form["diameter"])
        velocity = float(request.form["velocity"])
        distance = float(request.form["distance"])
        input_df = pd.DataFrame([[diameter, velocity, distance]], columns=['diameter', 'velocity', 'distance'])
        proba = model.predict_proba(input_df)[0]
        prediction = model.predict(input_df)[0]
        prediction_label = "Yes" if prediction == 1 else "No"
        confidence = proba[1] if prediction == 1 else proba[0]
        # Store for report
        global last_prediction
        last_prediction = {
            'diameter': diameter,
            'velocity': velocity,
            'distance': distance,
            'prediction_label': prediction_label,
            'confidence': confidence
        }
        # Load data for template
        data_file = os.path.join("data", "nasa_data.json")
        if os.path.exists(data_file):
            with open(data_file, "r") as f:
                data = json.load(f)
        else:
            data = {"near_earth_objects": {}}
        return render_template("index.html", prediction=prediction_label, confidence=confidence, data=data, importances=load_feature_importances())
    except:
        # Also pass data in error case
        data_file = os.path.join("data", "nasa_data.json")
        if os.path.exists(data_file):
            with open(data_file, "r") as f:
                data = json.load(f)
        else:
            data = {"near_earth_objects": {}}
        return render_template("index.html", prediction="error", confidence=None, data=data, importances=load_feature_importances())

@app.route("/download_report")
def download_report():
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(72, 720, "Asteroid Hazard Prediction Report")
    p.setFont("Helvetica", 12)
    y = 680
    p.drawString(72, y, f"Diameter (m): {last_prediction['diameter']}")
    y -= 24
    p.drawString(72, y, f"Velocity (km/h): {last_prediction['velocity']}")
    y -= 24
    p.drawString(72, y, f"Distance (km): {last_prediction['distance']}")
    y -= 24
    p.drawString(72, y, f"Prediction: {last_prediction['prediction_label']}")
    y -= 24
    if last_prediction['confidence'] is not None:
        p.drawString(72, y, f"Confidence: {last_prediction['confidence']*100:.1f}%")
    p.showPage()
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="asteroid_prediction_report.pdf", mimetype='application/pdf')

@app.route("/retrain", methods=["POST"])
def retrain():
    train_model()
    return redirect(url_for("index"))

def fetch_and_update_nasa_data():
    print("[Scheduler] Fetching latest NEO data and retraining model...")
    API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")
    start_date = datetime.date.today()
    end_date = start_date + datetime.timedelta(days=7)
    url = 'https://api.nasa.gov/neo/rest/v1/feed'
    params = {
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'api_key': API_KEY
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        os.makedirs('data', exist_ok=True)
        with open('data/nasa_data.json', 'w') as f:
            json.dump(data, f, indent=2)
        print("[Scheduler] NEO data updated. Retraining model...")
        train_model()
    except Exception as e:
        print(f"[Scheduler] Failed to fetch or update NEO data: {e}")

# Setup APScheduler to run fetch_and_update_nasa_data once per week
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(fetch_and_update_nasa_data, 'interval', weeks=1, next_run_time=datetime.datetime.now())
scheduler.start()

if __name__ == '__main__':
    app.run()
