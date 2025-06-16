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
from sklearn.metrics import confusion_matrix
import numpy as np

app = Flask(__name__)
# Load all trained ML models
models = {
    "Random Forest": joblib.load("model_rf.pkl"),
    "Decision Tree": joblib.load("model_dt.pkl"),
    "Logistic Regression": joblib.load("model_lr.pkl")
}

# Store last prediction for report download
last_prediction = {
    'diameter': None,
    'velocity': None,
    'distance': None,
    'prediction_label': None,
    'confidence': None,
    'model': None
}

def load_feature_importances():
    path = os.path.join("data", "feature_importance.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def get_model_comparison_data():
    # Load metrics
    with open("static/model_comparison_results.json") as f:
        metrics = json.load(f)
    # Load confusion matrices
    with open("static/model_confusion_matrices.json") as f:
        confusion_matrices = json.load(f)
    # Build confusion matrix HTML for each model
    def confusion_html(cm, model_name=None):
        label = f'<h3 class="cm-model-label">{model_name}</h3>' if model_name else ''
        return f'''{label}<table class="confusion-matrix-table"><tr><th></th><th>Pred 0</th><th>Pred 1</th></tr><tr><th>Actual 0</th><td>{cm[0][0]}</td><td>{cm[0][1]}</td></tr><tr><th>Actual 1</th><td>{cm[1][0]}</td><td>{cm[1][1]}</td></tr></table>'''
    rf_confusion_html = confusion_html(confusion_matrices["Random Forest"], "Random Forest")
    dt_confusion_html = confusion_html(confusion_matrices["Decision Tree"], "Decision Tree")
    lr_confusion_html = confusion_html(confusion_matrices["Logistic Regression"], "Logistic Regression")
    # Build metrics table HTML for each model
    def metrics_html(metrics, key):
        m = metrics[key]
        return f'''<table class="metrics-table"><tr><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr><tr><td>{m['accuracy']:.2f}</td><td>{m['precision']:.2f}</td><td>{m['recall']:.2f}</td><td>{m['f1']:.2f}</td></tr></table>'''
    rf_metrics_html = metrics_html(metrics, "Random Forest")
    dt_metrics_html = metrics_html(metrics, "Decision Tree")
    lr_metrics_html = metrics_html(metrics, "Logistic Regression")
    return rf_confusion_html, dt_confusion_html, lr_confusion_html, rf_metrics_html, dt_metrics_html, lr_metrics_html

@app.route("/")
def index():
    data_file = os.path.join("data", "nasa_data.json")
    data = {}
    if os.path.exists(data_file):
        with open(data_file, "r") as f:
            data = json.load(f)
    importances = load_feature_importances()
    # Always reload latest confusion matrices and metrics
    rf_confusion_html, dt_confusion_html, lr_confusion_html, rf_metrics_html, dt_metrics_html, lr_metrics_html = get_model_comparison_data()
    with open("static/model_comparison_results.json") as f:
        model_metrics = json.load(f)
    return render_template(
        "index.html",
        data=data,
        prediction=None,
        importances=importances,
        selected_model="Random Forest",
        rf_confusion_html=rf_confusion_html,
        dt_confusion_html=dt_confusion_html,
        lr_confusion_html=lr_confusion_html,
        rf_metrics_html=rf_metrics_html,
        dt_metrics_html=dt_metrics_html,
        lr_metrics_html=lr_metrics_html,
        model_metrics=model_metrics
    )

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
        selected_model = request.form.get("model_choice", "Random Forest")
        input_df = pd.DataFrame([[diameter, velocity, distance]], columns=['diameter', 'velocity', 'distance'])
        model = models.get(selected_model, models["Random Forest"])
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
            'confidence': confidence,
            'model': selected_model
        }
        # Load data for template
        data_file = os.path.join("data", "nasa_data.json")
        if os.path.exists(data_file):
            with open(data_file, "r") as f:
                data = json.load(f)
        else:
            data = {"near_earth_objects": {}}
        return render_template("index.html", prediction=prediction_label, confidence=confidence, data=data, importances=load_feature_importances(), selected_model=selected_model)
    except:
        # Also pass data in error case
        data_file = os.path.join("data", "nasa_data.json")
        if os.path.exists(data_file):
            with open(data_file, "r") as f:
                data = json.load(f)
        else:
            data = {"near_earth_objects": {}}
        return render_template("index.html", prediction="error", confidence=None, data=data, importances=load_feature_importances(), selected_model="Random Forest")

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
