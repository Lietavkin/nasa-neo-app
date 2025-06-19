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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = Flask(__name__)
# Load all trained ML models
models = {
    "Random Forest": joblib.load("model_rf.pkl"),
    "Decision Tree": joblib.load("model_dt.pkl"),
    "Logistic Regression": joblib.load("model_lr.pkl"),
    "KNN": joblib.load("model_knn.pkl"),
    "Support Vector Machine": joblib.load("model_svm.pkl") if os.path.exists("model_svm.pkl") else None,
    "XGBoost": joblib.load("model_xgb.pkl") if os.path.exists("model_xgb.pkl") else None,
    "My Hazard Classifier – NASA Style": joblib.load("model_my_classifier.pkl") if os.path.exists("model_my_classifier.pkl") else None
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
    with open("static/model_comparison_results.json") as f:
        metrics = json.load(f)
    with open("static/model_confusion_matrices.json") as f:
        confusion_matrices = json.load(f)
    def confusion_html(cm, model_name=None):
        label = f'<h3 class="cm-model-label">{model_name}</h3>' if model_name else ''
        return f'''{label}<table class="confusion-matrix-table"><tr><th></th><th>Pred 0</th><th>Pred 1</th></tr><tr><th>Actual 0</th><td>{cm[0][0]}</td><td>{cm[0][1]}</td></tr><tr><th>Actual 1</th><td>{cm[1][0]}</td><td>{cm[1][1]}</td></tr></table>'''
    rf_confusion_html = confusion_html(confusion_matrices["Random Forest"], "Random Forest")
    dt_confusion_html = confusion_html(confusion_matrices["Decision Tree"], "Decision Tree")
    lr_confusion_html = confusion_html(confusion_matrices["Logistic Regression"], "Logistic Regression")
    knn_confusion_html = confusion_html(confusion_matrices["KNN"], "K-Nearest Neighbors")
    svm_confusion_html = confusion_html(confusion_matrices["Support Vector Machine"], "Support Vector Machine") if "Support Vector Machine" in confusion_matrices else ""
    xgb_confusion_html = confusion_html(confusion_matrices["XGBoost"], "XGBoost") if "XGBoost" in confusion_matrices else ""
    myclf_confusion_html = confusion_html(confusion_matrices["My Hazard Classifier – NASA Style"], "My Hazard Classifier – NASA Style") if "My Hazard Classifier – NASA Style" in confusion_matrices else ""
    def metrics_html(metrics, key):
        m = metrics[key]
        return f'''<table class="metrics-table"><tr><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr><tr><td>{m['accuracy']:.2f}</td><td>{m['precision']:.2f}</td><td>{m['recall']:.2f}</td><td>{m['f1']:.2f}</td></tr></table>'''
    rf_metrics_html = metrics_html(metrics, "Random Forest")
    dt_metrics_html = metrics_html(metrics, "Decision Tree")
    lr_metrics_html = metrics_html(metrics, "Logistic Regression")
    knn_metrics_html = metrics_html(metrics, "KNN")
    svm_metrics_html = metrics_html(metrics, "Support Vector Machine") if "Support Vector Machine" in metrics else ""
    xgb_metrics_html = metrics_html(metrics, "XGBoost") if "XGBoost" in metrics else ""
    myclf_metrics_html = metrics_html(metrics, "My Hazard Classifier – NASA Style") if "My Hazard Classifier – NASA Style" in metrics else ""
    return (rf_confusion_html, dt_confusion_html, lr_confusion_html, knn_confusion_html, svm_confusion_html, xgb_confusion_html, myclf_confusion_html,
            rf_metrics_html, dt_metrics_html, lr_metrics_html, knn_metrics_html, svm_metrics_html, xgb_metrics_html, myclf_metrics_html)

def load_corr_heatmap_explanation():
    path = os.path.join("static", "eda_corr_heatmap_explanation.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return "This heatmap shows the pairwise correlations between input features. Brighter red or blue indicates stronger positive or negative correlation, respectively. Near-zero values mean the features are largely independent."

@app.route("/")
def index():
    data_file = os.path.join("data", "nasa_data.json")
    data = {}
    if os.path.exists(data_file):
        with open(data_file, "r") as f:
            data = json.load(f)
    importances = load_feature_importances()
    # Always reload latest confusion matrices and metrics
    rf_confusion_html, dt_confusion_html, lr_confusion_html, knn_confusion_html, svm_confusion_html, xgb_confusion_html, myclf_confusion_html, rf_metrics_html, dt_metrics_html, lr_metrics_html, knn_metrics_html, svm_metrics_html, xgb_metrics_html, myclf_metrics_html = get_model_comparison_data()
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
        knn_confusion_html=knn_confusion_html,
        svm_confusion_html=svm_confusion_html,
        xgb_confusion_html=xgb_confusion_html,
        myclf_confusion_html=myclf_confusion_html,
        rf_metrics_html=rf_metrics_html,
        dt_metrics_html=dt_metrics_html,
        lr_metrics_html=lr_metrics_html,
        knn_metrics_html=knn_metrics_html,
        svm_metrics_html=svm_metrics_html,
        xgb_metrics_html=xgb_metrics_html,
        myclf_metrics_html=myclf_metrics_html,
        model_metrics=model_metrics,
        corr_heatmap_explanation=load_corr_heatmap_explanation(),
        feature_importance_img='feature_importance_rf.png'
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
        # Extract all 8 features in the exact order expected by the model
        # feature_cols = ["absolute_magnitude_h", "diameter_max", "diameter_min", "velocity_kph", "velocity_kps", "distance_km", "distance_lunar", "distance_au"]
        absolute_magnitude_h = float(request.form["absolute_magnitude_h"])
        diameter_max = float(request.form["estimated_diameter_max"])
        diameter_min = float(request.form["estimated_diameter_min"])
        velocity_kph = float(request.form["velocity_kph"])
        velocity_kps = float(request.form["velocity_kps"])
        distance_km = float(request.form["distance_km"])
        distance_lunar = float(request.form["distance_lunar"])
        distance_au = float(request.form["distance_au"])
        
        selected_model = request.form.get("model_choice", "Random Forest")
        
        # Create input DataFrame with features in the exact order expected by the model
        feature_cols = ["absolute_magnitude_h", "diameter_max", "diameter_min", "velocity_kph", "velocity_kps", "distance_km", "distance_lunar", "distance_au"]
        input_data = [[absolute_magnitude_h, diameter_max, diameter_min, velocity_kph, velocity_kps, distance_km, distance_lunar, distance_au]]
        input_df = pd.DataFrame(input_data, columns=feature_cols)
        
        model = models.get(selected_model, models["Random Forest"])
        proba = model.predict_proba(input_df)[0]
        prediction = model.predict(input_df)[0]
        prediction_label = "Yes" if prediction == 1 else "No"
        confidence = proba[1] if prediction == 1 else proba[0]
        
        # Store for report
        global last_prediction
        last_prediction = {
            'absolute_magnitude_h': absolute_magnitude_h,
            'diameter_max': diameter_max,
            'diameter_min': diameter_min,
            'velocity_kph': velocity_kph,
            'velocity_kps': velocity_kps,
            'distance_km': distance_km,
            'distance_lunar': distance_lunar,
            'distance_au': distance_au,
            'prediction_label': prediction_label,
            'confidence': confidence,
            'model': selected_model
        }
        
        # Always return JSON for AJAX requests (fetch API)
        # Since we're using fetch(), we'll always return JSON and let the frontend handle it
        return jsonify({
            'prediction': prediction_label,
            'confidence': confidence,
            'model': selected_model,
            'success': True
        })
    except Exception as e:
        # Always return JSON for errors too
        return jsonify({
            'error': 'Prediction failed. Please check your inputs.',
            'success': False
        }), 400

@app.route("/download_report")
def download_report():
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(72, 720, "Asteroid Hazard Prediction Report")
    p.setFont("Helvetica", 12)
    y = 680
    
    # All 8 features
    p.drawString(72, y, f"Absolute Magnitude H: {last_prediction['absolute_magnitude_h']}")
    y -= 20
    p.drawString(72, y, f"Diameter Min (m): {last_prediction['diameter_min']}")
    y -= 20
    p.drawString(72, y, f"Diameter Max (m): {last_prediction['diameter_max']}")
    y -= 20
    p.drawString(72, y, f"Velocity (km/h): {last_prediction['velocity_kph']}")
    y -= 20
    p.drawString(72, y, f"Velocity (km/s): {last_prediction['velocity_kps']}")
    y -= 20
    p.drawString(72, y, f"Distance (km): {last_prediction['distance_km']}")
    y -= 20
    p.drawString(72, y, f"Distance (Lunar): {last_prediction['distance_lunar']}")
    y -= 20
    p.drawString(72, y, f"Distance (AU): {last_prediction['distance_au']}")
    y -= 30
    
    p.setFont("Helvetica-Bold", 14)
    p.drawString(72, y, f"Prediction: {last_prediction['prediction_label']}")
    y -= 24
    if last_prediction['confidence'] is not None:
        p.drawString(72, y, f"Confidence: {last_prediction['confidence']*100:.1f}%")
    y -= 24
    p.setFont("Helvetica", 10)
    p.drawString(72, y, f"Model Used: {last_prediction['model']}")
    
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

@app.route("/today")
def show_today_predictions():
    import pandas as pd
    try:
        df = pd.read_csv("neo_today_predictions.csv")
    except FileNotFoundError:
        return "❌ Today's prediction file not found. Please run the prediction script first."
    return render_template(
        "today.html",
        tables=[df.to_html(classes='data table table-bordered table-hover', index=False)],
        title="Today's NEO Predictions"
    )

@app.route("/api/feature_importance/<model_key>")
def api_feature_importance(model_key):
    # Map model_key to file
    fname = f"data/feature_importance_{model_key}.json"
    if os.path.exists(fname):
        with open(fname, "r") as f:
            return jsonify(json.load(f))
    # fallback to rf
    fallback = "data/feature_importance_rf.json"
    if os.path.exists(fallback):
        with open(fallback, "r") as f:
            return jsonify(json.load(f))
    return jsonify({})

@app.route("/api/daily_predictions/<model_key>")
def api_daily_predictions(model_key):
    fname = f"data/daily_predictions_{model_key}.json"
    if os.path.exists(fname):
        with open(fname, "r") as f:
            return jsonify(json.load(f))
    fallback = "data/daily_predictions_rf.json"
    if os.path.exists(fallback):
        with open(fallback, "r") as f:
            return jsonify(json.load(f))
    return jsonify([])

@app.route("/api/pca_projection/<model_key>")
def api_pca_projection(model_key):
    # Serve the precomputed PCA JSON for the selected model
    pca_json_path = os.path.join("data", f"pca_{model_key}.json")
    if not os.path.exists(pca_json_path):
        return jsonify([])
    with open(pca_json_path, "r") as f:
        points = json.load(f)
    return jsonify(points)

@app.route("/api/correlation/<model_key>")
def api_correlation(model_key):
    import pandas as pd
    import numpy as np
    import json
    import os
    
    # Load daily predictions for the model to get real data
    fname = f"data/daily_predictions_{model_key}.json"
    if not os.path.exists(fname):
        fname = "data/daily_predictions_rf.json"  # fallback
    
    if not os.path.exists(fname):
        return jsonify({"error": "No data available"})
    
    with open(fname, "r") as f:
        predictions_data = json.load(f)
    
    if not predictions_data:
        return jsonify({"error": "No prediction data available"})
    
    # Convert to DataFrame for correlation analysis
    df = pd.DataFrame(predictions_data)
    
    # Feature columns for correlation (both possible names)
    feature_cols = [
        "absolute_magnitude_h", "estimated_diameter_min", "estimated_diameter_max",
        "velocity_kph", "velocity_kps", "distance_km", "distance_lunar", "distance_au",
        "diameter_min", "diameter_max"  # alternative names
    ]
    
    # Filter to only available columns
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if len(available_cols) < 2:
        return jsonify({"error": "Insufficient data for correlation analysis"})
    
    # Load feature importance to get top features
    imp_fname = f"data/feature_importance_{model_key}.json"
    top_features = available_cols[:6]  # default to first 6 available
    
    if os.path.exists(imp_fname):
        with open(imp_fname, "r") as f:
            importances = json.load(f)
        # Get top 6 features by importance, matching to available columns
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        # Map feature importance names to available column names
        feature_mapping = {
            'diameter_min': 'estimated_diameter_min',
            'diameter_max': 'estimated_diameter_max'
        }
        
        top_features = []
        for feat, _ in sorted_features:
            # Check direct match first
            if feat in available_cols:
                top_features.append(feat)
            # Check mapped name
            elif feat in feature_mapping and feature_mapping[feat] in available_cols:
                top_features.append(feature_mapping[feat])
            # Check reverse mapping
            elif feat in feature_mapping.values():
                reverse_key = [k for k, v in feature_mapping.items() if v == feat]
                if reverse_key and reverse_key[0] in available_cols:
                    top_features.append(reverse_key[0])
        
        # Limit to top 6 and ensure we have enough
        top_features = top_features[:6]
        if len(top_features) < 3:  # fallback if too few features
            top_features = available_cols[:6]
    
    # Compute correlation matrix for top features
    try:
        corr_matrix = df[top_features].corr()
        
        # Convert to format suitable for heatmap
        corr_data = {
            "features": top_features,
            "matrix": corr_matrix.values.tolist(),
            "model": model_key.upper()
        }
        
        return jsonify(corr_data)
    except Exception as e:
        return jsonify({"error": f"Correlation computation failed: {str(e)}"})

if __name__ == '__main__':
    app.run()
