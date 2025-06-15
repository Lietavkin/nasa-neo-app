# NASA NEO Dashboard 🚀

A modern, interactive dashboard for tracking and analyzing Near-Earth Objects (asteroids) using NASA's public API, machine learning, and beautiful data visualizations.

---

## 🌌 Project Overview

This dashboard fetches the latest asteroid data from NASA's Near Earth Object Web Service (NeoWs), analyzes the risk of each asteroid using a machine learning model, and presents the results in a stunning, recruiter-ready web app. Users can:
- Predict asteroid hazard based on custom input
- Explore live asteroid data and risk scores
- Visualize model performance and feature importances
- Download a PDF report of their prediction
- See all visuals in a seamless, futuristic NASA-inspired UI

---

## ✨ Features
- **Live Asteroid Feed:** Pulls the latest 7 days of NEO data from NASA.
- **ML Hazard Prediction:** Predicts if an asteroid is hazardous using a trained Random Forest model.
- **Dynamic Risk Scoring:** Calculates a precise risk percentage for each asteroid based on its size and distance.
- **Interactive Visualizations:** Bar charts, scatter plots, and confusion matrix using Chart.js and Matplotlib.
- **PDF Report:** Download a summary of your prediction.
- **Automatic Retraining:** Model and data update weekly in the background.
- **Responsive, Futuristic UI:** Looks great on desktop and mobile.

---

## 🛠️ Technologies Used
- **Flask** (Python web framework)
- **scikit-learn** (Machine learning)
- **pandas, matplotlib, seaborn** (Data analysis & plotting)
- **Chart.js** (Frontend charts)
- **APScheduler** (Background jobs)
- **ReportLab** (PDF generation)
- **NASA NeoWs API** (Asteroid data)
- **HTML/CSS (Inter font, custom dark theme)**

---

## 🚀 How to Run Locally

1. **Clone the repo:**
   ```bash
   git clone https://github.com/yourusername/nasa-neo-dashboard.git
   cd nasa-neo-dashboard
   ```
2. **Create a virtual environment & activate:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the app:**
   ```bash
   python app.py
   ```
5. **Visit:** [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🖼️ Screenshots

> _Replace these with your own!_

| Dashboard Home | Prediction Example |
|---------------|-------------------|
| ![Dashboard](screenshots/dashboard.png) | ![Prediction](screenshots/predict.png) |

---

## 🌍 Live Demo

> _Add your deployed link here!_

[https://your-live-app-url.com](https://your-live-app-url.com)

---

## 📖 How It Works & How to Use

### What does this app do?
- **Fetches asteroid data** from NASA for the current week.
- **Trains a machine learning model** to predict if an asteroid is hazardous.
- **Calculates a risk score** for each asteroid based on its size and how close it will come to Earth.
- **Lets you enter your own asteroid parameters** (diameter, velocity, distance) and get a prediction and confidence score.
- **Visualizes** all data and model results in a beautiful dashboard.

### How to use it
1. **See the latest asteroids** in the table and charts.
2. **Enter diameter, velocity, and distance** in the form to predict hazard risk.
3. **Download a PDF report** of your prediction.
4. **Retrain the model** with the latest data (admin/advanced users).

### What do the risk scores mean?
- **Risk %** is calculated for each asteroid using:
  - **Hazardous asteroids:**
    ```
    risk = min(99, max(10, int((diameter * 1000000 / distance) * 2)))
    ```
  - **Non-hazardous asteroids:**
    ```
    risk = min(10, max(1, int(diameter * 100000 / distance)))
    ```
  - Where:
    - `diameter` = max estimated diameter in meters
    - `distance` = closest approach in kilometers
- **Interpretation:**
  - **🔴 High Risk:** Large, hazardous, and very close asteroids (risk > 70%)
  - **🟠 Moderate Risk:** Hazardous but not extremely close (risk 30-70%)
  - **🟢 Low Risk:** Non-hazardous or distant asteroids (risk < 10%)

### How does the ML model work?
- **Features:** Diameter, velocity, and distance.
- **Model:** Random Forest Classifier (scikit-learn)
- **Target:** NASA's `is_potentially_hazardous_asteroid` label
- **Retraining:** The model updates weekly with new NASA data.

---

## 🤔 Why this project?
- Showcases real-world data science, ML, and web dev skills.
- Beautiful, recruiter-ready UI.
- End-to-end: data ingestion, ML, visualization, and reporting.
- Built with production best practices (background jobs, PDF export, responsive design).

---

## 📬 Questions or Feedback?
Open an issue or PR, or reach out on [LinkedIn](https://www.linkedin.com/). 