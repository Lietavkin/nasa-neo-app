import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import os

def save_confusion_matrix(y_test, y_pred):
    plt.figure(figsize=(5, 4), facecolor='#101c2b')
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/confusion_matrix.png", facecolor='#101c2b')
    plt.close()

def save_feature_importance(model, features):
    importances = model.feature_importances_
    fig = plt.figure(figsize=(10, 5), facecolor='#0b1222', edgecolor='#0b1222')
    ax = fig.gca()
    bars = ax.barh(features, importances, color="#00eaff", edgecolor="#7ecbff", linewidth=3, zorder=3)
    ax.set_xlabel("Feature Importance", fontsize=18, color="white", labelpad=14)
    ax.set_title("Model Feature Importances", fontsize=22, color="white", pad=22, weight='bold')
    ax.tick_params(axis='x', colors='white', labelsize=16)
    ax.tick_params(axis='y', colors='white', labelsize=18)
    ax.set_facecolor('#0b1222')
    for spine in ax.spines.values():
        spine.set_visible(False)
        spine.set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.xaxis.label.set_weight('bold')
    ax.yaxis.label.set_weight('bold')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    # Add grid
    ax.xaxis.grid(True, color='#0ff1ff33', linestyle='--', linewidth=1.2, zorder=0)
    ax.yaxis.grid(False)
    # Add value labels with glow
    for bar in bars:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.3f}",
                va='center', ha='left', fontsize=18, color='#0ff1ff', fontweight='bold',
                path_effects=[
                    __import__('matplotlib.patheffects').patheffects.withStroke(linewidth=4, foreground='#0b1222')
                ])
    plt.tight_layout(pad=0.5)
    plt.savefig("static/feature_importance.png", facecolor='#0b1222', edgecolor='#0b1222', bbox_inches='tight', dpi=180, transparent=False)
    plt.close()
    # Save feature importances to data/feature_importance.json for web app
    feature_names = features
    importances_list = importances.tolist()
    importance_dict = dict(zip(feature_names, importances_list))
    os.makedirs("data", exist_ok=True)
    with open("data/feature_importance.json", "w") as f:
        json.dump(importance_dict, f, indent=2)
    print("✅ Feature importances saved to data/feature_importance.json")

def train_model():
    # Load data
    with open("data/nasa_data.json", "r") as f:
        raw_data = json.load(f)
    asteroids = []
    for date in raw_data["near_earth_objects"]:
        for a in raw_data["near_earth_objects"][date]:
            try:
                asteroids.append({
                    "diameter": a["estimated_diameter"]["meters"]["estimated_diameter_max"],
                    "velocity": float(a["close_approach_data"][0]["relative_velocity"]["kilometers_per_hour"]),
                    "distance": float(a["close_approach_data"][0]["miss_distance"]["kilometers"]),
                    "hazardous": int(a["is_potentially_hazardous_asteroid"])
                })
            except:
                continue
    df = pd.DataFrame(asteroids)
    X = df[["diameter", "velocity", "distance"]]
    y = df["hazardous"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    save_confusion_matrix(y_test, y_pred)
    save_feature_importance(model, ["diameter", "velocity", "distance"])
    joblib.dump(model, "model.pkl")
    print("✅ Model + images + feature importances saved.")

if __name__ == "__main__":
    train_model()

