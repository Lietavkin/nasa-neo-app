import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

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
    # Sort features by importance
    sorted_idx = importances.argsort()[::-1]
    sorted_importances = importances[sorted_idx]
    sorted_features = [features[i] for i in sorted_idx]
    # Highlight top 3
    colors = ["#ffb300" if i < 3 else "#00eaff" for i in range(len(features))]
    edge_colors = ["#ffd700" if i < 3 else "#7ecbff" for i in range(len(features))]
    fig = plt.figure(figsize=(11, 6), facecolor='none')
    ax = fig.gca()
    bars = ax.barh(sorted_features, sorted_importances, color=colors, edgecolor=edge_colors, linewidth=3, zorder=3)
    ax.set_xlabel("Importance Score", fontsize=18, color="white", labelpad=14)
    ax.set_ylabel("Feature Names", fontsize=18, color="white", labelpad=14)
    ax.set_title("Feature Importances (Top 3 Highlighted)", fontsize=22, color="white", pad=22, weight='bold')
    ax.tick_params(axis='x', colors='white', labelsize=16)
    ax.tick_params(axis='y', colors='white', labelsize=18)
    ax.set_facecolor('none')
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
                    __import__('matplotlib.patheffects').patheffects.withStroke(linewidth=4, foreground='#101c2b')
                ])
    plt.tight_layout(pad=0.5)
    plt.savefig("static/feature_importance.png", transparent=True, bbox_inches='tight', dpi=180)
    plt.close()
    # Save feature importances to data/feature_importance.json for web app
    feature_names = features
    importances_list = importances.tolist()
    importance_dict = dict(zip(feature_names, importances_list))
    os.makedirs("data", exist_ok=True)
    with open("data/feature_importance.json", "w") as f:
        json.dump(importance_dict, f, indent=2)
    print("✅ Feature importances saved to data/feature_importance.json")

def save_eda_plots(df):
    os.makedirs("static", exist_ok=True)
    # 1. Histogram of hazardous vs non-hazardous
    plt.figure(figsize=(7,5), facecolor='#101c2b')
    ax = sns.countplot(x='hazardous', data=df, palette=['#00eaff', '#ff4c4c'])
    ax.set_xticklabels(['Non-Hazardous', 'Hazardous'], color='white', fontsize=16)
    ax.set_xlabel('Asteroid Type', color='white', fontsize=18)
    ax.set_ylabel('Count', color='white', fontsize=18)
    ax.set_title('Asteroid Hazardous vs Non-Hazardous', color='white', fontsize=20, pad=16)
    ax.set_facecolor('#101c2b')
    ax.tick_params(axis='y', colors='white', labelsize=15)
    for spine in ax.spines.values():
        spine.set_color('white')
    plt.tight_layout()
    plt.savefig('static/eda_target_hist.png', facecolor='#101c2b')
    plt.close()
    # 2. Distribution of asteroid diameters
    plt.figure(figsize=(8,5), facecolor='#101c2b')
    ax = sns.histplot(df['diameter_max'], bins=30, color='#00eaff', edgecolor='#7ecbff', kde=True)
    ax.set_xlabel('Diameter (meters)', color='white', fontsize=18)
    ax.set_ylabel('Count', color='white', fontsize=18)
    ax.set_title('Distribution of Asteroid Diameters', color='white', fontsize=20, pad=16)
    ax.set_facecolor('#101c2b')
    ax.tick_params(axis='x', colors='white', labelsize=15)
    ax.tick_params(axis='y', colors='white', labelsize=15)
    for spine in ax.spines.values():
        spine.set_color('white')
    plt.tight_layout()
    plt.savefig('static/eda_diameter_hist.png', facecolor='#101c2b')
    plt.close()
    # 3. Correlation heatmap
    plt.figure(figsize=(7,6), facecolor='#101c2b')
    corr = df[['diameter_max', 'velocity_kph', 'distance_km', 'hazardous']].corr()
    ax = sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1.5, linecolor='#223a5e',
                     cbar_kws={'label': 'Correlation'}, annot_kws={'color':'white','fontsize':14})
    ax.set_title('Correlation Heatmap of Features', color='white', fontsize=20, pad=16)
    ax.set_facecolor('#101c2b')
    ax.tick_params(axis='x', colors='white', labelsize=15)
    ax.tick_params(axis='y', colors='white', labelsize=15)
    for spine in ax.spines.values():
        spine.set_color('white')
    plt.tight_layout()
    plt.savefig('static/eda_corr_heatmap.png', facecolor='#101c2b')
    plt.close()

def save_pca_projection(df, feature_cols):
    os.makedirs("static", exist_ok=True)
    scaler = StandardScaler()
    X = df[feature_cols].values
    y = df["hazardous"].values
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    # Plot
    plt.figure(figsize=(14, 6), facecolor='none')
    # Original feature space (first 2 features)
    ax1 = plt.subplot(1, 2, 1)
    for label, color, name in zip([0, 1], ['#00eaff', '#ff4c4c'], ['Non-Hazardous', 'Hazardous']):
        idx = (y == label)
        ax1.scatter(X[idx, 0], X[idx, 1], c=color, label=name, alpha=0.7, edgecolors='w', s=60)
    ax1.set_title('Original Feature Space', color='white', fontsize=18, pad=14)
    ax1.set_xlabel(feature_cols[0], color='white', fontsize=15)
    ax1.set_ylabel(feature_cols[1], color='white', fontsize=15)
    ax1.tick_params(colors='white', labelsize=13)
    ax1.set_facecolor('none')
    for spine in ax1.spines.values():
        spine.set_visible(False)
    # Custom legend with colored text
    handles = [mpatches.Patch(color='#00eaff', label='Non-Hazardous'),
               mpatches.Patch(color='#ff4c4c', label='Hazardous')]
    legend1 = ax1.legend(handles=handles, fontsize=12, loc='best', frameon=True, facecolor='#101c2b', edgecolor='#7ecbff')
    for text, color in zip(legend1.get_texts(), ['#00eaff', '#ff4c4c']):
        text.set_color(color)
    # PCA-reduced space
    ax2 = plt.subplot(1, 2, 2)
    for label, color, name in zip([0, 1], ['#00eaff', '#ff4c4c'], ['Non-Hazardous', 'Hazardous']):
        idx = (y == label)
        ax2.scatter(X_pca[idx, 0], X_pca[idx, 1], c=color, label=name, alpha=0.7, edgecolors='w', s=60)
    ax2.set_title('PCA-Reduced Space', color='white', fontsize=18, pad=14)
    ax2.set_xlabel('PC1', color='white', fontsize=15)
    ax2.set_ylabel('PC2', color='white', fontsize=15)
    ax2.tick_params(colors='white', labelsize=13)
    ax2.set_facecolor('none')
    for spine in ax2.spines.values():
        spine.set_visible(False)
    handles2 = [mpatches.Patch(color='#00eaff', label='Non-Hazardous'),
                mpatches.Patch(color='#ff4c4c', label='Hazardous')]
    legend2 = ax2.legend(handles=handles2, fontsize=12, loc='best', frameon=True, facecolor='#101c2b', edgecolor='#7ecbff')
    for text, color in zip(legend2.get_texts(), ['#00eaff', '#ff4c4c']):
        text.set_color(color)
    plt.tight_layout(pad=2.0)
    plt.savefig('static/pca_projection.png', transparent=True, bbox_inches='tight', dpi=180)
    plt.close()

def save_model_comparison(X, y):
    os.makedirs("static", exist_ok=True)
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }
    results = {}
    confusion_matrices = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        accs, precs, recs, f1s = [], [], [], []
        cm_sum = None
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accs.append(accuracy_score(y_test, y_pred))
            precs.append(precision_score(y_test, y_pred, zero_division=0))
            recs.append(recall_score(y_test, y_pred, zero_division=0))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))
            cm = confusion_matrix(y_test, y_pred)
            if cm_sum is None:
                cm_sum = cm
            else:
                cm_sum += cm
        results[name] = {
            "accuracy": sum(accs) / len(accs),
            "precision": sum(precs) / len(precs),
            "recall": sum(recs) / len(recs),
            "f1": sum(f1s) / len(f1s)
        }
        confusion_matrices[name] = cm_sum.tolist()
    # Save metrics
    import json
    with open("static/model_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("static/model_confusion_matrices.json", "w") as f:
        json.dump(confusion_matrices, f, indent=2)

def train_model():
    # Load data
    with open("data/nasa_data.json", "r") as f:
        raw_data = json.load(f)
    asteroids = []
    for date in raw_data["near_earth_objects"]:
        for a in raw_data["near_earth_objects"][date]:
            try:
                asteroids.append({
                    "absolute_magnitude_h": a["absolute_magnitude_h"],
                    "diameter_max": a["estimated_diameter"]["meters"]["estimated_diameter_max"],
                    "diameter_min": a["estimated_diameter"]["meters"]["estimated_diameter_min"],
                    "velocity_kph": float(a["close_approach_data"][0]["relative_velocity"]["kilometers_per_hour"]),
                    "velocity_kps": float(a["close_approach_data"][0]["relative_velocity"]["kilometers_per_second"]),
                    "distance_km": float(a["close_approach_data"][0]["miss_distance"]["kilometers"]),
                    "distance_lunar": float(a["close_approach_data"][0]["miss_distance"]["lunar"]),
                    "distance_au": float(a["close_approach_data"][0]["miss_distance"]["astronomical"]),
                    "hazardous": int(a["is_potentially_hazardous_asteroid"])
                })
            except:
                continue
    df = pd.DataFrame(asteroids)
    save_eda_plots(df)
    feature_cols = [
        "absolute_magnitude_h", "diameter_max", "diameter_min",
        "velocity_kph", "velocity_kps",
        "distance_km", "distance_lunar", "distance_au"
    ]
    X = df[feature_cols]
    y = df["hazardous"]
    save_model_comparison(X, y)
    # For prediction, fit on all data and save models
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X, y)
    joblib.dump(rf_model, "model_rf.pkl")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X, y)
    joblib.dump(dt_model, "model_dt.pkl")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X, y)
    joblib.dump(lr_model, "model_lr.pkl")
    # Use Random Forest for visuals/feature importance by default
    y_pred = rf_model.predict(X)
    print("Classification Report:\n", classification_report(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
    save_confusion_matrix(y, y_pred)
    save_feature_importance(rf_model, feature_cols)
    save_pca_projection(df, feature_cols)
    print("✅ All models saved as model_rf.pkl, model_dt.pkl, model_lr.pkl. Model + images + feature importances saved.")

if __name__ == "__main__":
    train_model()

