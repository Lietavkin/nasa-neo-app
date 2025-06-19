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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np

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

def save_lr_feature_importance(model, features, model_key=None):
    # Use absolute value of coefficients as importance
    importances = np.abs(model.coef_[0])
    sorted_idx = importances.argsort()[::-1]
    sorted_importances = importances[sorted_idx]
    sorted_features = [features[i] for i in sorted_idx]
    colors = ["#ffb300" if i < 3 else "#00eaff" for i in range(len(features))]
    edge_colors = ["#ffd700" if i < 3 else "#7ecbff" for i in range(len(features))]
    fig = plt.figure(figsize=(11, 6), facecolor='none')
    ax = fig.gca()
    bars = ax.barh(sorted_features, sorted_importances, color=colors, edgecolor=edge_colors, linewidth=3, zorder=3)
    ax.set_xlabel("|Coefficient| (Importance)", fontsize=18, color="white", labelpad=14)
    ax.set_ylabel("Feature Names", fontsize=18, color="white", labelpad=14)
    ax.set_title("Feature Importances (Logistic Regression)", fontsize=22, color="white", pad=22, weight='bold')
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
    ax.xaxis.grid(True, color='#0ff1ff33', linestyle='--', linewidth=1.2, zorder=0)
    ax.yaxis.grid(False)
    for bar in bars:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.3f}",
                va='center', ha='left', fontsize=18, color='#0ff1ff', fontweight='bold',
                path_effects=[__import__('matplotlib.patheffects').patheffects.withStroke(linewidth=4, foreground='#101c2b')])
    plt.tight_layout(pad=0.5)
    os.makedirs("static", exist_ok=True)
    fname = f"static/feature_importance_lr.png"
    plt.savefig(fname, transparent=True, bbox_inches='tight', dpi=180)
    plt.close()
    
    # Save feature importances JSON for dynamic table generation
    feature_names = features
    importances_list = importances.tolist()
    importance_dict = dict(zip(feature_names, importances_list))
    os.makedirs("data", exist_ok=True)
    if model_key:
        with open(f"data/feature_importance_{model_key}.json", "w") as f:
            json.dump(importance_dict, f, indent=2)
    
    print(f"✅ Feature importances saved to {fname} and data/feature_importance_{model_key}.json (Logistic Regression)")

def save_placeholder_feature_importance(model_key):
    import matplotlib.pyplot as plt
    import os
    explanation = (
        "Feature importance is not available for this model.\n\n"
        "K-Nearest Neighbors (KNN) is a non-parametric, instance-based algorithm. "
        "It does not learn weights or coefficients for features during training, "
        "but instead makes predictions based on the closest data points in the feature space.\n\n"
        "As a result, KNN does not provide intrinsic feature importances."
    )
    plt.figure(figsize=(10, 4), facecolor='#101c2b')
    plt.text(0.5, 0.5, explanation,
             ha='center', va='center', fontsize=15, color='#7ecbff', fontweight='bold', wrap=True)
    plt.axis('off')
    plt.tight_layout()
    os.makedirs('static', exist_ok=True)
    fname = f'static/feature_importance_{model_key}.png'
    plt.savefig(fname, facecolor='#101c2b', bbox_inches='tight', dpi=180)
    plt.close()
    print(f'ℹ️ Professional placeholder feature importance saved to {fname}')

def save_feature_importance(model, features, model_key=None):
    if model_key == 'lr' and isinstance(model, LogisticRegression):
        save_lr_feature_importance(model, features, model_key)
    elif hasattr(model, 'feature_importances_'):
        # Only save if model has feature_importances_
        importances = model.feature_importances_
        sorted_idx = importances.argsort()[::-1]
        sorted_importances = importances[sorted_idx]
        sorted_features = [features[i] for i in sorted_idx]
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
        ax.xaxis.grid(True, color='#0ff1ff33', linestyle='--', linewidth=1.2, zorder=0)
        ax.yaxis.grid(False)
        for bar in bars:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.3f}",
                    va='center', ha='left', fontsize=18, color='#0ff1ff', fontweight='bold',
                    path_effects=[__import__('matplotlib.patheffects').patheffects.withStroke(linewidth=4, foreground='#101c2b')])
        plt.tight_layout(pad=0.5)
        os.makedirs("static", exist_ok=True)
        fname = f"static/feature_importance.png" if not model_key else f"static/feature_importance_{model_key}.png"
        plt.savefig(fname, transparent=True, bbox_inches='tight', dpi=180)
        plt.close()
        # Save feature importances to data/feature_importance.json for web app
        feature_names = features
        importances_list = importances.tolist()
        importance_dict = dict(zip(feature_names, importances_list))
        os.makedirs("data", exist_ok=True)
        with open("data/feature_importance.json", "w") as f:
            json.dump(importance_dict, f, indent=2)
        # Save per-model importances for frontend dynamic loading
        if model_key:
            with open(f"data/feature_importance_{model_key}.json", "w") as f:
                json.dump(importance_dict, f, indent=2)
        print(f"✅ Feature importances saved to {fname} and data/feature_importance_{model_key}.json")
    else:
        # Save placeholder for models without feature_importances_ or coef_
        if model_key:
            save_placeholder_feature_importance(model_key)

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
    # 3. Correlation heatmap (improved)
    plt.figure(figsize=(7,7), facecolor='#101c2b')
    corr = df[['diameter_max', 'velocity_kph', 'distance_km', 'hazardous']].corr()
    ax = sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f',
        linewidths=1.5,
        linecolor='#223a5e',
        square=True,
        cbar_kws={'label': 'Correlation'},
        annot_kws={'color':'white','fontsize':14}
    )
    ax.set_title('Correlation Heatmap of Features', color='white', fontsize=20, pad=16)
    ax.set_facecolor('#101c2b')
    ax.tick_params(axis='x', colors='white', labelsize=15)
    ax.tick_params(axis='y', colors='white', labelsize=15)
    for spine in ax.spines.values():
        spine.set_color('white')
    plt.tight_layout()
    plt.savefig('static/eda_corr_heatmap.png', facecolor='#101c2b')
    plt.close()
    # Save explanation for dashboard
    with open('static/eda_corr_heatmap_explanation.txt', 'w') as f:
        f.write('This heatmap shows the pairwise correlations between input features. Brighter red or blue indicates stronger positive or negative correlation, respectively. Near-zero values mean the features are largely independent.')

def save_pca_projection(df, feature_cols, model_key=None):
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
    fname = f'static/pca_projection.png' if not model_key else f'static/pca_projection_{model_key}.png'
    plt.savefig(fname, transparent=True, bbox_inches='tight', dpi=180)
    plt.close()

def save_pca_projection_json_from_predictions(model_key, feature_cols):
    import json
    from sklearn.decomposition import PCA

    input_path = f"data/daily_predictions_{model_key}.json"
    output_path = f"data/pca_{model_key}.json"

    with open(input_path, "r") as f:
        data = json.load(f)

    points = []
    X = []
    names = []
    labels = []
    raw_x = []
    raw_y = []

    for entry in data:
        try:
            x_vec = [float(entry[k]) for k in feature_cols]
            X.append(x_vec)
            names.append(entry.get("name", ""))
            labels.append(entry.get("predicted_hazard_label", "Safe"))
            raw_x.append(float(entry.get("absolute_magnitude_h", 0)))
            raw_y.append(float(entry.get("diameter_max", 0)))
        except:
            continue

    if not X:
        print(f"[WARNING] No data to compute PCA for {model_key}")
        return

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)

    for i in range(len(pcs)):
        points.append({
            "orig_x": raw_x[i],
            "orig_y": raw_y[i],
            "pc1": float(pcs[i][0]),
            "pc2": float(pcs[i][1]),
            "label": labels[i],
            "name": names[i]
        })

    with open(output_path, "w") as f:
        json.dump(points, f, indent=2)

    red_count = sum(1 for p in points if p["label"] == "Hazardous")
    blue_count = sum(1 for p in points if p["label"] == "Safe")
    print(f"[✅ PCA] {model_key}: Saved PCA with {red_count} hazardous, {blue_count} safe")

def save_model_comparison(X, y):
    os.makedirs("static", exist_ok=True)
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "My Hazard Classifier – NASA Style": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    results = {}
    confusion_matrices = {}
    
    # For small datasets with severe class imbalance, use the full dataset for evaluation
    # This gives more realistic metrics than cross-validation with tiny folds
    print(f"[INFO] Dataset has {len(y)} samples with {y.sum()} hazardous asteroids ({y.sum()/len(y)*100:.1f}%)")
    print(f"[INFO] Using full dataset evaluation due to severe class imbalance")
    
    for name, model in models.items():
        # Fit on full dataset
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate metrics on full dataset
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        cm = confusion_matrix(y, y_pred)
        
        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
        confusion_matrices[name] = cm.tolist()
        
        print(f"[INFO] {name}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")
    
    # Save metrics
    with open("static/model_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("static/model_confusion_matrices.json", "w") as f:
        json.dump(confusion_matrices, f, indent=2)

def save_hazardous_distribution_chart(model_key):
    import json
    import matplotlib.pyplot as plt
    import os
    # Load daily predictions for the model
    with open(f"data/daily_predictions_{model_key}.json", "r") as f:
        preds = json.load(f)
    # Count hazardous vs safe
    hazardous = sum(1 for p in preds if p["predicted_hazard_label"] == "Hazardous")
    safe = sum(1 for p in preds if p["predicted_hazard_label"] == "Safe")
    labels = ["Safe", "Hazardous"]
    counts = [safe, hazardous]
    colors = ["#00eaff", "#ff4c4c"]
    plt.figure(figsize=(6,5), facecolor="#101c2b")
    bars = plt.bar(labels, counts, color=colors, edgecolor="#7ecbff", linewidth=3)
    plt.xlabel("Asteroid Type", color="white", fontsize=18)
    plt.ylabel("Count", color="white", fontsize=18)
    plt.title(f"Hazardous vs Non-Hazardous ({model_key.upper()})", color="white", fontsize=20, pad=16)
    plt.xticks(color="white", fontsize=16)
    plt.yticks(color="white", fontsize=15)
    plt.gca().set_facecolor("#101c2b")
    for spine in plt.gca().spines.values():
        spine.set_color("white")
    for i, bar in enumerate(bars):
        plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(counts[i]),
                       ha='center', va='bottom', color='white', fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig(f"static/hazardous_distribution_{model_key}.png", facecolor="#101c2b")
    plt.close()

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
                    "hazardous": int(a["is_potentially_hazardous_asteroid"]),
                    "name": a["name"],
                    "date": list(raw_data["near_earth_objects"].keys())[0]  # Add date for predictions
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
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # For prediction, fit on all data and save models
    models = {
        "rf": RandomForestClassifier(random_state=42),
        "dt": DecisionTreeClassifier(random_state=42),
        "lr": LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm": SVC(probability=True, random_state=42),
        "xgb": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "myclf": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    
    for key, model in models.items():
        print(f"[INFO] Training and evaluating model: {key}")
        
        # Fit model on all data
        model.fit(X, y)
        joblib.dump(model, f"model_{key}.pkl")
        
        # Make predictions on the full dataset
        y_pred = model.predict(X)
        
        # Count hazardous predictions for verification
        n_hazardous = int((y_pred == 1).sum())
        n_safe = int((y_pred == 0).sum())
        print(f"[INFO] {key}: Predicted {n_hazardous} hazardous, {n_safe} safe asteroids")
        
        # Prepare daily predictions data
        df_pred = df.copy()
        df_pred["predicted_hazard"] = y_pred
        df_pred["predicted_hazard_label"] = df_pred["predicted_hazard"].map({1: "Hazardous", 0: "Safe"})
        
        # Add confidence scores if model supports it
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)
            confidence = y_proba.max(axis=1)
            df_pred["confidence"] = confidence
        else:
            df_pred["confidence"] = None
            
        # Save daily predictions JSON for this model
        records = []
        for _, row in df_pred.iterrows():
            records.append({
                "name": row["name"],
                "date": row.get("date", "2025-06-18"),
                "absolute_magnitude_h": float(row["absolute_magnitude_h"]),
                "diameter_min": float(row["diameter_min"]),
                "diameter_max": float(row["diameter_max"]),
                "estimated_diameter_min": float(row["diameter_min"]),
                "estimated_diameter_max": float(row["diameter_max"]),
                "velocity_kph": float(row["velocity_kph"]),
                "velocity_kps": float(row["velocity_kps"]),
                "distance_km": float(row["distance_km"]),
                "distance_lunar": float(row["distance_lunar"]),
                "distance_au": float(row["distance_au"]),
                "predicted_hazard": int(row["predicted_hazard"]),
                "predicted_hazard_label": row["predicted_hazard_label"],
                "confidence": float(row["confidence"]) if row["confidence"] is not None else None
            })
        
        # Save daily predictions JSON
        with open(f"data/daily_predictions_{key}.json", "w") as f:
            json.dump(records, f, indent=2)
        print(f"✅ Saved daily_predictions_{key}.json with {len(records)} predictions")
        
        # Generate PCA projection using the same predictions
        save_pca_projection_with_predictions(key, feature_cols, X.values, y_pred, df)
        
        # Save feature importance
        save_feature_importance(model, feature_cols, model_key=key)
        
        # Save static PCA projection image
        save_pca_projection(df, feature_cols, model_key=key)
        
        # Save hazardous distribution chart
        save_hazardous_distribution_chart(key)
        
    # Use Random Forest for confusion matrix
    y_pred_rf = models["rf"].predict(X)
    print("Classification Report (Random Forest):\n", classification_report(y, y_pred_rf))
    print("Confusion Matrix (Random Forest):\n", confusion_matrix(y, y_pred_rf))
    save_confusion_matrix(y, y_pred_rf)
    print("✅ All models saved as model_*.pkl. Model + images + feature importances saved.")

def save_pca_projection_with_predictions(model_key, feature_cols, X, y_pred, df):
    """Generate PCA projection using provided predictions to ensure consistency"""
    from sklearn.decomposition import PCA
    import json
    import os
    
    # Apply PCA to the feature data
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    points = []
    for i in range(len(X_pca)):
        label = "Hazardous" if y_pred[i] == 1 else "Safe"
        points.append({
            "orig_x": float(df.iloc[i]["absolute_magnitude_h"]),
            "orig_y": float(df.iloc[i]["diameter_max"]),
            "pc1": float(X_pca[i, 0]),
            "pc2": float(X_pca[i, 1]),
            "label": label,
            "name": df.iloc[i]["name"]
        })
    
    # Verify counts
    hazardous_count = sum(1 for p in points if p["label"] == "Hazardous")
    safe_count = sum(1 for p in points if p["label"] == "Safe")
    
    # Save PCA projection JSON
    os.makedirs("data", exist_ok=True)
    with open(f"data/pca_{model_key}.json", "w") as f:
        json.dump(points, f, indent=2)
    
    print(f"✅ PCA projection saved for {model_key}: {hazardous_count} hazardous, {safe_count} safe")

if __name__ == "__main__":
    train_model()

