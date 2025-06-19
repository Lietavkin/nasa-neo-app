import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

# Load cleaned data
df = pd.read_csv("neo_clean.csv")

# Features and target
X = df.drop(columns=["is_hazardous"])
y = df["is_hazardous"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models to train
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# Evaluate all models using cross-validation (F1 score)
best_model = None
best_score = -1
best_name = None
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='f1')
    mean_score = scores.mean()
    print(f"{name}: Mean F1 (5-fold CV): {mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_model = model
        best_name = name

# Retrain best model on full training data and evaluate on test set
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print(f"\nBest Model: {best_name}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the best model
joblib.dump(best_model, "model_my_classifier.pkl")
print("✅ Best model saved as model_my_classifier.pkl") 