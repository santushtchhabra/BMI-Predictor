# train_models.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns   # optional, nice heatmaps

# --- Config ---
DATA_PATH = r"D:\Minor project\500_Person_Gender_Height_Weight_Index.csv"
OUT_DIR = "static/images"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Load dataset ---
df = pd.read_csv(DATA_PATH)

# If BMI column not present, compute it:
if "BMI" in df.columns:
    df["BMI_value"] = df["BMI"]
else:
    if {"Height", "Weight"}.issubset(df.columns):
        df["BMI_value"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    else:
        raise ValueError("Dataset missing BMI and/or Height/Weight columns. Check CSV.")

# Ensure gender numeric: map Male->1, Female->0
if df["Gender"].dtype == object:
    df["Gender_num"] = df["Gender"].map({"Male": 1, "male": 1, "Female": 0, "female": 0})
else:
    df["Gender_num"] = df["Gender"]

# Create BMI category labels
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 24.9:
        return "Normal"
    elif bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

df["BMI_cat"] = df["BMI_value"].apply(bmi_category)

# Feature matrix and targets
X = df[["Gender_num", "Height", "Weight"]].values
y_reg = df["BMI_value"].values
y_clf = df["BMI_cat"].values

# Train/test split
X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# --- Regressor: Decision Tree ---
dt_reg = DecisionTreeRegressor(random_state=42, max_depth=6)
dt_reg.fit(X_train, y_train_reg)
y_pred_reg = dt_reg.predict(X_test)

# Metrics for regressor
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"Regressor MSE: {mse:.4f}, R2: {r2:.4f}")

# Save regressor
joblib.dump(dt_reg, "models/dt_regressor.pkl")
print("✅ Saved models/dt_regressor.pkl")

# Plot: predicted vs actual (regression)
plt.figure(figsize=(6,5))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.7)
plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], linestyle="--")
plt.xlabel("Actual BMI")
plt.ylabel("Predicted BMI (DecisionTree)")
plt.title("Predicted vs Actual BMI")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pred_vs_actual_bmi.png"))
plt.close()

# --- Classifier: Decision Tree ---
dt_clf = DecisionTreeClassifier(random_state=42, max_depth=6)
dt_clf.fit(X_train, y_train_clf)
y_pred_clf = dt_clf.predict(X_test)

# Save classifier
joblib.dump(dt_clf, "models/dt_classifier.pkl")
print("✅ Saved models/dt_classifier.pkl")

# Classification report & confusion matrix
print("Classification report:\n", classification_report(y_test_clf, y_pred_clf))
cm = confusion_matrix(y_test_clf, y_pred_clf, labels=["Underweight","Normal","Overweight","Obese"])

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Under","Normal","Over","Obese"],
            yticklabels=["Under","Normal","Over","Obese"],
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (DecisionTreeClassifier)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
plt.close()

# --- Feature Importances ---
importances_reg = getattr(dt_reg, "feature_importances_", None)
importances_clf = getattr(dt_clf, "feature_importances_", None)
feature_names = ["Gender", "Height", "Weight"]

def plot_feature_importances(importances, filename, title):
    if importances is None:
        return
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(6,4))
    plt.bar([feature_names[i] for i in idx], importances[idx])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename))
    plt.close()

plot_feature_importances(importances_reg, "feat_imp_regressor.png", "Regressor Feature Importances")
plot_feature_importances(importances_clf, "feat_imp_classifier.png", "Classifier Feature Importances")

print("✅ All plots saved to", OUT_DIR)
