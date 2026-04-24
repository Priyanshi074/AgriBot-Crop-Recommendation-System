# ml_model/train_advanced_model.py

import pandas as pd
import numpy as np
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================================
# LOAD WEATHER DATASET
# ==========================================================

df = pd.read_csv("../dataset/complete_dataset.csv")


# ==========================================================
# CLEAN FUNCTIONS
# ==========================================================

def avg_temp(val):
    numbers = re.findall(r"\d+\.?\d*", str(val))  # ✅ FIXED (handles decimals)
    if len(numbers) >= 2:
        return (float(numbers[0]) + float(numbers[1])) / 2
    elif len(numbers) == 1:
        return float(numbers[0])
    return np.nan


def clean_numeric(val, default):
    numbers = re.findall(r"\d+\.?\d*", str(val))  # ✅ FIXED
    if len(numbers) > 0:
        return float(numbers[0])
    return default


# ==========================================================
# APPLY CLEANING
# ==========================================================

df["temperature"] = df["temperature"].apply(avg_temp)
df["humidity"] = df["humidity"].apply(lambda x: clean_numeric(x, 50))
df["rainfall"] = df["rainfall"].apply(lambda x: clean_numeric(x, 0))


# ==========================================================
# FIX MONTH COLUMN
# ==========================================================

df["month"] = df["month"].astype(str).str.strip().str[:3].str.lower()

month_map = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
}

df["month"] = df["month"].map(month_map)
df = df.dropna(subset=["month"])
df["month"] = df["month"].astype(int)


# ==========================================================
# NORMALIZE STATE
# ==========================================================

df["state"] = df["state"].astype(str).str.strip().str.lower()


# ==========================================================
# CLEAN NUMERIC
# ==========================================================

df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")

df = df.dropna()


# ==========================================================
# SAVE CLEANED DATASET
# ==========================================================

df = df[["state", "month", "temperature", "humidity", "rainfall"]]
df.to_csv("../dataset/cleaned_weather.csv", index=False)

print("✅ Clean dataset ready!")


# ==========================================================
# TRAIN CROP MODEL
# ==========================================================

data = pd.read_csv("../dataset/crop_data.csv")


# ---------------- CLEAN NUMERIC ----------------

numeric_cols = ["N","P","K","temperature","humidity","ph","rainfall"]

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna()


# ---------------- FEATURES ----------------

X = data.drop("label", axis=1)


# ---------------- LABEL ENCODING (VERY IMPORTANT) ----------------

le = LabelEncoder()
y = le.fit_transform(data["label"])


# ---------------- SCALING ----------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ---------------- SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ---------------- MODELS ----------------

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)


# ---------------- ENSEMBLE ----------------

model = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb)],
    voting="soft"
)


# ---------------- TRAIN ----------------

model.fit(X_train, y_train)


# ---------------- EVALUATE ----------------

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ==========================================================
# SAVE MODELS
# ==========================================================

joblib.dump(model, "../agri_project/crop_app/crop_model.pkl")
joblib.dump(scaler, "../agri_project/crop_app/scaler.pkl")
joblib.dump(le, "../agri_project/crop_app/label_encoder.pkl")  # ✅ IMPORTANT

print("\n✅ Model, Scaler & Encoder saved successfully!")