import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from PIL import Image

# 📁 Paths (VERY IMPORTANT)
TRAIN_PATH = "../train"
TEST_PATH = "../test"

IMG_SIZE = (64, 64)

# -----------------------------
# LOAD DATA FUNCTION
# -----------------------------
def load_data(path):
    X = []
    y = []
    class_names = os.listdir(path)

    for label, folder in enumerate(class_names):
        folder_path = os.path.join(path, folder)

        if not os.path.isdir(folder_path):
            continue

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            try:
                # img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)   
                img = cv2.imread(img_path)
                img = cv2.resize(img, (128, 128))   # ✅ IMPORTANT               
                # img = np.array(img).flatten()   # convert image → 1D vector
                X.append(img)
                y.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

    return np.array(X), np.array(y), class_names


# -----------------------------
# LOAD TRAIN + TEST DATA
# -----------------------------
print("📥 Loading training data...")
X_train, y_train, class_names = load_data(TRAIN_PATH)

print("📥 Loading testing data...")
X_test, y_test, _ = load_data(TEST_PATH)

print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Testing samples: {len(X_test)}")

# -----------------------------
# TRAIN MODEL
# -----------------------------
print("🚀 Training model...")

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# -----------------------------
# TEST MODEL
# -----------------------------
print("🧪 Testing model...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "disease_model.pkl")
joblib.dump(class_names, "class_names.pkl")

print("💾 Model saved successfully!")