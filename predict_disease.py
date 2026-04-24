import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("crop_disease_model.h5")

# Class names (same order as training folders)
class_names = [
    "Apple_leaf",
    "Apple_rust_leaf",
    "Apple_Scab_Leaf",
    "Bell_pepper_leaf",
    "Bell_pepper_leaf_spot",
    "Potato_leaf_early_blight",
    "Potato_leaf_late_blight",
    "Tomato_Early_blight"
]

def predict_disease(img_path):

    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction)

    print("Predicted Disease:", predicted_class)
    print("Confidence:", confidence)

# Example test image
predict_disease("test_image.jpg")