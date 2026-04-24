# import os
# import numpy as np
# import base64
# from django.shortcuts import render
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing import image

# # Load model

# # Classes
# class_names = [
#     "Apple_leaf",
#     "Apple_rust_leaf",
#     "Apple_Scab_Leaf",
#     "Bell_pepper_leaf",
#     "Bell_pepper_leaf_spot",
#     "Potato_leaf_early_blight",
#     "Potato_leaf_late_blight",
#     "Tomato_Early_blight"
# ]

# # Remedies
# remedies = {
#     "Apple_leaf": "Leaf is healthy. No treatment needed.",
#     "Apple_rust_leaf": "Apply fungicide and remove infected leaves.",
#     "Apple_Scab_Leaf": "Use sulfur-based fungicide and prune affected areas.",
#     "Bell_pepper_leaf": "Healthy leaf. Maintain proper watering.",
#     "Bell_pepper_leaf_spot": "Remove infected leaves and apply copper fungicide.",
#     "Potato_leaf_early_blight": "Apply fungicide and practice crop rotation.",
#     "Potato_leaf_late_blight": "Use copper fungicide and avoid excess moisture.",
#     "Tomato_Early_blight": "Remove infected leaves and apply fungicide."
# }

# # Convert to readable disease names
# def get_disease_name(predicted_class):
#     predicted_class = predicted_class.lower()

#     if "rust" in predicted_class:
#         return "Rust Disease"
#     elif "scab" in predicted_class:
#         return "Scab Disease"
#     elif "early_blight" in predicted_class:
#         return "Early Blight"
#     elif "late_blight" in predicted_class:
#         return "Late Blight"
#     elif "spot" in predicted_class:
#         return "Leaf Spot"
#     else:
#         return "Healthy"

# # Prediction function
# # def predict_disease(img_path):
# #     from tensorflow.keras.models import load_model   # ✅ moved here
# #     from tensorflow.keras.preprocessing import image

# #     model = load_model("crop_disease_model.h5")

# #     img = image.load_img(img_path, target_size=(128,128))
# #     img_array = image.img_to_array(img)
# #     img_array = img_array / 255.0
# #     img_array = np.expand_dims(img_array, axis=0)

# #     prediction = model.predict(img_array)

# #     predicted_class = class_names[np.argmax(prediction)]

# #     return predicted_class, remedies[predicted_class]

# def predict_disease(img_path):
#     try:
#         import tensorflow as tf
#         from tensorflow.keras.preprocessing import image

#         model = tf.keras.models.load_model("crop_disease_model.h5")

#         img = image.load_img(img_path, target_size=(128,128))
#         img_array = image.img_to_array(img)
#         img_array = img_array / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         prediction = model.predict(img_array)
#         predicted_class = class_names[np.argmax(prediction)]

#         return predicted_class, remedies[predicted_class]

#     except ModuleNotFoundError:
#         # ✅ TensorFlow not installed
#         return "Apple_leaf", "Demo Mode: TensorFlow not installed"

#     except Exception as e:
#         return "Apple_leaf", f"Error: {str(e)}"

# # MAIN VIEW
# def upload_image(request):

#     if request.method == "POST":

#         file_path = None  #  FIX: always initialize

#         # CAMERA CASE
#         if request.POST.get('captured_image'):
#             image_data = request.POST.get('captured_image')

#             format, imgstr = image_data.split(';base64,')
#             img_bytes = base64.b64decode(imgstr)

#             file_path = "media/captured.png"

#             with open(file_path, 'wb') as f:
#                 f.write(img_bytes)

#         #  FILE UPLOAD CASE
#         elif request.FILES.get('image'):
#             img = request.FILES['image']

#             file_path = os.path.join("media", img.name)

#             with open(file_path, 'wb+') as destination:
#                 for chunk in img.chunks():
#                     destination.write(chunk)

#         # NOTHING SELECTED
#         else:
#             return render(request, "disease_detection/upload.html", {
#                 "error": "Please upload or capture an image"
#             })

#         #  Prediction
#         # predicted_class, remedy = predict_disease(file_path)
#         try:
#             predicted_class, remedy = predict_disease(file_path)
#         except:
#             predicted_class, remedy = "Apple_leaf", "Prediction failed"
#         result = get_disease_name(predicted_class)

#         image_url = "/" + file_path

#         return render(request, "disease_detection/result.html", {
#             "result": result,
#             "remedy": remedy,
#             "image_url": image_url
#         })

#     return render(request, "disease_detection/upload.html")


import os
import base64
import numpy as np
import joblib
from PIL import Image
from django.shortcuts import render
from .models import DiseasePrediction

# ==============================
# LOAD MODEL (ONLY ONCE)
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, 'ml_model', 'disease_model.pkl')
CLASS_PATH = os.path.join(PROJECT_ROOT, 'ml_model', 'class_names.pkl')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

if not os.path.exists(CLASS_PATH):
    raise FileNotFoundError(f"Class file not found at {CLASS_PATH}")

model = joblib.load(MODEL_PATH)
class_names = joblib.load(CLASS_PATH)

# ==============================
# REMEDIES
# ==============================
remedies = {
    "Apple_leaf": "Leaf is healthy. No treatment needed.",
    "Apple_rust_leaf": "Apply fungicide and remove infected leaves.",
    "Apple_Scab_Leaf": "Use sulfur-based fungicide and prune affected areas.",
    "Bell_pepper_leaf": "Healthy leaf. Maintain proper watering.",
    "Bell_pepper_leaf_spot": "Remove infected leaves and apply copper fungicide.",
    "Potato_leaf_early_blight": "Apply fungicide and practice crop rotation.",
    "Potato_leaf_late_blight": "Use copper fungicide and avoid excess moisture.",
    "Tomato_Early_blight": "Remove infected leaves and apply fungicide."
}

# ==============================
# CLEAN NAME FOR UI
# ==============================
def get_disease_name(predicted_class):
    predicted_class = predicted_class.lower()

    if "rust" in predicted_class:
        return "Rust Disease"
    elif "scab" in predicted_class:
        return "Scab Disease"
    elif "early_blight" in predicted_class:
        return "Early Blight"
    elif "late_blight" in predicted_class:
        return "Late Blight"
    elif "spot" in predicted_class:
        return "Leaf Spot"
    else:
        return "Healthy"

# ==============================
# PREDICTION FUNCTION (NO TF)
# ==============================
def predict_disease(img_path):
    img = Image.open(img_path).convert("RGB").resize((64, 64))
    img_array = np.array(img).flatten().reshape(1, -1)

    prediction = model.predict(img_array)[0]
    predicted_class = class_names[prediction]

    return predicted_class, remedies.get(predicted_class, "No remedy available")

# ==============================
# MAIN VIEW
# ==============================
def upload_image(request):

    if request.method == "POST":

        file_path = None

        # CAMERA INPUT
        if request.POST.get('captured_image'):
            image_data = request.POST.get('captured_image')

            format, imgstr = image_data.split(';base64,')
            img_bytes = base64.b64decode(imgstr)

            file_path = os.path.join("media", "captured.png")

            with open(file_path, 'wb') as f:
                f.write(img_bytes)

        # FILE UPLOAD
        elif request.FILES.get('image'):
            img = request.FILES['image']

            file_path = os.path.join("media", img.name)

            with open(file_path, 'wb+') as destination:
                for chunk in img.chunks():
                    destination.write(chunk)

        else:
            return render(request, "disease_detection/upload.html", {
                "error": "Please upload or capture an image"
            })

        # ==============================
        # PREDICTION
        # ==============================
        try:
            predicted_class, remedy = predict_disease(file_path)
            result = get_disease_name(predicted_class)
        except Exception as e:
            result = "Prediction Failed"
            remedy = str(e)

        image_url = "/" + file_path
        if request.user.is_authenticated:
            DiseasePrediction.objects.create(
            user=request.user,
            disease=result,
            remedy=remedy,
            image=file_path
        )
            
        return render(request, "disease_detection/result.html", {
            "result": result,
            "remedy": remedy,
            "image_url": image_url
        })

    return render(request, "disease_detection/upload.html")
