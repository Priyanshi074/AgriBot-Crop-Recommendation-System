import joblib
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

from django.shortcuts import render, redirect
from django.db.models import Count, Avg
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required

from .models import CropPrediction



# ---------------- MAIN PAGES ----------------

def main_home(request):
    return render(request, "main_home.html")


def signup(request):
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Check if user exists
        if User.objects.filter(username=email).exists():
            return render(request, 'signup.html', {
                "error": "User already exists"
            })

        # Create user
        user = User.objects.create_user(
            username=email,
            email=email,
            password=password,
            first_name=name
        )

        login(request, user)  # Auto login
        return redirect('/features/')

    return render(request, 'signup.html')


def login_view(request):
    if request.method == "POST":
        email = request.POST.get('email')
        password = request.POST.get('password')

        user = authenticate(request, username=email, password=password)

        if user is not None:
            login(request, user)
            return redirect('/features/')        
        else:
            return render(request, 'login.html', {
                "error": "Invalid email or password"
            })

    return render(request, 'login.html')

@login_required(login_url='/login/')
def features(request):
    return render(request, 'features.html')
# ---------------- CONFIG ----------------

API_KEY = "YOUR_API_KEY"

CITY_TO_STATE = {
    "delhi": "delhi",
    "mumbai": "maharashtra",
    "pune": "maharashtra",
    "nagpur": "maharashtra",
    "lucknow": "uttar pradesh",
    "kanpur": "uttar pradesh",
    "agra": "uttar pradesh",
    "jaipur": "rajasthan",
    "udaipur": "rajasthan",
    "patna": "bihar",
    "bhopal": "madhya pradesh",
    "indore": "madhya pradesh",
    "chennai": "tamil nadu",
    "coimbatore": "tamil nadu",
    "kolkata": "west bengal",
    "hyderabad": "telangana",
    "bangalore": "karnataka"
}

CROP_NAME_MAP = {
    "rice": "Rice",
    "maize": "Maize",
    "chickpea": "Chickpea",
    "kidneybeans": "Kidney Beans",
    "pigeonpeas": "Pigeon Peas",
    "mothbeans": "Moth Beans",
    "mungbean": "Moong",
    "blackgram": "Urad",
    "lentil": "Lentil",
    "pomegranate": "Pomegranate",
    "banana": "Banana",
    "mango": "Mango",
    "grapes": "Grapes",
    "watermelon": "Watermelon",
    "muskmelon": "Muskmelon",
    "apple": "Apple",
    "orange": "Orange",
    "papaya": "Papaya",
    "coconut": "Coconut",
    "cotton": "Cotton",
    "jute": "Jute",
    "coffee": "Coffee"
}


# ---------------- PATH ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "cleaned_weather.csv")

model = joblib.load(os.path.join(BASE_DIR, "crop_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))


# ---------------- LOAD WEATHER DATA ----------------

try:
    weather_df = pd.read_csv(DATASET_PATH)
    weather_df["state"] = weather_df["state"].astype(str).str.strip().str.lower()
    weather_df["month"] = weather_df["month"].astype(int)
except:
    weather_df = pd.DataFrame()


# ---------------- HISTORICAL WEATHER ----------------

def get_historical_weather(city):
    try:
        if weather_df.empty:
            return 25, 60, 100

        city = city.strip().lower()

        TYPO_FIX = {
            "gujrat": "gujarat",
            "gujurat": "gujarat",
        }

        if city in TYPO_FIX:
            city = TYPO_FIX[city]

        state = CITY_TO_STATE.get(city)

        if not state:
            if city in weather_df["state"].unique():
                state = city

        if not state:
            return 25, 60, 100

        month = datetime.now().month

        row = weather_df[
            (weather_df["state"] == state) &
            (weather_df["month"] == month)
        ]

        if not row.empty:
            return (
                float(row["temperature"].mean()),
                float(row["humidity"].mean()),
                float(row["rainfall"].mean())
            )

        state_data = weather_df[weather_df["state"] == state]

        if not state_data.empty:
            return (
                float(state_data["temperature"].mean()),
                float(state_data["humidity"].mean()),
                float(state_data["rainfall"].mean())
            )

        return (
            float(weather_df["temperature"].mean()),
            float(weather_df["humidity"].mean()),
            float(weather_df["rainfall"].mean())
        )

    except:
        return 25, 60, 100


# ---------------- CROP PRICE ----------------

def get_crop_price(crop):
    mapped_crop = CROP_NAME_MAP.get(crop.lower(), crop)

    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

    params = {
        "api-key": "579b464db66ec23bdd000001852e6567a65641594def70e31542ad3b",
        "format": "json",
        "limit": 1,
        "filters[commodity]": mapped_crop
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if data.get("records"):
                r = data["records"][0]

                return {
                    "min_price": r.get("min_price"),
                    "max_price": r.get("max_price"),
                    "market": r.get("market"),
                    "state": r.get("state")
                }
    except:
        pass

    return None


# ---------------- HOME ----------------

def home(request):
    if request.method == "POST":
        try:
            N = float(request.POST.get("nitrogen", 0))
            P = float(request.POST.get("phosphorus", 0))
            K = float(request.POST.get("potassium", 0))
            ph = float(request.POST.get("ph", 7))
            city = request.POST.get("city", "Delhi")

            temp, humidity, rainfall = get_historical_weather(city)

            input_data = pd.DataFrame(
                [[N, P, K, temp, humidity, ph, rainfall]],
                columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
            )

            input_scaled = scaler.transform(input_data)
            probs = model.predict_proba(input_scaled)[0]

            # Top 3 crops
            top3_idx = np.argsort(probs)[-3:][::-1]

            recommendations = []

            for i in top3_idx:
                crop_name = le.inverse_transform([i])[0]
                prob = probs[i]

                price_data = get_crop_price(crop_name)

                recommendations.append({
                    "crop": crop_name,
                    "confidence": round(prob * 100, 2),
                    "price": price_data
                })

            # Best crop
            prediction = recommendations[0]["crop"]
            confidence = recommendations[0]["confidence"]

            # ✅ SAVE ONLY ONCE (correct way)
            if request.user.is_authenticated:
                CropPrediction.objects.create(
                    user=request.user,
                    nitrogen=N,
                    phosphorus=P,
                    potassium=K,
                    ph=ph,
                    city=city,
                    temperature=temp,
                    humidity=humidity,
                    rainfall=rainfall,
                    predicted_crop=prediction,
                    confidence=confidence
                )

            return render(request, "result.html", {
                "recommendations": recommendations,
                "top_3": [(r["crop"], r["confidence"]) for r in recommendations],
                "prediction": prediction,
                "confidence": confidence,
                "temperature": temp,
                "humidity": humidity,
                "rainfall": rainfall,
                "city": city
            })

        except Exception as e:
            print("ERROR:", e)
            return render(request, "home.html", {"error": str(e)})
#     if request.method == "POST":
#         try:
#             N = float(request.POST.get("nitrogen", 0))
#             P = float(request.POST.get("phosphorus", 0))
#             K = float(request.POST.get("potassium", 0))
#             ph = float(request.POST.get("ph", 7))
#             city = request.POST.get("city", "Delhi")

#             temp, humidity, rainfall = get_historical_weather(city)

#             input_data = pd.DataFrame(
#                 [[N, P, K, temp, humidity, ph, rainfall]],
#                 columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
#             )

#             input_scaled = scaler.transform(input_data)
#             probs = model.predict_proba(input_scaled)[0]

#             # Top 3 crops
#             top3_idx = np.argsort(probs)[-3:][::-1]

#             recommendations = []

#             for i in top3_idx:
#                 crop_name = le.inverse_transform([i])[0]
#                 prob = probs[i]

#                 price_data = get_crop_price(crop_name)

#                 recommendations.append({
#                     "crop": crop_name,
#                     "confidence": round(prob * 100, 2),
#                     "price": price_data
#                 })

#                 if request.user.is_authenticated:
#                     CropPrediction.objects.create(
#                     user=request.user,
#                     crop=predicted_crop,
#                     confidence=confidence_score
#                 )
                    
#             prediction = recommendations[0]["crop"]
#             confidence = recommendations[0]["confidence"]

#             # Save prediction
#             CropPrediction.objects.create(
#                 nitrogen=N,
#                 phosphorus=P,
#                 potassium=K,
#                 ph=ph,
#                 city=city,
#                 temperature=temp,
#                 humidity=humidity,
#                 rainfall=rainfall,
#                 predicted_crop=prediction,
#                 confidence=confidence
#             )

#             return render(request, "result.html", {
#                 "recommendations": recommendations,
#                 "top_3": [(r["crop"], r["confidence"]) for r in recommendations],
#                 "prediction": prediction,
#                 "confidence": confidence,
#                 "temperature": temp,
#                 "humidity": humidity,
#                 "rainfall": rainfall,
#                 "city": city
#             })

#         except Exception as e:
#             return render(request, "home.html", {"error": str(e)})

    return render(request, "home.html")


# ---------------- DASHBOARD ----------------

@login_required(login_url='/login/')
def dashboard(request):
    total_predictions = CropPrediction.objects.count()

    crop_distribution = (
        CropPrediction.objects
        .values("predicted_crop")
        .annotate(count=Count("predicted_crop"))
        .order_by("-count")
    )

    avg_confidence = CropPrediction.objects.aggregate(
        Avg("confidence")
    )["confidence__avg"]

    return render(request, "dashboard.html", {
        "total_predictions": total_predictions,
        "crop_distribution": crop_distribution,
        "avg_confidence": round(avg_confidence or 0, 2)
    })


# ---------------- LOGOUT ----------------

def logout_view(request):
    logout(request)
    return redirect('/')
