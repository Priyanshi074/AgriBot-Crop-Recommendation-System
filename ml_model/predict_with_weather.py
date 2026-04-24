import joblib
import numpy as np
import requests
import pandas as pd

API_KEY = "9b750e497d41cd1f9877e52db116bd1b"

model = joblib.load("crop_model.pkl")

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]

    rainfall = 0
    if "rain" in data and "1h" in data["rain"]:
        rainfall = data["rain"]["1h"]

    return temperature, humidity, rainfall

def predict_crop(N, P, K, ph, city):
    temp, humidity, _ = get_weather(city)

    forecast_rainfall = get_forecast_rainfall(city)

    historical_rainfall = get_historical_rainfall(city)

    rainfall = forecast_rainfall + historical_rainfall

    input_data = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temp,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    prediction = model.predict(input_data)

    return prediction[0]

result = predict_crop(90, 40, 40, 6.5, "Delhi")
print("Recommended Crop:", result)

