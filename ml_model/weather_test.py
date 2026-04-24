import requests

API_KEY = "9b750e497d41cd1f9877e52db116bd1b"

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]

    # Rainfall may not always exist
    rainfall = 0
    if "rain" in data and "1h" in data["rain"]:
        rainfall = data["rain"]["1h"]

    return temperature, humidity, rainfall

temp, hum, rain = get_weather("Delhi")
print("Temperature:", temp)
print("Humidity:", hum)
print("Rainfall:", rain)