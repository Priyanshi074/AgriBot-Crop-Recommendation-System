import os
import re
from datetime import datetime
from django.urls import reverse, NoReverseMatch
import ollama

# ===================== OLLAMA AI ===================== #
def ask_local_ai(prompt):
    try:
        response = ollama.chat(
            model='phi3',  # ⚡ fast model
            messages=[
                {
                    "role": "system",
                "content": (
                    "You are an agriculture expert ONLY. "
                    "Answer strictly related to agriculture, farming, crops, soil, weather, irrigation, and plant diseases. "
                    "If the question is not related to agriculture, reply: "
                    "'⚠️ I can only answer agriculture-related questions.' "
                    "Keep answers short (2-3 lines)."
                )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response['message']['content']

    except Exception as e:
        print("Ollama Error:", e)
        return "⚠️ AI not available"


# ===================== STATE DATA ===================== #
STATE_CROPS = {
    "andhra pradesh": "Rice, Tobacco, Chillies, Cotton, Turmeric, Maize",
    "arunachal pradesh": "Rice, Maize, Millet, Ginger, Mustard, Oranges",
    "assam": "Tea, Jute, Rice, Mustard, Sugarcane, Areca nut",
    "bihar": "Rice, Wheat, Maize, Sugarcane, Litchi, Jute",
    "chhattisgarh": "Rice, Maize, Millets, Oilseeds, Small Millets",
    "goa": "Cashew, Coconut, Rice, Areca nut, Mango",
    "gujarat": "Cotton, Groundnut, Cumin, Castor, Onion, Tobacco",
    "haryana": "Wheat, Rice, Mustard, Sugarcane, Cotton, Sunflower",
    "himachal pradesh": "Apples, Wheat, Maize, Barley, Seed Potato, Cherries",
    "jharkhand": "Rice, Maize, Pulses, Vegetables, Niger Seed",
    "karnataka": "Coffee, Ragi, Sugarcane, Sunflower, Sandalwood, Silk",
    "kerala": "Rubber, Coconut, Spices, Tea, Cashew, Coffee, Cardamom",
    "madhya pradesh": "Soybean, Wheat, Gram, Pulses, Garlic, Linseed",
    "maharashtra": "Cotton, Soybean, Sugarcane, Jowar, Grapes, Onions, Alphonso Mango",
    "manipur": "Rice, Maize, Pineapple, Oranges, Turmeric",
    "meghalaya": "Rice, Maize, Pineapple, Turmeric, Ginger, Strawberry",
    "mizoram": "Rice, Maize, Ginger, Turmeric, Passion Fruit",
    "nagaland": "Rice, Maize, Coffee, Cardamom, Tea, Bamboo",
    "odisha": "Rice, Jute, Oilseeds, Cashew, Rubber, Turmeric",
    "punjab": "Wheat, Rice, Cotton, Barley, Kinnow, Mustard",
    "rajasthan": "Bajra, Wheat, Mustard, Guar Seed, Cumin, Coriander",
    "sikkim": "Cardamom, Ginger, Orange, Buckwheat, Large Cardamom",
    "tamil nadu": "Rice, Sugarcane, Groundnut, Bananas, Coconut, Turmeric, Flowers",
    "telangana": "Cotton, Rice, Maize, Turmeric, Chillies, Soya",
    "tripura": "Rice, Rubber, Bamboo, Jackfruit, Pineapple",
    "uttar pradesh": "Wheat, Sugarcane, Rice, Potatoes, Mustard, Mentha, Mangoes",
    "uttarakhand": "Rice, Wheat, Finger Millet, Soybeans, Basmati Rice, Plums",
    "west bengal": "Rice, Jute, Tea, Potatoes, Tobacco, Betel Vine",
    "jammu and kashmir": "Saffron, Apples, Walnuts, Almonds, Cherries",
    "ladakh": "Barley, Apricots, Buckwheat, Seabuckthorn",
    "andaman and nicobar islands": "Coconut, Areca nut, Rubber, Red Oil Palm",
    "puducherry": "Rice, Coconut, Sugarcane, Bananas",
    "delhi": "Wheat, Vegetables, Flowers, Greenhouse crops",
    "chandigarh": "Wheat, Maize, Vegetables, Flowers",
    "dadra and nagar haveli and daman and diu": "Rice, Ragi, Pulses, Mangoes",
    "lakshadweep": "Coconut, Coir, Fish products (Tuna)"
}


# ===================== HELPERS ===================== #
def get_urls():
    try:
        crop_url = reverse('home')
    except NoReverseMatch:
        crop_url = "/crop/"

    try:
        disease_url = reverse('disease_home')
    except NoReverseMatch:
        disease_url = "/disease/"

    return crop_url, disease_url


def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    return "Good evening"


# ===================== RULE-BASED ENGINE ===================== #
def rule_based_response(msg):
    msg = msg.lower().strip()
    crop_url, disease_url = get_urls()
    greet = get_greeting()

    # ===== 1. STATE DETECTION ===== #
    # for state, crops in STATE_CROPS.items():
    #     if state in msg:
    #         return f"🌾 In {state.title()}, major crops are: {crops}"

    # ===== 2. RULES ===== #
    rules = [

        # Greeting
        (r"\b(hi|hello|hey|hii|hola)\b",
         lambda: f"👋 {greet}! I am AgriBot. How can I help you today?"),

        # Thanks
        (r"(thanks|thank you|thx)",
         lambda: "😊 You're welcome! Happy farming! 🌾"),
        # ===== PROJECT FEATURE EXPLANATIONS ===== #

        (r"(how to use crop recommendation|use crop recommendation|how crop recommendation works)",
        lambda: f"""🌱 <b>How to use Crop Recommendation:</b><br>
        1. Enter soil details (N, P, K values)<br>
        2. Enter pH and city <br>
        3. Click predict<br>
        👉 <a href="{crop_url}">Go to Crop Recommendation</a>"""),

        (r"(what is crop recommendation)",
        lambda: "🌾 Crop Recommendation is a system that suggests the best crop based on soil nutrients (NPK), pH, and environmental conditions."),

        (r"(how to use disease detection|how to use disease prediction)",
        lambda: f"""🍃 <b>How to use Disease Detection:</b><br>
        1. Upload a leaf image<br>
        2. Click detect<br>
        3. View disease result and solution<br>
        👉 <a href="{disease_url}">Go to Disease Detection</a>"""),

        (r"(what is disease detection|what is disease prediction)",
        lambda: "🍂 Disease Detection identifies plant diseases from leaf images using machine learning and suggests treatments."),


        # # ===== FEATURE LINKS ===== #
        # (r"(recommend.*crop|crop suggestion|best crop|suggest crop)",
        #  lambda: f'🌱 Click here for Crop Recommendation 👉 <a href="{crop_url}">Go to Crop Page</a>'),

        # (r"(detect.*disease|leaf disease|plant disease)",
        #  lambda: f'🍃 Click here for Disease Detection 👉 <a href="{disease_url}">Go to Disease Page</a>'),

        # # ===== DISEASE INFO ===== #
        # (r"(rust)",
        #  lambda: "🍂 Rust Disease:\n• Orange/brown spots\n• Use fungicide"),

        # (r"(scab)",
        #  lambda: "🍃 Scab Disease:\n• Rough dark spots\n• Remove infected parts"),

        # (r"(early blight)",
        #  lambda: "🌿 Early Blight:\n• Brown ring spots\n• Crop rotation"),

        # (r"(late blight)",
        #  lambda: "🌧 Late Blight:\n• Water-soaked spots\n• Remove infected plants"),

        # (r"(leaf spot|spots on leaves)",
        #  lambda: "🍂 Leaf Spot:\n• Brown/black spots\n• Use fungicide"),

        # # ===== BASIC KNOWLEDGE ===== #
        # (r"(what is farming|define agriculture)",
        #  lambda: "🌱 Farming is growing crops and raising animals"),

        # (r"(what is irrigation)",
        #  lambda: "💧 Irrigation is supplying water to crops"),

        # (r"(what is fertilizer)",
        #  lambda: "🧪 Fertilizers provide nutrients like Nitrogen, Phosphorus, Potassium"),

        # (r"(soil|ph)",
        #  lambda: "🌍 Ideal soil pH for most crops is 6–7"),

        # # ===== AGRI HELP ===== #
        # (r"(pest|insect)",
        #  lambda: "🐛 Use neem oil or suitable pesticides"),

        # (r"(weather)",
        #  lambda: "🌦 Weather plays a crucial role in crop growth"),

        # (r"(scheme|government)",
        #  lambda: "🏛 Popular schemes: PM-KISAN, Crop Insurance, Soil Health Card"),

        # ===== HELP ===== #
        # (r"(help|guide)",
        #  lambda: f'📌 <a href="{crop_url}">Crop</a> | <a href="{disease_url}">Disease</a>'),
    ]

    for pattern, response in rules:
        if re.search(pattern, msg):
            return response()

    return None  # ⚠️ IMPORTANT (not fallback text)


# ===================== MAIN FUNCTION ===================== #
def get_bot_response(message):
    msg = message.lower()

    # ✅ 1. Rule-based (FIRST PRIORITY)
    rule_reply = rule_based_response(msg)
    if rule_reply:
        return rule_reply

    # ✅ 2. Smart Navigation (extra safety)
    crop_url, disease_url = get_urls()

    if "crop recommendation" in msg or "crop recommend" in msg :
        return f'🌾 Visit Crop Recommendation 👉 <a href="{crop_url}">Open</a>'

    if "disease" in msg or "leaf" in msg:
        return f'🍃 Visit Disease Detection 👉 <a href="{disease_url}">Open</a>'

    # ✅ 3. Fallback → Ollama AI
    return ask_local_ai(message)
# import os
# import re
# from datetime import datetime
# from django.conf import settings
# from django.urls import reverse, NoReverseMatch
# import requests
# import ollama

# # def ask_local_ai(prompt):
# #     try:
# #         response = requests.post(
# #             "http://localhost:11434/api/generate",
# #             json={
# #                 "model": "llama3",
# #                 "prompt": f"You are an agriculture expert. Answer clearly:\n{prompt}",
# #                 "stream": False
# #             }
# #         )

# #         data = response.json()
# #         return data.get("response", "No response")

# #     except Exception as e:
# #         print("Local AI Error:", e)
# #         return "⚠️ AI not available"
# import ollama

# def ask_local_ai(prompt):
#     response = ollama.chat(
#         model='phi3',   # 🔥 faster model
#         messages=[{'role': 'user', 'content': prompt}]
#     )
#     return response['message']['content']

# # Main function used by views
# # def get_bot_response(msg):
# #     return ask_local_ai(msg)
# def get_bot_response(message):
#     try:
#         response = ollama.chat(
#             model='llama3',
#             messages=[
#                 {"role": "system", "content": "You are an agriculture expert."},
#                 {"role": "user", "content": message}
#             ]
#         )

#         return response['message']['content']

#     except Exception as e:
#         print("Ollama Error:", e)
#         return "⚠️ AI not available"
# # ===================== STATE DATA ===================== #
# STATE_CROPS = {
#     "andhra pradesh": "Rice, Tobacco, Chillies",
#     "arunachal pradesh": "Rice, Maize, Millet",
#     "assam": "Tea, Jute, Rice",
#     "bihar": "Rice, Wheat, Maize",
#     "chhattisgarh": "Rice, Maize, Millets",
#     "goa": "Cashew, Coconut, Rice",
#     "gujarat": "Cotton, Groundnut",
#     "haryana": "Wheat, Rice, Mustard",
#     "himachal pradesh": "Apples, Wheat, Maize",
#     "jharkhand": "Rice, Maize, Pulses",
#     "karnataka": "Coffee, Ragi, Sugarcane",
#     "kerala": "Rubber, Coconut, Spices",
#     "madhya pradesh": "Soybean, Wheat",
#     "maharashtra": "Cotton, Soybean, Sugarcane",
#     "manipur": "Rice, Maize, Pineapple",
#     "meghalaya": "Rice, Maize, Pineapple",
#     "mizoram": "Rice, Maize",
#     "nagaland": "Rice, Maize, Coffee",
#     "odisha": "Rice, Jute, Oilseeds",
#     "punjab": "Wheat, Rice",
#     "rajasthan": "Bajra, Wheat, Mustard",
#     "sikkim": "Cardamom, Ginger",
#     "tamil nadu": "Rice, Sugarcane, Groundnut",
#     "telangana": "Cotton, Rice, Maize",
#     "tripura": "Rice, Rubber",
#     "uttar pradesh": "Wheat, Sugarcane, Rice",
#     "uttarakhand": "Rice, Wheat",
#     "west bengal": "Rice, Jute, Tea",
#     "jammu and kashmir": "Saffron, Apples"
# }


# # ===================== HELPERS ===================== #
# def get_urls():
#     try:
#         crop_url = reverse('home')
#     except NoReverseMatch:
#         crop_url = "/crop/"

#     try:
#         disease_url = reverse('disease_home')
#     except NoReverseMatch:
#         disease_url = "/disease/"

#     return crop_url, disease_url


# def get_greeting():
#     hour = datetime.now().hour
#     if hour < 12:
#         return "Good morning"
#     elif hour < 17:
#         return "Good afternoon"
#     return "Good evening"


# # ===================== RcdULE-BASED ENGINE ===================== #
# def rule_based_response(msg):
#     msg = msg.lower().strip()
#     crop_url, disease_url = get_urls()
#     greet = get_greeting()

#     # ===== 1. STATE DETECTION (TOP PRIORITY) ===== #
#     for state, crops in STATE_CROPS.items():
#         if state in msg:
#             return f"🌾 In {state.title()}, major crops are: {crops}"

#     # ===== 2. INTENT RULES ===== #
#     rules = [

#         # Greeting
#         (r"\b(hi|hello|hey|hii|hola)\b",
#          lambda: f"👋 {greet}! I am AgriBot. How can I help you today?"),

#         # Thanks
#         (r"(thanks|thank you|thx)",
#          lambda: "😊 You're welcome! Happy farming! 🌾"),

#         # Feature links
#         (r"(recommend.*crop|crop suggestion|best crop)",
#          lambda: f'🌱 <a href="{crop_url}">Crop Recommendation</a><br>Enter soil data'),

#         (r"(detect.*disease|leaf disease)",
#          lambda: f'🍃 <a href="{disease_url}">Disease Detection</a><br>Upload leaf image'),

#         # ===== DISEASES ===== #
#         (r"(rust)",
#          lambda: "🍂 Rust Disease:\n• Orange/brown spots\n• Use fungicide"),

#         (r"(scab)",
#          lambda: "🍃 Scab Disease:\n• Rough dark spots\n• Remove infected parts"),

#         (r"(early blight)",
#          lambda: "🌿 Early Blight:\n• Brown ring spots\n• Crop rotation"),

#         (r"(late blight)",
#          lambda: "🌧 Late Blight:\n• Water-soaked spots\n• Remove infected plants"),

#         (r"(leaf spot|spots on leaves)",
#          lambda: "🍂 Leaf Spot:\n• Brown/black spots\n• Fungicide"),

#         # ===== BASIC QUESTIONS ===== #
#         (r"(what is farming|define agriculture)",
#          lambda: "🌱 Farming is growing crops and raising animals"),

#         (r"(what is irrigation)",
#          lambda: "💧 Irrigation is supplying water to crops"),

#         (r"(what is fertilizer)",
#          lambda: "🧪 Fertilizers provide nutrients (NPK)"),

#         (r"(soil|ph)",
#          lambda: "🌍 Ideal soil pH: 6–7"),

#         # ===== AGRI ===== #
#         (r"(pest|insect)",
#          lambda: "🐛 Use neem oil or pesticides"),

#         (r"(weather)",
#          lambda: "🌦 Weather affects crop growth"),

#         (r"(price|mandi)",
#          lambda: "💰 Use crop prediction for price"),

#         (r"(scheme|government)",
#          lambda: "🏛 PM-KISAN | Crop Insurance | Soil Card"),

#         # ===== SYSTEM ===== #
#         (r"(how.*work|logic)",
#          lambda: "🤖 Uses soil (NPK, pH) & weather data"),

#         # ===== HELP ===== #
#         (r"(help|guide)",
#          lambda: f'📌 <a href="{crop_url}">Crop</a> | <a href="{disease_url}">Disease</a>'),
#     ]

#     for pattern, response in rules:
#         if re.search(pattern, msg):
#             return response()

#     return "🤖 I didn't understand. Try asking about crops, diseases, or states."
