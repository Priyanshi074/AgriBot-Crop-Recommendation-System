from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import get_bot_response
from django.shortcuts import render
from django.http import JsonResponse
from .utils import get_bot_response


def chatbot_page(request):
    return render(request, "chatbot/chatbot.html")


# def chatbot_api(request):
#     if request.method == "POST":
#         user_msg = request.POST.get("message")
#         bot_reply = get_bot_response(user_msg)
#         return JsonResponse({"response": bot_reply})

@csrf_exempt
def chatbot_api(request):
    if request.method == "POST":
        user_msg = request.POST.get("message")
        print("User:", user_msg)   # 👈 debug

        bot_reply = get_bot_response(user_msg)
        print("Bot:", bot_reply)   # 👈 debug

        return JsonResponse({"response": bot_reply})
# ===================== API ENDPOINT ===================== #
# @csrf_exempt
# def chatbot_api(request):
#     """
#     Handles AJAX requests from frontend
#     """
#     if request.method == "POST":
#         user_msg = request.POST.get("message", "").strip()

#         if not user_msg:
#             return JsonResponse({
#                 "response": "⚠️ Please enter a message."
#             })

#         # 🔥 Hybrid bot response (rule + AI)
#         bot_reply = get_bot_response(user_msg)

#         return JsonResponse({
#             "response": bot_reply
#         })

#     return JsonResponse({
#         "response": "Invalid request"
#     })