from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot_page, name='chatbot'),
    path('get-response/', views.chatbot_api, name='chatbot_api'),  # ✅ FIXED
]