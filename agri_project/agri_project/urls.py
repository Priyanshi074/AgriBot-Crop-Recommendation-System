from django.contrib import admin
from django.urls import path, include
from crop_app import views
from django.conf import settings
from django.conf.urls.static import static

# urlpatterns = [
#     path('admin/', admin.site.urls),

#     # 🌿 MAIN LANDING PAGE (UI you created)
#     path('', views.main_home, name='main_home'),
#     path('', views.signup, name='signup'),
#     # 🌾 Crop Recommendation Module
#     path('crop/', views.home, name='home'),
#     path('dashboard/', views.dashboard, name='dashboard'),
#     path('login/', views.login_view, name='login'),

#     # 🦠 Disease Detection Module
#     path('disease/', include('disease_detection.urls')),

#     # 🌐 Language Support (important for your previous error)
#     path('i18n/', include('django.conf.urls.i18n')),
# ]

urlpatterns = [
    path('admin/', admin.site.urls),

    path('', views.main_home, name='main_home'),
    path('signup/', views.signup, name='signup'),   # ✅ FIXED
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    path('crop/', views.home, name='home'),
    path('dashboard/', views.dashboard),
    path('features/', views.features, name='features'),
    path('disease/', include('disease_detection.urls')),
    path('i18n/', include('django.conf.urls.i18n')),

]
# 📁 Media files (for uploaded images)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)