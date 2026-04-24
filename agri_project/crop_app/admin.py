from django.contrib import admin
from .models import CropPrediction

@admin.register(CropPrediction)
class CropPredictionAdmin(admin.ModelAdmin):
    list_display = (
        "predicted_crop",
        "confidence",
        "city",
        "created_at"
    )
    list_filter = ("predicted_crop", "city")
    search_fields = ("city",)