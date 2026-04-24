from django.db import models
from django.contrib.auth.models import User

class CropPrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)

    nitrogen = models.FloatField()
    phosphorus = models.FloatField()
    potassium = models.FloatField()
    ph = models.FloatField()
    city = models.CharField(max_length=100)

    temperature = models.FloatField()
    humidity = models.FloatField()
    rainfall = models.FloatField()

    predicted_crop = models.CharField(max_length=100, default="Unknown")
    confidence = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.predicted_crop} ({self.city})"