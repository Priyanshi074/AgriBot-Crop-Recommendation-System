from django.db import models
from django.contrib.auth.models import User

class DiseasePrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    disease = models.CharField(max_length=100)
    remedy = models.TextField()
    image = models.ImageField(upload_to='disease_images/')
    created_at = models.DateTimeField(auto_now_add=True)
