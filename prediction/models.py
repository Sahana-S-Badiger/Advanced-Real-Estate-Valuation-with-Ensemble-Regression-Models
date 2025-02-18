from django.db import models

class HousePrediction(models.Model):
    transaction_date = models.FloatField()
    house_age = models.FloatField()
    distance_to_mrt = models.FloatField()
    convenience_stores = models.IntegerField()
    latitude = models.FloatField()
    longitude = models.FloatField()
    predicted_price = models.FloatField(null=True, blank=True)