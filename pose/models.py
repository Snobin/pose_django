from django.db import models

class Video(models.Model):
    Body_part = models.CharField(max_length=100)
    File = models.FileField(upload_to='videos/')

