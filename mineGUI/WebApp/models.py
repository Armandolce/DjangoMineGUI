from django.db import models
from django.core import validators

class Proyecto(models.Model):
    Nombre = models.CharField(max_length=100)
    descripcion = models.CharField(max_length=100)
    URL = models.CharField(max_length=100, null=True)
    data = models.FileField(upload_to='WebApp/data/')

    def __str__(self):
        return self.title

    def delete(self, *args, **kwargs):
        self.data.delete()
        super().delete(*args, **kwargs)