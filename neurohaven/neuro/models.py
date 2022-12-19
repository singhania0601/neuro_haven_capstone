from django.db import models

# Create your models here.

class Signup(models.Model):
    username = models.CharField(max_length=122)
    email = models.CharField(max_length=122)
    password = models.CharField(max_length=122)
    cpass = models.CharField(max_length=122)
    