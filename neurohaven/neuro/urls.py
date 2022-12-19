import imp
from django.contrib import admin
from django.urls import path
from neuro import views
urlpatterns = [
    path("",views.home, name = "home"),
    path("login/",views.login,name = "login"),
    path("upload/",views.upload, name = "upload"),
    path("result/", views.result, name = "result"),
    path('logout',views.logout_view,name="logout"),
]
