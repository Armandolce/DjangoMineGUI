from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('home/', views.home),
    path('pyScript/', views.pyscript),
    path('EDA/', views.EDA),
    path('Busqueda/',views.busqueda),
]