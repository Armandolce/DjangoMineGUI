from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('home/', views.home),
    path('EligeEDA/',views.preEDA),
    path('EDA/<int:pk>', views.EDA, name='EDA'),
    path('Busqueda/',views.busqueda),
    path('Proyectos/', views.lista_Proyectos, name='project_list'),
    path('Proyectos/creaProyecto/', views.crea_Proyecto, name='upload_project'),
    path('Proyectos/eliminarProyecto/<int:pk>/', views.delete_project, name='delete_project'),
]