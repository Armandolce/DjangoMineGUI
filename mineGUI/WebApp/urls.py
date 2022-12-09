from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('home/', views.home),
    path('EligeEDA/',views.preEDA),
    path('EDA/<int:pk>', views.EDA, name='EDA'),
    path('EligePCA/',views.prePCA),
    path('PCA/<int:pk>', views.PCA_, name='PCA'),
    path('EligeAB/',views.preAB),
    path('AB_P/<int:pk>', views.AB_P, name='AB_P'),
    path('AB_C/<int:pk>', views.AB_C, name='AB_C'),
    path('Busqueda/',views.busqueda),
    path('Proyectos/', views.lista_Proyectos, name='project_list'),
    path('Proyectos/creaProyecto/', views.crea_Proyecto, name='upload_project'),
    path('Proyectos/eliminarProyecto/<int:pk>/', views.delete_project, name='delete_project'),
]