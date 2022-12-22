from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('home/', views.home),
   #Algoritmos
    path('EligeEDA/',views.preEDA),
    path('EDA/<int:pk>', views.EDA, name='EDA'),

    path('EligePCA/',views.prePCA),
    path('PCA/<int:pk>', views.PCA_1, name='PCA'),
    path('PCA-2/<int:pk>', views.PCA_2, name='PCA2'),
    
    path('EligeAD/',views.preAD),
    path('AD_P/<int:pk>', views.AD_P, name='AD_P'),
    path('AD_P-2/<int:pk>', views.AD_P_2, name='AD_P2'),
    path('AD_C/<int:pk>', views.AD_C, name='AD_C'),
    
    path('EligeBA/',views.preBA),
    path('BA_P/<int:pk>', views.BA_P, name='BA_P'),
    path('BA_P-2/<int:pk>', views.BA_P_2, name='BA_P2'),
    
    path('BA_C/<int:pk>', views.BA_C, name='BA_C'),
    
    #Proyectos
    path('Proyectos/', views.lista_Proyectos, name='project_list'),
    path('Proyectos/creaProyecto/', views.crea_Proyecto, name='upload_project'),
    path('Proyectos/eliminarProyecto/<int:pk>/', views.delete_project, name='delete_project'),
    #Pruebas
    path('Busqueda/',views.busqueda),
]