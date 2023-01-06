from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('home/', views.home),
    path('Selector/<str:Alg>', views.selector),
   
   #Algoritmos
    path('EDA/<int:pk>', views.EDA, name='EDA'),

    path('PCA/<int:pk>', views.PCA_1, name='PCA'),
    path('PCA-2/<int:pk>', views.PCA_2, name='PCA2'),
    
    path('AD/<int:pk>/<str:algType>',views.AD, name='AD'),
    path('AD-2/<int:pk>/<str:algType>',views.AD2, name='AD2'),
    path('AD-3/<int:pk>/<str:algType>',views.AD3, name='AD3'),
    path('ADError/<int:pk>/<str:algType>', views.ADErrror, name='ADError'),
    
    path('BA/<int:pk>/<str:algType>',views.BA, name='BA'),
    path('BA-2/<int:pk>/<str:algType>',views.BA2, name='BA2'),
    path('BA-3/<int:pk>/<str:algType>',views.BA3, name='BA3'),
    path('BAError/<int:pk>/<str:algType>', views.BAErrror, name='BAError'),
    
    path('SC/<int:pk>',views.SegClas, name='SC'),
    path('SC_2/<int:pk>',views.SegClas_2, name='SC_2'),
    path('SC_3/<int:pk>',views.SegClas_3, name='SC_3'),

    path('SVM/<int:pk>/<str:algType>', views.SVM, name='SVM'),
    path('SVM_2/<int:pk>/<str:algType>', views.SVM_2, name='SVM2'),
    path('SVM_3/<int:pk>/<str:algType>', views.SVM_3, name='SVM3'),

    #Proyectos
    path('Proyectos/', views.lista_Proyectos, name='project_list'),
    path('Proyectos/creaProyecto/', views.crea_Proyecto, name='upload_project'),
    path('Proyectos/eliminarProyecto/<int:pk>/', views.delete_project, name='delete_project'),
    #Pruebas
    path('Busqueda/',views.busqueda),
]