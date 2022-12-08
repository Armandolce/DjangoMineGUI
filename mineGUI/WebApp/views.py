from django.shortcuts import render, redirect

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .forms import ProjectForm
from .models import Proyecto

# Create your views here.
def home(request):
    return render(request,'home.html')

def lista_Proyectos(request):
    proyectos = Proyecto.objects.all()
    return render(request, 'Proyectos.html', {
        'proyectos': proyectos
    })

def crea_Proyecto(request):
    if request.method == 'POST':
        form = ProjectForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('project_list')
    else:
        form = ProjectForm()
    
    return render(request, 'crea_Proyecto.html', { 'form': form})

def delete_project(request, pk):
    if request.method == 'POST':
        proyecto = Proyecto.objects.get(pk=pk)
        proyecto.delete()
    return redirect('project_list')

def busqueda(request):
    return render(request, 'Busqueda.html')

def preEDA(request):
    proyectos = Proyecto.objects.all()
    return render(request, 'EligeEDA.html', {'projects': proyectos})

def prePCA(request):
    proyectos = Proyecto.objects.all()
    return render(request, 'EligePCA.html', {'projects': proyectos})

def PCA_(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    source = proyecto.data
    context = {}
    #Comienzo del algoritmo
    df = pd.read_csv(source)
    df2 = df[:10]
    context['df']=df2
    
    #Forma del df
    size = df.shape
    context['size']=size
    
    #Correlaciones
    correlaciones = df.corr()
    #Mapa de calor de correlaciones
    calor = px.imshow(correlaciones, text_auto=True, aspect="auto")
    layout_corr = {
    'title': proyecto.Nombre,
    'height': 240,
    'width': 240,
    }
    mapaC = plot({'data': calor, 'layout': layout_corr}, output_type='div')
    context['corr']=correlaciones
    context['mapaC'] = mapaC

    #Estandarizacion de datos
    Estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
    NuevaMatriz = df.drop(columns=df.select_dtypes('object'))    # Se quitan las variables nominales
    NuevaMat = NuevaMatriz.dropna() 
    MEstandarizada = Estandarizar.fit_transform(NuevaMat)     
    ME = pd.DataFrame(MEstandarizada, columns=NuevaMat.columns)
    ME2 = ME[:10]
    context['ME']=ME2

    #Instancia de componente PCA
    pca = PCA(n_components=None)     #Se instancia el objeto PCA    #pca=PCA(n_components=None), pca=PCA(.85)
    pca.fit(MEstandarizada)        #Se obtiene los componentes
    pcaPrint = pca.components_
    context['pca1']=pcaPrint

    #Numero componentes
    Varianza = pca.explained_variance_ratio_
    context['Var']=Varianza
    nComp = 0
    VarAc = 0
    while VarAc <= 0.85:
        nComp +=1
        VarAc = sum(Varianza[0:nComp])
        print(nComp)

    context['nComp']=nComp-1
    context['VarAc']=VarAc
    print('Varianza acumulada:', VarAc)   

    #Grafica Varianza acumulada
    figV = px.line(np.cumsum(pca.explained_variance_ratio_))
    # Setting layout of the figure.
    layout = {
        'title': 'Grafica varianza',
        'xaxis_title': 'Numero de componentes', 
        'yaxis_title': 'Varianza acumulada',
        'height': 144,
        'width': 60,
    }
    figVar = plot({'data': figV, 'layout': layout}, output_type='div')

    context['figVar']=figVar

    #Paso 6
    CargasComponentes = pd.DataFrame(abs(pca.components_[0:nComp-1]), columns=NuevaMat.columns)
    context['CargasC']=CargasComponentes   

    return render(request, 'PCA.html', context)

def EDA(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    source = proyecto.data

    #Comienzo del algoritmo
    df = pd.read_csv(source)
    df2 = df[:10]
    
    #Forma del df
    size = df.shape
    
    #Tipos de datos
    tipos = []
    for i in range(df.shape[1]):
        column = df.columns.values[i]
        value = df[column].dtype
        tipos.append(str(column) + ': ' + str(value))
    
    #Valores nulos
    nulos = []
    for i in range(df.shape[1]):
        column = df.columns.values[i]
        value = df[column].isnull().sum()
        nulos.append(str(column) + ': ' + str(value))

    #Funcion info
    #*PENDIENTE*

    #Resumen estadistico de variables numericas
    df3 = df.describe()

    #Histogramas
    histogramas = []
    for i in range(df.shape[1]):
        dataType = df.columns.values[i]
        if df[dataType].dtype != object:
            fig = px.histogram(df, x=df.columns[i])

            # Setting layout of the figure.
            layout = {
                'title': df.columns[i],
                'xaxis_title': 'X', 
                'yaxis_title': 'Y',
                'height': 144,
                'width': 60,
            }
            # Getting HTML needed to render the plot.
            plot_div = plot({'data': fig, 'layout': layout}, 
                            output_type='div')
            histogramas.append(plot_div)
    
    #Diagramas de caja
    cajas = []
    for i in range(df.shape[1]):
        dataType = df.columns.values[i]
        if df[dataType].dtype != object:
            fig = px.box(df, x=df.columns[i])

            # Setting layout of the figure.
            layout = {
                'title': df.columns[i],
                'xaxis_title': 'X', 
                'yaxis_title': 'Y',
                'height': 240,
                'width': 240,
            }
            # Getting HTML needed to render the plot.
            plot_div = plot({'data': fig, 'layout': layout}, 
                            output_type='div')
            cajas.append(plot_div)
    
    #Verificar que el dataframe contenga variables no numericas
    try:
        df.describe(include='object')
    except:
        objects = False
    else:
        objects = True
    
    #Toma de decision en caso de haber variables no numericas
    if(objects == True):
        #Distribucion variables categoricas
        df4 = df.describe(include='object')
        #Plots de las distribuciones
        Cat = []
        for col in df.select_dtypes(include='object'):
            if df[col].nunique()< 10:
                fig = px.histogram(df, y=col)
                # Setting layout of the figure.
                layout = {
                    'title': col,
                    'height': 240,
                    'width': 240,
                }
                # Getting HTML needed to render the plot.
                plot_div = plot({'data': fig, 'layout': layout}, 
                                output_type='div')
                Cat.append(plot_div)
        
        #Agrupacion por variables categoricas
        groups = []
        for col in df.select_dtypes(include='object'):
            if df[col].nunique() < 10:
                dataG = df.groupby(col).agg(['mean']).reset_index()
                print(dataG)
                groups.append(dataG)
        
        context={'plot_div': histogramas, 'df': df2, 'size' : size, 'diagramsCaja' : cajas, 'tipos': tipos, 'nulos': nulos, 'df3': df3, 'Cat' : Cat, 'df4': df4, 'flag': objects, 'groups': groups}
    else:
        context={'plot_div': histogramas, 'df': df2, 'size' : size, 'diagramsCaja' : cajas, 'tipos': tipos, 'nulos': nulos, 'df3': df3, 'flag': objects, 'mapaC': mapaC}
    
    #Correlaciones
    correlaciones = df.corr()
    #Mapa de calor de correlaciones
    calor = px.imshow(correlaciones, text_auto=True, aspect="auto")
    layout_corr = {
    'title': proyecto.Nombre,
    'height': 240,
    'width': 240,
    }
    mapaC = plot({'data': calor, 'layout': layout_corr}, output_type='div')
    context['mapaC'] = mapaC
    return render(request, 'EDA.html', context)