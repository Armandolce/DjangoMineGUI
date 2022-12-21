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

#Vistas de Proyectos
def lista_Proyectos(request):
    proyectos = Proyecto.objects.all()
    return render(request, 'Proyectos/Proyectos.html', {
        'proyectos': proyectos
    })

def crea_Proyecto(request):
    Nombre = request.POST['Nombre']
    Desc = request.POST['descripcion']
    URL = request.POST['URL']
    data = request.FILES['data']

    Proyecto.objects.create(Nombre=Nombre, descripcion=Desc, URL = URL, data = data)
    return redirect('project_list')

def delete_project(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    proyecto.delete()
    return redirect('project_list')

#Vistas de algoritmos
def preEDA(request):
    proyectos = Proyecto.objects.all()
    return render(request, 'EDA/EligeEDA.html', {'proyectos': proyectos})

def prePCA(request):
    proyectos = Proyecto.objects.all()
    return render(request, 'PCA/EligePCA.html', {'proyectos': proyectos})

def preAD(request):
    proyectos = Proyecto.objects.all()
    return render(request, 'Arboles/EligeArb.html', {'proyectos': proyectos})

def preBA(request):
    proyectos = Proyecto.objects.all()
    return render(request, 'Bosques/EligeBos.html', {'proyectos': proyectos})


def PCA_1(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    source = proyecto.data
    context = {}
    context['pk'] = pk
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

    mapaC = plot({'data': calor}, output_type='div')
    context['corr']=correlaciones
    context['mapaC'] = mapaC

    #Estandarizacion de datos
    Estandarizar = StandardScaler()
    NuevaMatriz = df.drop(columns=df.select_dtypes('object'))
    NuevaMat = NuevaMatriz.dropna()
    MEstandarizada = Estandarizar.fit_transform(NuevaMat)     
    ME = pd.DataFrame(MEstandarizada, columns=NuevaMat.columns)
    ME2 = ME[:10]
    context['ME']=ME2

    #Instancia de componente PCA
    pca = PCA(n_components=None)
    pca.fit(MEstandarizada)     
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

    #Grafica Varianza acumulada
    figV = px.line(np.cumsum(pca.explained_variance_ratio_))
    figV.update_xaxes(title_text='Numero de componentes')
    figV.update_yaxes(title_text='Varianza acumulada')
    figVar = plot({'data': figV}, output_type='div')

    context['figVar']=figVar

    #Paso 6
    CargasComponentes = pd.DataFrame(abs(pca.components_[0:nComp-1]), columns=NuevaMat.columns)
    context['CargasC']=CargasComponentes

    return render(request, 'PCA/PCA.html', context)

def PCA_2(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    colDrop = request.POST.getlist('columnas')
    source = proyecto.data
    context = {}
    df = pd.read_csv(source)
    nDf = df.drop(columns=colDrop)
    context['ndf']=nDf[:10]
    #Forma del nuevo df
    size = nDf.shape
    context['size']=size
    return render(request, 'PCA/PCA2.html', context)

def EDA(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    source = proyecto.data
    context = {}
    
    #Comienzo del algoritmo
    df = pd.read_csv(source)
    df2 = df[:10]
    context['df'] = df2
    
    #Forma del df
    size = df.shape
    context['size'] = size
    
    #Tipos de datos
    tipos = []
    for i in range(df.shape[1]):
        column = df.columns.values[i]
        value = df[column].dtype
        tipos.append(str(column) + ': ' + str(value))
    context['tipos'] = tipos
    
    #Valores nulos
    nulos = []
    for i in range(df.shape[1]):
        column = df.columns.values[i]
        value = df[column].isnull().sum()
        nulos.append(str(column) + ': ' + str(value))
    context['nulos'] = nulos

    #Resumen estadistico de variables numericas
    df3 = df.describe()
    context['df3'] = df3

    #Histogramas
    histogramas = []
    for i in range(df.shape[1]):
        dataType = df.columns.values[i]
        if df[dataType].dtype != object:
            fig = px.histogram(df, x=df.columns[i])
            
            plot_div = plot({'data': fig}, output_type='div')
            histogramas.append(plot_div)

    context['plot_div'] = histogramas

    #Diagramas de caja
    cajas = []
    for i in range(df.shape[1]):
        dataType = df.columns.values[i]
        if df[dataType].dtype != object:
            fig = px.box(df, x=df.columns[i])            
            plot_div = plot({'data': fig}, output_type='div')
            cajas.append(plot_div)
    context['diagramsCaja'] = cajas

    #Verificar que el dataframe contenga variables no numericas
    try:
        df.describe(include='object')
    except:
        objects = False
    else:
        objects = True
    context['flag'] = objects

    #Toma de decision en caso de haber variables no numericas
    if(objects == True):
        #Distribucion variables categoricas
        df4 = df.describe(include='object')
        context['df4']=df4
        #Plots de las distribuciones
        Cat = []
        for col in df.select_dtypes(include='object'):
            if df[col].nunique()< 10:
                fig = px.histogram(df, y=col)
                plot_div = plot({'data': fig}, output_type='div')
                Cat.append(plot_div)

        context['Cat']=Cat

        #Agrupacion por variables categoricas
        groups = []
        for col in df.select_dtypes(include='object'):
            if df[col].nunique() < 10:
                dataG = df.groupby(col).agg(['mean'])
                print(dataG)
                groups.append(dataG)
        context['groups']=groups
    
    #Correlaciones
    correlaciones = df.corr()
    #Mapa de calor de correlaciones
    calor = px.imshow(correlaciones, text_auto=True, aspect="auto")
    mapaC = plot({'data': calor}, output_type='div')
    context['mapaC'] = mapaC
    return render(request, 'EDA/EDA.html', context)

def AD_P(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    source = proyecto.data
    context = {}
    #Comienzo del algoritmo
    df = pd.read_csv(source)
    df2 = df[:10]
    context['df'] = df2
    
    #Forma del df
    size = df.shape
    context['size'] = size
    
    #Tipos de datos
    tipos = []
    for i in range(df.shape[1]):
        column = df.columns.values[i]
        value = df[column].dtype
        tipos.append(str(column) + ': ' + str(value))
    context['tipos'] = tipos
    
    #Valores nulos
    nulos = []
    for i in range(df.shape[1]):
        column = df.columns.values[i]
        value = df[column].isnull().sum()
        nulos.append(str(column) + ': ' + str(value))
    context['nulos'] = nulos

    #Resumen estadistico de variables numericas
    df3 = df.describe()
    context['df3'] = df3
    
    #Limpieza de datos (Nulos, categoricos)
    NuevaMatriz = df.drop(columns=df.select_dtypes('object'))    # Se quitan las variables nominales
    NuevaMat = NuevaMatriz.dropna() 
    ME = NuevaMat[:10]
    context['ME']=ME

    return render(request, 'Arboles/AD_P.html', context)

def AD_C(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    source = proyecto.data
    context = {}
    #Comienzo del algoritmo
    df = pd.read_csv(source)
    df2 = df[:10]
    context['df'] = df2
    
    #Forma del df
    size = df.shape
    context['size'] = size
    
    #Tipos de datos
    tipos = []
    for i in range(df.shape[1]):
        column = df.columns.values[i]
        value = df[column].dtype
        tipos.append(str(column) + ': ' + str(value))
    context['tipos'] = tipos
    
    #Valores nulos
    nulos = []
    for i in range(df.shape[1]):
        column = df.columns.values[i]
        value = df[column].isnull().sum()
        nulos.append(str(column) + ': ' + str(value))
    context['nulos'] = nulos

    #Resumen estadistico de variables numericas
    df3 = df.describe()
    context['df3'] = df3
    
    #Estandarizacion de datos
    NuevaMatriz = df.drop(columns=df.select_dtypes('object'))    # Se quitan las variables nominales
    NuevaMat = NuevaMatriz.dropna() 
    ME = NuevaMat[:10]
    context['ME']=ME

    return render(request, 'Arboles/AD_C.html', context)

def BA_P(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    source = proyecto.data
    context = {}
    #Comienzo del algoritmo
    df = pd.read_csv(source)
    df2 = df[:10]
    context['df'] = df2
    
    #Forma del df
    size = df.shape
    context['size'] = size
    
    #Tipos de datos
    tipos = []
    for i in range(df.shape[1]):
        column = df.columns.values[i]
        value = df[column].dtype
        tipos.append(str(column) + ': ' + str(value))
    context['tipos'] = tipos
    
    #Valores nulos
    nulos = []
    for i in range(df.shape[1]):
        column = df.columns.values[i]
        value = df[column].isnull().sum()
        nulos.append(str(column) + ': ' + str(value))
    context['nulos'] = nulos

    #Resumen estadistico de variables numericas
    df3 = df.describe()
    context['df3'] = df3
    
    #Estandarizacion de datos
    NuevaMatriz = df.drop(columns=df.select_dtypes('object'))    # Se quitan las variables nominales
    NuevaMat = NuevaMatriz.dropna() 
    ME = NuevaMat[:10]
    context['ME']=ME

    return render(request, 'Bosques/BA_P.html', context)

def BA_C(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    source = proyecto.data
    context = {}
    #Comienzo del algoritmo
    df = pd.read_csv(source)
    df2 = df[:10]
    context['df'] = df2
    
    #Forma del df
    size = df.shape
    context['size'] = size
    
    #Tipos de datos
    tipos = []
    for i in range(df.shape[1]):
        column = df.columns.values[i]
        value = df[column].dtype
        tipos.append(str(column) + ': ' + str(value))
    context['tipos'] = tipos
    
    #Valores nulos
    nulos = []
    for i in range(df.shape[1]):
        column = df.columns.values[i]
        value = df[column].isnull().sum()
        nulos.append(str(column) + ': ' + str(value))
    context['nulos'] = nulos

    #Resumen estadistico de variables numericas
    df3 = df.describe()
    context['df3'] = df3
    
    #Estandarizacion de datos
    NuevaMatriz = df.drop(columns=df.select_dtypes('object'))    # Se quitan las variables nominales
    NuevaMat = NuevaMatriz.dropna() 
    ME = NuevaMat[:10]
    context['ME']=ME

    return render(request, 'Bosques/BA_C.html', context)


#Vistas con ideas antiguas o pruebas
def busqueda(request):
    proyectoP = Proyecto.objects.get(pk=18)
    source = proyectoP.data
    context = {}
    #Comienzo del algoritmo
    df = pd.read_csv(source)
    df2 = df[:10]
    context['df']=df2
    return render(request, 'Busqueda.html', context)
