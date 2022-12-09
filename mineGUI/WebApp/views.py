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
    
    return render(request, 'crea_proyecto.html', { 'form': form})

def delete_project(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    proyecto.delete()
    return redirect('project_list')

#Vistas de algoritmos
def preEDA(request):
    proyectos = Proyecto.objects.all()
    return render(request, 'EligeEDA.html', {'proyectos': proyectos})

def prePCA(request):
    proyectos = Proyecto.objects.all()
    return render(request, 'EligePCA.html', {'proyectos': proyectos})

def preAB(request):
    proyectos = Proyecto.objects.all()
    return render(request, 'EligeAB.html', {'proyectos': proyectos})

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
    Estandarizar = StandardScaler()
    NuevaMatriz = df.drop(columns=df.select_dtypes('object'))
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

    #Grafica Varianza acumulada
    figV = px.line(np.cumsum(pca.explained_variance_ratio_))
    figV.update_xaxes(title_text='Numero de componentes')
    figV.update_yaxes(title_text='Varianza acumulada')
    # Setting layout of the figure.
    layout = {
        'title': 'Grafica varianza',
        'height': 144,
        'width': 60,
    }
    figVar = plot({'data': figV, 'layout': layout}, output_type='div')

    context['figVar']=figVar

    #Paso 6
    CargasComponentes = pd.DataFrame(abs(pca.components_[0:nComp-1]), columns=NuevaMat.columns)
    context['CargasC']=CargasComponentes

    muestra = 0.50
    n_df = df
    for i in range(CargasComponentes.shape[1]):
        column = CargasComponentes.columns.values[i]
        if( np.any(CargasComponentes[column].values > muestra) == False ):
            n_df = n_df.drop(columns=[column])
    print_ndf = n_df[:10]
    context['ndf']=print_ndf

    size2 = n_df.shape
    context['sizeNdf']=size2 

    return render(request, 'PCA.html', context)

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

            # Setting layout of the figure.
            layout = {
                'title': df.columns[i],
                'xaxis_title': 'X', 
                'yaxis_title': 'Y',
                'height': 144,
                'width': 60,
            }
            
            plot_div = plot({'data': fig, 'layout': layout}, output_type='div')
            histogramas.append(plot_div)

    context['plot_div'] = histogramas

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
            
            plot_div = plot({'data': fig, 'layout': layout}, output_type='div')
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
                # Setting layout of the figure.
                layout = {
                    'title': col,
                    'height': 240,
                    'width': 240,
                }
                
                plot_div = plot({'data': fig, 'layout': layout}, output_type='div')
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
    layout_corr = {
    'title': proyecto.Nombre,
    'height': 240,
    'width': 240,
    }
    mapaC = plot({'data': calor, 'layout': layout_corr}, output_type='div')
    context['mapaC'] = mapaC
    return render(request, 'EDA.html', context)

def AB_P(request, pk):
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

    return render(request, 'AB_P.html', context)

def AB_C(request, pk):
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

    return render(request, 'AB_C.html', context)

#Vistas con ideas antiguas
def busqueda(request):
    return render(request, 'Busqueda.html')
