from django.shortcuts import render, redirect
from pathlib import Path
import os

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn import model_selection

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.tree import export_text

from .models import Proyecto

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

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

#Vistas previas de algoritmos
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

#Algoritmos
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
    context['pk'] = pk
    df = pd.read_csv(source)
    nDf = df.drop(columns=colDrop)
    context['ndf']=nDf[:10]
    #Forma del nuevo df
    size = nDf.shape
    context['size']=size
    return render(request, 'PCA/PCA2.html', context)


def AD_P(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    source = proyecto.data
    context = {}
    context['pk'] = pk
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
    
    #Limpieza de datos categoricos
    NuevaMatriz = df.drop(columns=df.select_dtypes('object'))
    ME = NuevaMatriz[:10]
    context['ME']=ME

    #Correlaciones
    correlaciones = df.corr()
    #Mapa de calor de correlaciones
    calor = px.imshow(correlaciones, text_auto=True, aspect="auto")

    mapaC = plot({'data': calor}, output_type='div')
    context['corr']=correlaciones
    context['mapaC'] = mapaC

    return render(request, 'Arboles/AD_P.html', context)

def AD_P_2(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    context = {}
    context['pk'] = pk
    #Obtencion de Var Cat y Pred
    predictoras = request.POST.getlist('predictora')
    pronosticar = request.POST['pronostico']
    
    #Paso usual
    source = proyecto.data
    df = pd.read_csv(source)
    
    #Limpiamos de nuevo Xd'nt
    #NuevaMatriz = df.drop(columns=df.select_dtypes('object'))
    NuevaMat = df.dropna() 


    #Seleccion variables Predictoras y de pronostico
    aux = pd.DataFrame(NuevaMat[predictoras])
    X = np.array(NuevaMat[predictoras])
    Xout = pd.DataFrame(data=X, columns=aux.columns.values)
    Xout.to_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/X.csv'), index=False)
    context['X'] = Xout[:10]

    Y = np.array(NuevaMat[pronosticar])
    Yout = pd.DataFrame(Y)
    
    Yout.to_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/Y.csv'), index=False)

    
    context['Y'] = Yout[:10]

    #Division de los datos
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                        test_size = 0.2, 
                                                                        random_state = 0, 
                                                                        shuffle = True)

    #Visualizacion de datos de prueba
    Xtest = pd.DataFrame(X_test)
    context['Xtest'] = Xtest[:10]

    #Entrenamiento
    PronosticoAD = DecisionTreeRegressor(random_state=0)
    PronosticoAD.fit(X_train, Y_train)

    #Se genera el pronóstico
    Y_Pronostico = PronosticoAD.predict(X_test)
    Ypronostico = pd.DataFrame(Y_Pronostico)
    context['YPron'] = Ypronostico[:10]

    #Comparacion entre pronostico y prueba
    Valores = pd.DataFrame(Y_test, Y_Pronostico)
    Valores2 = Valores.reset_index()
    ValoresOut = Valores2.rename(columns={Valores2.columns[0]: 'Prueba', Valores2.columns[1]: 'Pronostico'})
    #print(ValoresOut)
    context['Valores'] = ValoresOut[:10]

    #Obtencion del ajuste de Bondad
    Score = r2_score(Y_test, Y_Pronostico)
    context['Score'] = Score

    #Criterios
    criterios = []
    criterios.append(PronosticoAD.criterion)
    criterios.append(mean_absolute_error(Y_test, Y_Pronostico))
    criterios.append(mean_squared_error(Y_test, Y_Pronostico))
    criterios.append(mean_squared_error(Y_test, Y_Pronostico, squared=False))
    context['criterios'] = criterios
    

    #Dataframe de la importancia de variables
    Importancia = pd.DataFrame({'Variable': list(NuevaMat[predictoras]),
                            'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
    context['Imp'] = Importancia
    print(Importancia)

    #Reporte o arbol en texto
    Reporte = export_text(PronosticoAD, feature_names = predictoras)
    RepOut = []
    RepOut = Reporte.split("\n")
    context['reportes'] = RepOut

    #Eleccion para nuevo pronostico
    predictorasOut = NuevaMat[predictoras]
    context['Pred'] = predictorasOut[:10]


    return render(request, 'Arboles/AD_P-2.html', context)

def AD_P_3(request,pk):
    proyecto = Proyecto.objects.get(pk=pk)
    context = {}
    context['pk'] = pk
    Val = request.POST.getlist('NPron')
    X = pd.read_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/X.csv'))
    Y = pd.read_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/Y.csv'))

    #Division de los datos
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                        test_size = 0.2, 
                                                                        random_state = 0, 
                                                                        shuffle = True)
    
    #Entrenamiento
    PronosticoAD = DecisionTreeRegressor(random_state=0)
    PronosticoAD.fit(X_train, Y_train)

    col = list(X.columns)
    datoOut = {}
    for i in range(X.shape[1]):
        datoOut[col[i]] = int(Val[i])
    Npron = pd.DataFrame(datoOut, index=[0])
    context['DfN'] = Npron
    resultado = PronosticoAD.predict(Npron)
    print(resultado)
    context['resultado'] = resultado

    return render(request, 'Arboles/AD_P-3.html', context)


def AD_C(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    source = proyecto.data
    context = {}
    context['pk'] = pk
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
    
    #Limpieza de datos categoricos
    NuevaMatriz = df.drop(columns=df.select_dtypes('object'))
    ME = NuevaMatriz[:10]
    context['ME']=ME

    #Correlaciones
    correlaciones = df.corr()
    #Mapa de calor de correlaciones
    calor = px.imshow(correlaciones, text_auto=True, aspect="auto")

    mapaC = plot({'data': calor}, output_type='div')
    context['corr']=correlaciones
    context['mapaC'] = mapaC

    return render(request, 'Arboles/AD_C.html', context)

def AD_C_2(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    context = {}
    context['pk'] = pk
    #Obtencion de Var Cat y Pred
    predictoras = request.POST.getlist('predictora')
    pronosticar = request.POST['pronostico']
    
    #Paso usual
    source = proyecto.data
    df = pd.read_csv(source)
    
    #Limpiamos de nuevo Xd'nt
    #NuevaMatriz = df.drop(columns=df.select_dtypes('object'))
    NuevaMat = df.dropna() 


    #Seleccion variables Predictoras y de pronostico
    aux = pd.DataFrame(NuevaMat[predictoras])
    X = np.array(NuevaMat[predictoras])
    Xout = pd.DataFrame(data=X, columns=aux.columns.values)
    Xout.to_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/X.csv'), index=False)
    context['X'] = Xout[:10]

    Y = np.array(NuevaMat[pronosticar])
    Yout = pd.DataFrame(Y)
    
    Yout.to_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/Y.csv'), index=False)

    
    context['Y'] = Yout[:10]

    #Division de los datos
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)


    #Entrenamiento
    ClasificacionAD = DecisionTreeClassifier(random_state=0)
    ClasificacionAD.fit(X_train, Y_train)

    #Se genera el pronóstico
    Y_ClasificacionAD = ClasificacionAD.predict(X_validation)
    print(Y_ClasificacionAD)

    #Comparacion entre pronostico y prueba
    Valores = pd.DataFrame(Y_validation, Y_ClasificacionAD)
    Valores2 = Valores.reset_index()
    ValoresOut = Valores2.rename(columns={Valores2.columns[0]: 'Prueba', Valores2.columns[1]: 'Pronostico'})
    #print(ValoresOut)
    context['Valores'] = ValoresOut[:10]

    #Obtencion del ajuste de Bondad
    Score = accuracy_score(Y_validation, Y_ClasificacionAD)
    context['Score'] = Score

    #Matriz de clasificacion
    ModeloClasificacion1 = ClasificacionAD.predict(X_validation)
    Matriz_Clasificacion1 = pd.crosstab(Y_validation.ravel(), 
                                   ModeloClasificacion1, 
                                   rownames=['Actual'], 
                                   colnames=['Clasificación']) 
    print(Matriz_Clasificacion1)
    context['MClas']=Matriz_Clasificacion1

    #Criterios
    criterios = []
    criterios.append(ClasificacionAD.criterion)
    criterios.append(accuracy_score(Y_validation, Y_ClasificacionAD))
    context['criterios'] = criterios
    ReporteC =classification_report(Y_validation, Y_ClasificacionAD, output_dict=True)
    RepClas = pd.DataFrame(ReporteC).transpose()
    context['ReporteClas'] = RepClas

    #Dataframe de la importancia de variables
    Importancia = pd.DataFrame({'Variable': list(NuevaMat[predictoras]),
                            'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
    context['Imp'] = Importancia
    print(Importancia)

    #Reporte o arbol en texto
    Reporte = export_text(ClasificacionAD, feature_names = predictoras)
    RepOut = []
    RepOut = Reporte.split("\n")
    context['reportes'] = RepOut

    #Eleccion para nuevo pronostico
    predictorasOut = NuevaMat[predictoras]
    context['Pred'] = predictorasOut[:10]


    return render(request, 'Arboles/AD_C-2.html', context)

def AD_C_3(request,pk):
    proyecto = Proyecto.objects.get(pk=pk)
    context = {}
    context['pk'] = pk
    Val = request.POST.getlist('NClas')
    X = pd.read_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/X.csv'))
    Y = pd.read_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/Y.csv'))

    #Division de los datos
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)


    #Entrenamiento
    ClasificacionAD = DecisionTreeClassifier(random_state=0)
    ClasificacionAD.fit(X_train, Y_train)

    col = list(X.columns)
    datoOut = {}
    for i in range(X.shape[1]):
        datoOut[col[i]] = int(Val[i])
    Npron = pd.DataFrame(datoOut, index=[0])
    context['DfN'] = Npron
    resultado = ClasificacionAD.predict(Npron)
    print(resultado)
    context['resultado'] = resultado

    return render(request, 'Arboles/AD_C-3.html', context)


def BA_P(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    source = proyecto.data
    context = {}
    context['pk'] = pk
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
    
    #Limpieza de datos categoricos
    NuevaMatriz = df.drop(columns=df.select_dtypes('object'))
    ME = NuevaMatriz[:10]
    context['ME']=ME

    #Correlaciones
    correlaciones = df.corr()
    #Mapa de calor de correlaciones
    calor = px.imshow(correlaciones, text_auto=True, aspect="auto")

    mapaC = plot({'data': calor}, output_type='div')
    context['corr']=correlaciones
    context['mapaC'] = mapaC

    return render(request, 'Bosques/BA_P.html', context)

def BA_P_2(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    context = {}
    context['pk'] = pk
    #Obtencion de Var Cat y Pred
    predictoras = request.POST.getlist('predictora')
    pronosticar = request.POST['pronostico']
    
    #Paso usual
    source = proyecto.data
    df = pd.read_csv(source)
    
    #Limpiamos de nuevo Xd'nt
    #NuevaMatriz = df.drop(columns=df.select_dtypes('object'))
    NuevaMat = df.dropna() 


    #Seleccion variables Predictoras y de pronostico
    aux = pd.DataFrame(NuevaMat[predictoras])
    X = np.array(NuevaMat[predictoras])
    Xout = pd.DataFrame(data=X, columns=aux.columns.values)
    Xout.to_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/X.csv'), index=False)
    context['X'] = Xout[:10]

    Y = np.array(NuevaMat[pronosticar])
    Yout = pd.DataFrame(Y)
    
    Yout.to_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/Y.csv'), index=False)

    
    context['Y'] = Yout[:10]

    #Division de los datos
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                        test_size = 0.2, 
                                                                        random_state = 0, 
                                                                        shuffle = True)

    #Visualizacion de datos de prueba
    Xtest = pd.DataFrame(X_test)
    context['Xtest'] = Xtest[:10]

    #Entrenamiento
    PronosticoBA = RandomForestRegressor(random_state=0)
    PronosticoBA.fit(X_train, Y_train)

    #Se genera el pronóstico
    Y_Pronostico = PronosticoBA.predict(X_test)
    Ypronostico = pd.DataFrame(Y_Pronostico)
    context['YPron'] = Ypronostico[:10]

    #Comparacion entre pronostico y prueba
    Valores = pd.DataFrame(Y_test, Y_Pronostico)
    Valores2 = Valores.reset_index()
    ValoresOut = Valores2.rename(columns={Valores2.columns[0]: 'Prueba', Valores2.columns[1]: 'Pronostico'})
    #print(ValoresOut)
    context['Valores'] = ValoresOut[:10]

    #Obtencion del ajuste de Bondad
    Score = r2_score(Y_test, Y_Pronostico)
    context['Score'] = Score

    #Criterios
    criterios = []
    criterios.append(PronosticoBA.criterion)
    criterios.append(mean_absolute_error(Y_test, Y_Pronostico))
    criterios.append(mean_squared_error(Y_test, Y_Pronostico))
    criterios.append(mean_squared_error(Y_test, Y_Pronostico, squared=False))
    context['criterios'] = criterios
    

    #Dataframe de la importancia de variables
    Importancia = pd.DataFrame({'Variable': list(NuevaMat[predictoras]),
                            'Importancia': PronosticoBA.feature_importances_}).sort_values('Importancia', ascending=False)
    context['Imp'] = Importancia
    print(Importancia)

    #Reporte o arbol en texto
    Estimador = PronosticoBA.estimators_[50]
    
    Reporte = export_text(Estimador, feature_names = predictoras)
    RepOut = []
    RepOut = Reporte.split("\n")
    context['reportes'] = RepOut

    #Eleccion para nuevo pronostico
    predictorasOut = NuevaMat[predictoras]
    context['Pred'] = predictorasOut[:10]


    return render(request, 'Bosques/BA_P-2.html', context)

def BA_P_3(request,pk):
    proyecto = Proyecto.objects.get(pk=pk)
    context = {}
    context['pk'] = pk
    Val = request.POST.getlist('NPron')
    X = pd.read_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/X.csv'))
    Y = pd.read_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/Y.csv'))

    #Division de los datos
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                        test_size = 0.2, 
                                                                        random_state = 0, 
                                                                        shuffle = True)
    
    #Entrenamiento
    PronosticoAD = RandomForestRegressor(random_state=0)
    PronosticoAD.fit(X_train, Y_train)

    col = list(X.columns)
    datoOut = {}
    for i in range(X.shape[1]):
        datoOut[col[i]] = int(Val[i])
    Npron = pd.DataFrame(datoOut, index=[0])
    context['DfN'] = Npron
    resultado = PronosticoAD.predict(Npron)
    print(resultado)
    context['resultado'] = resultado

    return render(request, 'Bosques/BA_P-3.html', context)


def BA_C(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    source = proyecto.data
    context = {}
    context['pk'] = pk
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
    
    #Limpieza de datos categoricos
    NuevaMatriz = df.drop(columns=df.select_dtypes('object'))
    ME = NuevaMatriz[:10]
    context['ME']=ME

    #Correlaciones
    correlaciones = df.corr()
    #Mapa de calor de correlaciones
    calor = px.imshow(correlaciones, text_auto=True, aspect="auto")

    mapaC = plot({'data': calor}, output_type='div')
    context['corr']=correlaciones
    context['mapaC'] = mapaC


    return render(request, 'Bosques/BA_C.html', context)

def BA_C_2(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    context = {}
    context['pk'] = pk
    #Obtencion de Var Cat y Pred
    predictoras = request.POST.getlist('predictora')
    pronosticar = request.POST['pronostico']
    
    #Paso usual
    source = proyecto.data
    df = pd.read_csv(source)
    
    #Limpiamos de nuevo Xd'nt
    #NuevaMatriz = df.drop(columns=df.select_dtypes('object'))
    NuevaMat = df.dropna() 


    #Seleccion variables Predictoras y de pronostico
    aux = pd.DataFrame(NuevaMat[predictoras])
    X = np.array(NuevaMat[predictoras])
    Xout = pd.DataFrame(data=X, columns=aux.columns.values)
    Xout.to_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/X.csv'), index=False)
    context['X'] = Xout[:10]

    Y = np.array(NuevaMat[pronosticar])
    Yout = pd.DataFrame(Y)
    
    Yout.to_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/Y.csv'), index=False)

    
    context['Y'] = Yout[:10]

    #Division de los datos
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)


    #Entrenamiento
    ClasificacionBA = RandomForestClassifier(random_state=0)
    ClasificacionBA.fit(X_train, Y_train)

    #Se genera el pronóstico
    Y_ClasificacionAD = ClasificacionBA.predict(X_validation)
    print(Y_ClasificacionAD)

    #Comparacion entre pronostico y prueba
    Valores = pd.DataFrame(Y_validation, Y_ClasificacionAD)
    Valores2 = Valores.reset_index()
    ValoresOut = Valores2.rename(columns={Valores2.columns[0]: 'Prueba', Valores2.columns[1]: 'Pronostico'})
    #print(ValoresOut)
    context['Valores'] = ValoresOut[:10]

    #Obtencion del ajuste de Bondad
    Score = accuracy_score(Y_validation, Y_ClasificacionAD)
    context['Score'] = Score

    #Matriz de clasificacion
    ModeloClasificacion1 = ClasificacionBA.predict(X_validation)
    Matriz_Clasificacion1 = pd.crosstab(Y_validation.ravel(), 
                                   ModeloClasificacion1, 
                                   rownames=['Actual'], 
                                   colnames=['Clasificación']) 
    print(Matriz_Clasificacion1)
    context['MClas']=Matriz_Clasificacion1

    #Criterios
    criterios = []
    criterios.append(ClasificacionBA.criterion)
    criterios.append(accuracy_score(Y_validation, Y_ClasificacionAD))
    context['criterios'] = criterios
    ReporteC =classification_report(Y_validation, Y_ClasificacionAD, output_dict=True)
    RepClas = pd.DataFrame(ReporteC).transpose()
    context['ReporteClas'] = RepClas

    #Dataframe de la importancia de variables
    Importancia = pd.DataFrame({'Variable': list(NuevaMat[predictoras]),
                            'Importancia': ClasificacionBA.feature_importances_}).sort_values('Importancia', ascending=False)
    context['Imp'] = Importancia
    print(Importancia)

    #Reporte o arbol en texto

    #Reporte o arbol en texto
    Estimador = ClasificacionBA.estimators_[50]

    Reporte = export_text(Estimador, feature_names = predictoras)
    RepOut = []
    RepOut = Reporte.split("\n")
    context['reportes'] = RepOut

    #Eleccion para nuevo pronostico
    predictorasOut = NuevaMat[predictoras]
    context['Pred'] = predictorasOut[:10]


    return render(request, 'Bosques/BA_C-2.html', context)

def BA_C_3(request,pk):
    proyecto = Proyecto.objects.get(pk=pk)
    context = {}
    context['pk'] = pk
    Val = request.POST.getlist('NClas')
    X = pd.read_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/X.csv'))
    Y = pd.read_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/Y.csv'))

    #Division de los datos
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                        test_size = 0.2, 
                                                                        random_state = 0, 
                                                                        shuffle = True)
    
    #Entrenamiento
    ClasificacionBA = RandomForestClassifier(random_state=0)
    ClasificacionBA.fit(X_train, Y_train)

    col = list(X.columns)
    datoOut = {}
    for i in range(X.shape[1]):
        datoOut[col[i]] = int(Val[i])
    Npron = pd.DataFrame(datoOut, index=[0])
    context['DfN'] = Npron
    resultado = ClasificacionBA.predict(Npron)
    print(resultado)
    context['resultado'] = resultado

    return render(request, 'Bosques/BA_C-3.html', context)


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
