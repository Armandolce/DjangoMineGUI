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

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from sklearn.svm import SVC                         #Support vector classifier
from sklearn.metrics import RocCurveDisplay

from .models import Proyecto


# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

#Funcion encargada de los primeros procesos de la mayoria de algoritmos.
def showDatos(pk):
    proyecto = Proyecto.objects.get(pk=pk)
    source = proyecto.data
    context = {}
    context['pk'] = pk
    #Comienzo del algoritmo
    df = pd.read_csv(source)
    for i in range(df.shape[1]):
        df.columns.values[i] = df.columns.values[i].replace(" ","_")
    
    df2 = df.iloc[np.r_[0:5, -5:0]]
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
    ME = NuevaMatriz.iloc[np.r_[0:5, -5:0]]
    context['ME']=ME

    #Correlaciones
    correlaciones = df.corr()
    mask = np.triu(np.ones_like(correlaciones, dtype=np.bool))
    correlaciones = correlaciones.mask(mask)
    #Mapa de calor de correlaciones
    calor = px.imshow(correlaciones, text_auto=True, aspect="auto")

    mapaC = plot({'data': calor}, output_type='div')
    context['corr']=correlaciones
    context['mapaC'] = mapaC

    return(context) 

# Create your views here.
def home(request):
    return render(request,'home.html')

#######################
# Vistas de Proyectos #
#######################

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
    
    if Desc == '':
        Desc = "Descripcion no proporcionada."

    ext = os.path.splitext(data.name)[1]
    print(ext)
    valid_extensions = '.csv'
    if not ext.lower() in valid_extensions:
        return redirect('/ErrorProyecto')
    else:
        Proyecto.objects.create(Nombre=Nombre, descripcion=Desc, URL = URL, data = data)
        return redirect('project_list')

def delete_project(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    proyecto.delete()
    return redirect('project_list')

def ErrorProyecto(request):
    proyectos = Proyecto.objects.all()
    return render(request, 'Proyectos/ProyectosError.html', {
        'proyectos': proyectos
    })

##############
# Algoritmos #
##############

#Vista previa de algoritmos
def selector(request, Alg):
    context = {}
    Flag = False
    FlagSVM = False
    
    if Alg == 'SC':
        AlgSC = 'Segmentacion y Clasificacion'
        context['AlgN'] = AlgSC
        context['Alg'] = Alg
    else:
        context['AlgN'] = Alg
        context["Alg"] = Alg
    
    if Alg  == 'AD':
        AlgAB = 'Arboles de decision'
        context['AlgN'] = AlgAB
        Flag = True
    elif Alg == 'BA':
        AlgBA = 'Bosques Aleatorios'
        context['AlgN'] = AlgBA
        Flag = True
    else:
        Flag = False
    
    if Alg  == 'SVM':
        AlgAB = 'Maquinas de soporte vectorial'
        context['AlgN'] = AlgAB
        FlagSVM = True
    else:
        FlagSVM = False

    context['flag'] = Flag
    context['flagSVM'] = FlagSVM
    proyectos = Proyecto.objects.all()

    context['proyectos'] = proyectos
    return render(request, "Selector.html", context)

#EDA
def EDA(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    source = proyecto.data
    context = {}
    
    #Comienzo del algoritmo
    df = pd.read_csv(source)
    df2 = df.iloc[np.r_[0:5, -5:0]]
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
    mask = np.triu(np.ones_like(correlaciones, dtype=np.bool))
    correlaciones = correlaciones.mask(mask)
    #Mapa de calor de correlaciones
    calor = px.imshow(correlaciones, text_auto=True, aspect="auto")
    mapaC = plot({'data': calor}, output_type='div')
    context['mapaC'] = mapaC
    return render(request, 'EDA/EDA.html', context)

#PCA
def PCA_1(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    source = proyecto.data
    context = {}
    context['pk'] = pk
    #Comienzo del algoritmo
    df = pd.read_csv(source)
    for i in range(df.shape[1]):
        df.columns.values[i] = df.columns.values[i].replace(" ","_")
    df2 = df.iloc[np.r_[0:5, -5:0]]
    context['df']=df2
    
    #Forma del df
    size = df.shape
    context['size']=size
    
    #Correlaciones
    correlaciones = df.corr()
    mask = np.triu(np.ones_like(correlaciones, dtype=np.bool))
    correlaciones = correlaciones.mask(mask)
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
    ME2 = ME.iloc[np.r_[0:5, -5:0]]
    context['ME']=ME2

    #Instancia de componente PCA
    PCAOut = []
    pca = PCA(n_components=None)
    pca.fit(MEstandarizada)     
    pcaPrint = pca.components_
    for data in pcaPrint:
        PCAOut.append(data)
    context['pca1'] = PCAOut

    #Numero componentes
    Varianza = pca.explained_variance_ratio_
    context['Var']=Varianza
    nComp = 1
    VarAc = 0
    while True:
        VarAc = sum(Varianza[0:nComp])
        if VarAc < 0.9:
            nComp +=1
            print(nComp)
        else:
            if VarAc > 0.9:
                VarAc= sum(Varianza[0:nComp-1])
                nComp -=1
            break
    
    context['nComp']=nComp
    context['VarAc']=VarAc

    #Grafica Varianza acumulada
    figV = px.line(np.cumsum(pca.explained_variance_ratio_), markers=True)
    figV.update_xaxes(title_text='Numero de componentes')
    figV.update_yaxes(title_text='Varianza acumulada')
    figVar = plot({'data': figV}, output_type='div')

    context['figVar']=figVar

    #Paso 6
    CargasComponentes = pd.DataFrame(abs(pca.components_[0:nComp]), columns=NuevaMat.columns)
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
    outDF = nDf.iloc[np.r_[0:5, -5:0]]
    context['ndf']=outDF
    #Forma del nuevo df
    size = nDf.shape
    context['size']=size
    return render(request, 'PCA/PCA2.html', context)

#Arboles
def AD(request, pk, algType):
    context = showDatos(pk)
    flag = bool
    context['type'] = algType

    match algType:
        case 'P':
            context['AlgName'] = 'Pronostico'
            flag = True
        case 'C':
            context['AlgName'] = 'Clasificacion'
            flag = False
    context['flag'] = flag
    return render(request, 'Arboles/AD.html', context)

def AD2(request, pk, algType):
    proyecto = Proyecto.objects.get(pk=pk)
    context = {}
    context['pk'] = pk
    context['type'] = algType
    flag = bool
    #Obtencion de Var Cat y Pred
    predictoras = request.POST.getlist('predictora')
    pronosticar = request.POST['pronostico']
    
    #verificar que no hayas elegido la misma columna en ambos
    for i in range (len(predictoras)):
        if(predictoras[i] == pronosticar):
            return redirect('/ADError/{}/{}'.format(pk,algType))
    
    if( len(predictoras) <= 0):
        return redirect('/ADError/{}/{}'.format(pk,algType))
    #Paso usual
    source = proyecto.data
    df = pd.read_csv(source) 
    for i in range(df.shape[1]):
        df.columns.values[i] = df.columns.values[i].replace(" ","_")   
    
    #Limpiamos de nuevo
    NuevaMat = df.dropna() 


    #Seleccion variables Predictoras y de pronostico
    aux = pd.DataFrame(NuevaMat[predictoras])
    X = np.array(NuevaMat[predictoras])
    Xout = pd.DataFrame(data=X, columns=aux.columns.values)
    Xout.to_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/X.csv'), index=False)
    context['X'] = Xout.iloc[np.r_[0:5, -5:0]]

    Y = np.array(NuevaMat[pronosticar])
    Yout = pd.DataFrame(Y)
    
    Yout.to_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/Y.csv'), index=False)
    context['Y'] = Yout.iloc[np.r_[0:5, -5:0]]

    #Division de los datos
    X_train, X_dos, Y_train, Y_dos = model_selection.train_test_split(X, Y, 
                                                                                    test_size = 0.2, 
                                                                                    random_state = 0,
                                                                                    shuffle = True)
    match algType:
        case 'P':
            context['AlgName'] = 'Pronostico'
            #Entrenamiento
            ModeloAD = DecisionTreeRegressor(random_state=0)
            
            #Visualizacion de datos de prueba
            Xtest = pd.DataFrame(X_dos)
            context['Xtest'] = Xtest.iloc[np.r_[0:5, -5:0]]
            
            flag = True
            
        case 'C':
            context['AlgName'] = 'Clasificacion'
            #Entrenamiento
            ModeloAD = DecisionTreeClassifier(random_state=0)
            flag = False

    context['flag'] = flag
    
    ModeloAD.fit(X_train, Y_train)
    
    #Se genera el pron??stico
    Y_Pronostico = ModeloAD.predict(X_dos)
    Ypronostico = pd.DataFrame(Y_Pronostico)
    context['YPron'] = Ypronostico.iloc[np.r_[0:5, -5:0]]

    #Comparacion entre pronostico y prueba
    Valores = pd.DataFrame(Y_dos, Y_Pronostico)
    Valores2 = Valores.reset_index()
    ValoresOut = Valores2.rename(columns={Valores2.columns[0]: 'Prueba', Valores2.columns[1]: 'Pronostico'})
    context['Valores'] = ValoresOut.iloc[np.r_[0:5, -5:0]]

    match algType:
        case 'P':

            #Obtencion del ajuste de Bondad
            Score = r2_score(Y_dos, Y_Pronostico)
            context['Score'] = Score

            #Criterios
            criterios = []
            criterios.append(ModeloAD.criterion)
            criterios.append(mean_absolute_error(Y_dos, Y_Pronostico))
            criterios.append(mean_squared_error(Y_dos, Y_Pronostico))
            criterios.append(mean_squared_error(Y_dos, Y_Pronostico, squared=False))
            context['criterios'] = criterios

        case 'C':

            #Obtencion del ajuste de Bondad
            Score = accuracy_score(Y_dos, Y_Pronostico)
            context['Score'] = Score

            #Matriz de clasificacion
            ModeloClasificacion1 = ModeloAD.predict(X_dos)
            Matriz_Clasificacion1 = pd.crosstab(Y_dos.ravel(), 
                                        ModeloClasificacion1, 
                                        rownames=['Actual'], 
                                        colnames=['Clasificaci??n']) 
            context['MClas']=Matriz_Clasificacion1

            #Criterios
            criterios = []
            criterios.append(ModeloAD.criterion)
            criterios.append(accuracy_score(Y_dos, Y_Pronostico))
            context['criterios'] = criterios
            ReporteC =classification_report(Y_dos, Y_Pronostico, output_dict=True)
            RepClas = pd.DataFrame(ReporteC).transpose()
            context['ReporteClas'] = RepClas         

    #Dataframe de la importancia de variables
    Importancia = pd.DataFrame({'Variable': list(NuevaMat[predictoras]),
                            'Importancia': ModeloAD.feature_importances_}).sort_values('Importancia', ascending=False)
    context['Imp'] = Importancia

    #Reporte o arbol en texto

    Reporte = export_text(ModeloAD, feature_names = predictoras)
    RepOut = []
    RepOut = Reporte.split("\n")
    context['reportes'] = RepOut

    #Eleccion para nuevo pronostico
    predictorasOut = NuevaMat[predictoras]
    context['Pred'] = predictorasOut.iloc[np.r_[0:5, -5:0]]

    return render(request, 'Arboles/AD2.html', context)

def AD3(request,pk, algType):
    context = {}
    context['pk'] = pk
    context['type'] = algType
    Val = request.POST.getlist('Nvals')
    X = pd.read_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/X.csv'))
    Y = pd.read_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/Y.csv'))

    #Division de los datos
    X_train, X_dos, Y_train, Y_dos = model_selection.train_test_split(X, Y, 
                                                                        test_size = 0.2, 
                                                                        random_state = 0, 
                                                                        shuffle = True)
    
    match algType:
        case 'P':
            context['AlgName'] = 'Pronostico'
            #Entrenamiento
            ModeloAD = DecisionTreeRegressor(random_state=0)            
            flag = True
            
        case 'C':
            context['AlgName'] = 'Clasificacion'
            #Entrenamiento
            ModeloAD = DecisionTreeClassifier(random_state=0)
            flag = False

    context['flag'] = flag
    
    ModeloAD.fit(X_train, Y_train)

    col = list(X.columns)
    datoOut = {}
    for i in range(X.shape[1]):
        datoOut[col[i]] = float(Val[i])
    Npron = pd.DataFrame(datoOut, index=[0])
    context['DfN'] = Npron
    resultado = ModeloAD.predict(Npron)
    context['resultado'] = resultado

    return render(request, 'Arboles/AD3.html', context)

def ADErrror(request, pk, algType):
    context = showDatos(pk)
    flag = bool
    context['type'] = algType

    match algType:
        case 'P':
            context['AlgName'] = 'Pronostico'
            flag = True
        case 'C':
            context['AlgName'] = 'Clasificacion'
            flag = False
    context['flag'] = flag

    return render(request, 'Arboles/ADError.html', context)

#Bosques
def BA(request, pk, algType):
    context = showDatos(pk)
    flag = bool
    context['type'] = algType

    match algType:
        case 'P':
            context['AlgName'] = 'Pronostico'
            flag = True
        case 'C':
            context['AlgName'] = 'Clasificacion'
            flag = False
    context['flag'] = flag
    return render(request, 'Bosques/BA.html', context)

def BA2(request, pk, algType):
    proyecto = Proyecto.objects.get(pk=pk)
    context = {}
    context['pk'] = pk
    context['type'] = algType
    flag = bool
    #Obtencion de Var Cat y Pred
    predictoras = request.POST.getlist('predictora')
    pronosticar = request.POST['pronostico']
    
    #verificar que no hayas elegido la misma columna en ambos y que se haya escogido al menos una variable predictora
    for i in range (len(predictoras)):
        if(predictoras[i] == pronosticar):
            return redirect('/BAError/{}/{}'.format(pk,algType))
    
    if( len(predictoras) <= 0):
        return redirect('/BAError/{}/{}'.format(pk,algType))

    #Paso usual
    source = proyecto.data
    df = pd.read_csv(source)
    for i in range(df.shape[1]):
        df.columns.values[i] = df.columns.values[i].replace(" ","_")
    
    #Limpiamos de nuevo
    NuevaMat = df.dropna() 


    #Seleccion variables Predictoras y de pronostico
    aux = pd.DataFrame(NuevaMat[predictoras])
    X = np.array(NuevaMat[predictoras])
    Xout = pd.DataFrame(data=X, columns=aux.columns.values)
    Xout.to_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/X.csv'), index=False)
    context['X'] = Xout.iloc[np.r_[0:5, -5:0]]

    Y = np.array(NuevaMat[pronosticar])
    Yout = pd.DataFrame(Y)
    
    Yout.to_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/Y.csv'), index=False)
    context['Y'] = Yout.iloc[np.r_[0:5, -5:0]]

    #Division de los datos
    X_train, X_dos, Y_train, Y_dos = model_selection.train_test_split(X, Y, 
                                                                                    test_size = 0.2, 
                                                                                    random_state = 0,
                                                                                    shuffle = True)
    match algType:
        case 'P':
            context['AlgName'] = 'Pronostico'
            #Entrenamiento
            ModeloBA = RandomForestRegressor(random_state=0)
            
            #Visualizacion de datos de prueba
            Xtest = pd.DataFrame(X_dos)
            context['Xtest'] = Xtest.iloc[np.r_[0:5, -5:0]]
            
            flag = True
            
        case 'C':
            context['AlgName'] = 'Clasificacion'
            #Entrenamiento
            ModeloBA = RandomForestClassifier(random_state=0)
            flag = False

    context['flag'] = flag
    
    ModeloBA.fit(X_train, Y_train)
    
    #Se genera el pron??stico
    Y_Pronostico = ModeloBA.predict(X_dos)
    Ypronostico = pd.DataFrame(Y_Pronostico)
    context['YPron'] = Ypronostico.iloc[np.r_[0:5, -5:0]]

    #Comparacion entre pronostico y prueba
    Valores = pd.DataFrame(Y_dos, Y_Pronostico)
    Valores2 = Valores.reset_index()
    ValoresOut = Valores2.rename(columns={Valores2.columns[0]: 'Prueba', Valores2.columns[1]: 'Pronostico'})
    context['Valores'] = ValoresOut.iloc[np.r_[0:5, -5:0]]

    match algType:
        case 'P':

            #Obtencion del ajuste de Bondad
            Score = r2_score(Y_dos, Y_Pronostico)
            context['Score'] = Score

            #Criterios
            criterios = []
            criterios.append(ModeloBA.criterion)
            criterios.append(mean_absolute_error(Y_dos, Y_Pronostico))
            criterios.append(mean_squared_error(Y_dos, Y_Pronostico))
            criterios.append(mean_squared_error(Y_dos, Y_Pronostico, squared=False))
            context['criterios'] = criterios

        case 'C':

            #Obtencion del ajuste de Bondad
            Score = accuracy_score(Y_dos, Y_Pronostico)
            context['Score'] = Score

            #Matriz de clasificacion
            ModeloClasificacion1 = ModeloBA.predict(X_dos)
            Matriz_Clasificacion1 = pd.crosstab(Y_dos.ravel(), 
                                        ModeloClasificacion1, 
                                        rownames=['Actual'], 
                                        colnames=['Clasificaci??n']) 
            context['MClas']=Matriz_Clasificacion1

            #Criterios
            criterios = []
            criterios.append(ModeloBA.criterion)
            criterios.append(accuracy_score(Y_dos, Y_Pronostico))
            context['criterios'] = criterios
            ReporteC =classification_report(Y_dos, Y_Pronostico, output_dict=True)
            RepClas = pd.DataFrame(ReporteC).transpose()
            context['ReporteClas'] = RepClas         

    #Dataframe de la importancia de variables
    Importancia = pd.DataFrame({'Variable': list(NuevaMat[predictoras]),
                            'Importancia': ModeloBA.feature_importances_}).sort_values('Importancia', ascending=False)
    context['Imp'] = Importancia

    #Reporte o arbol en texto
    Estimador = ModeloBA.estimators_[50]

    Reporte = export_text(Estimador, feature_names = predictoras)
    RepOut = []
    RepOut = Reporte.split("\n")
    context['reportes'] = RepOut

    #Eleccion para nuevo pronostico
    predictorasOut = NuevaMat[predictoras]
    context['Pred'] = predictorasOut.iloc[np.r_[0:5, -5:0]]

    return render(request, 'Bosques/BA2.html', context)

def BA3(request,pk, algType):
    context = {}
    context['pk'] = pk
    context['type'] = algType
    Val = request.POST.getlist('Nvals')
    X = pd.read_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/X.csv'))
    Y = pd.read_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/Y.csv'))

    #Division de los datos
    X_train, X_dos, Y_train, Y_dos = model_selection.train_test_split(X, Y, 
                                                                        test_size = 0.2, 
                                                                        random_state = 0, 
                                                                        shuffle = True)
    
    match algType:
        case 'P':
            context['AlgName'] = 'Pronostico'
            #Entrenamiento
            ModeloBA = RandomForestRegressor(random_state=0)
            flag = True
            
        case 'C':
            context['AlgName'] = 'Clasificacion'
            #Entrenamiento
            ModeloBA = RandomForestClassifier(random_state=0)
            flag = False

    ModeloBA.fit(X_train, Y_train)
    context['flag'] = flag

    col = list(X.columns)
    datoOut = {}
    print(X.shape[1])
    for i in range(X.shape[1]):
        datoOut[col[i]] = float(Val[i])
    Npron = pd.DataFrame(datoOut, index=[0])
    context['DfN'] = Npron
    resultado = ModeloBA.predict(Npron)
    print(resultado)
    context['resultado'] = resultado

    return render(request, 'Bosques/BA3.html', context)

def BAErrror(request, pk, algType):
    context = showDatos(pk)
    flag = bool
    context['type'] = algType

    match algType:
        case 'P':
            context['AlgName'] = 'Pronostico'
            flag = True
        case 'C':
            context['AlgName'] = 'Clasificacion'
            flag = False
    context['flag'] = flag

    return render(request, 'Bosques/BAError.html', context)

#Segmentacion y clasificacion
def SegClas(request, pk):
    context = showDatos(pk)
    return render(request, 'SegClas/Clusters.html', context)

def SegClas_2(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    context = {}
    context['pk'] = pk
    #Obtencion de Var Cat y Pred
    Elim = request.POST.getlist('Elim')
    Modelo = request.POST.getlist('Modelo')
    
    #verificar que no hayas elegido la misma columna en ambos y que se haya escogido al menos una variable predictora
    for i in range (len(Modelo)):
        for j in range (len(Elim)):
            if(Modelo[i] == Elim[j]):
                return redirect('/SCError/{}'.format(pk))
    
    if( len(Modelo) <= 2):
        return redirect('/SCError/{}'.format(pk))

    #Paso usual
    source = proyecto.data
    df = pd.read_csv(source)
    for i in range(df.shape[1]):
        df.columns.values[i] = df.columns.values[i].replace(" ","_")
    
    #Limpiamos y especificamos las variables del modelo
    NuevaMatriz = df.drop(columns=Elim)
    MatrizDef = NuevaMatriz[Modelo]
    NuevaMat = MatrizDef.dropna() 
    
    #Estandarizacion de datos
    Estandarizar = StandardScaler()
    MEstandarizada = Estandarizar.fit_transform(NuevaMat)     
    ME = pd.DataFrame(MEstandarizada, columns=NuevaMat.columns)
    ME2 = ME.iloc[np.r_[0:5, -5:0]]
    context['ME']=ME2

    #Definici??n de k clusters para K-means
    #Se utiliza random_state para inicializar el generador interno de n??meros aleatorios
    SSE = []
    for i in range(2, 10):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(MEstandarizada)
        SSE.append(km.inertia_)

    #Grafica Elbow
    figElb = px.line(SSE, markers=True)
    figElb.update_layout(
        title="Elbow Method",
        xaxis_title="Cantidad de clusters *k*",
        yaxis_title="SSE"
    )

    figVar = plot({'data': figElb}, output_type='div')
    context['figVar']=figVar

    kl = KneeLocator(range(2, 10), SSE, curve="convex", direction="decreasing")
    context['knee']= kl.elbow

    #Se crean las etiquetas de los elementos en los clusters
    MParticional = KMeans(n_clusters=kl.elbow, random_state=0).fit(MEstandarizada)
    MParticional.predict(MEstandarizada)
    context['Mpart'] = MParticional.labels_

    NuevaMat['clusterP'] = MParticional.labels_
    context['DfClust'] = NuevaMat.iloc[np.r_[0:5, -5:0]]

    #Cantidad de elementos en los clusters
    ClustEl =NuevaMat.groupby(['clusterP'])['clusterP'].count()
    ClustOut = pd.DataFrame(ClustEl).transpose()
    context['ClustEl'] = ClustOut

    #Centroides
    CentroidesP = NuevaMat.groupby('clusterP').mean()
    context['CentroidesP'] = CentroidesP

    #Grafica 3d
    figScat = px.scatter_3d(data_frame =NuevaMat, 
                            x=MEstandarizada[:,0], 
                            y=MEstandarizada[:,1], 
                            z=MEstandarizada[:,2], 
                            color = NuevaMat['clusterP'], 
                            hover_name=NuevaMat['clusterP'], symbol='clusterP')
    figScat.add_scatter3d(x=MParticional.cluster_centers_[:, 0], 
                        y=MParticional.cluster_centers_[:, 1], 
                        z=MParticional.cluster_centers_[:, 2], mode='markers')
    
    figScatOut = plot({'data': figScat}, output_type='div')
    context['figVar2']=figScatOut

    #Seleccion variables Predictoras y de pronostico
    aux = pd.DataFrame(NuevaMat.loc[:, NuevaMat.columns != 'clusterP'])
    X = np.array(NuevaMat.loc[:, NuevaMat.columns != 'clusterP'])
    Xout = pd.DataFrame(data=X, columns=aux.columns.values)
    Xout.to_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/X.csv'), index=False)
    context['X'] = Xout.iloc[np.r_[0:5, -5:0]]

    Y = np.array(NuevaMat[['clusterP']])
    Yout = pd.DataFrame(Y)
    Yout.to_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/Y.csv'), index=False)

    
    context['Y'] = Yout[:10]

    #Division de los datos
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)
    
    #ClasificacionBA = RandomForestClassifier(random_state=0)
    #ClasificacionBA.fit(X_train, Y_train)

    ClasificacionBA = RandomForestClassifier(n_estimators=105,
                                            max_depth=8, 
                                            min_samples_split=4, 
                                            min_samples_leaf=2, 
                                            random_state=1234)
    ClasificacionBA.fit(X_train, Y_train)
    
    #Clasificaci??n final 
    Y_ClasificacionBA = ClasificacionBA.predict(X_validation)


    #Comparacion entre pronostico y prueba
    Valores = pd.DataFrame(Y_validation, Y_ClasificacionBA)
    Valores2 = Valores.reset_index()
    ValoresOut = Valores2.rename(columns={Valores2.columns[0]: 'Prueba', Valores2.columns[1]: 'Pronostico'})
    #print(ValoresOut)
    context['Valores'] = ValoresOut.iloc[np.r_[0:5, -5:0]]

    #Obtencion del ajuste de Bondad
    Score = accuracy_score(Y_validation, Y_ClasificacionBA)
    context['Score'] = Score

    #Matriz de clasificacion
    ModeloClasificacion1 = ClasificacionBA.predict(X_validation)
    Matriz_Clasificacion1 = pd.crosstab(Y_validation.ravel(), 
                                   ModeloClasificacion1, 
                                   rownames=['Reales'], 
                                   colnames=['Clasificaci??n']) 
    context['MClas']=Matriz_Clasificacion1

    #Criterios
    criterios = []
    criterios.append(ClasificacionBA.criterion)
    print('Importancia variables: \n', ClasificacionBA.feature_importances_)
    criterios.append(accuracy_score(Y_validation, Y_ClasificacionBA))
    context['criterios'] = criterios
    ReporteC =classification_report(Y_validation, Y_ClasificacionBA, output_dict=True)
    RepClas = pd.DataFrame(ReporteC).transpose()
    context['ReporteClas'] = RepClas

    #Dataframe de la importancia de variables
    Importancia = pd.DataFrame({'Variable': list(NuevaMat.loc[:, NuevaMat.columns != 'clusterP']),
                            'Importancia': ClasificacionBA.feature_importances_}).sort_values('Importancia', ascending=False)
    context['Imp'] = Importancia

    #Reporte o arbol en texto
    Estimador = ClasificacionBA.estimators_[50]

    Reporte = export_text(Estimador, feature_names = list(NuevaMat.loc[:, NuevaMat.columns != 'clusterP']))
    RepOut = []
    RepOut = Reporte.split("\n")
    context['reportes'] = RepOut

    #Eleccion para nuevo pronostico
    context['Pred'] = Modelo

    #Lista dummy para en numero de clases en las curvas
    num = []
    for i in range (kl.elbow):
        num.append(i)

    #Rendimiento
    y_score = ClasificacionBA.predict_proba(X_validation)
    y_test_bin = label_binarize(Y_validation, classes=num)
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    AUC = []
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        if i == 0 :
            figAUC = px.line(x=fpr[i], y=tpr[i])
            figAUC.update_xaxes(range=[-0.05, 1.05])
            figAUC.update_yaxes(range=[-0.05, 1.05])
        else:
            figAUC.add_scatter(x=fpr[i], y=tpr[i], mode='lines')
        AUC.append('AUC para la clase {}: {}'.format(i+1, auc(fpr[i], tpr[i])))
    figAUCout = plot({'data': figAUC}, output_type='div')
    context['textAUC'] = AUC
    context['AUC'] = figAUCout
    return render(request, 'SegClas/Clusters2.html', context)

def SegClas_3(request,pk):
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
    
    #ClasificacionBA = RandomForestClassifier(random_state=0)
    #ClasificacionBA.fit(X_train, Y_train)

    ClasificacionBA = RandomForestClassifier(n_estimators=105,
                                            max_depth=8, 
                                            min_samples_split=4, 
                                            min_samples_leaf=2, 
                                            random_state=1234)
    ClasificacionBA.fit(X_train, Y_train)
    
    col = list(X.columns)
    datoOut = {}
    for i in range(X.shape[1]):
        datoOut[col[i]] = float(Val[i])
    Npron = pd.DataFrame(datoOut, index=[0])
    context['DfN'] = Npron
    resultado = ClasificacionBA.predict(Npron)
    print(resultado)
    context['resultado'] = resultado

    return render(request, 'SegClas/Clusters3.html', context)

def SCErrror(request, pk):
    context = showDatos(pk)
    return render(request, 'SegClas/SCError.html', context)

#Maquinas de soporte vectorial
def SVM(request, pk, algType):
    context = showDatos(pk)
    context['type'] = algType

    match algType:
        case 'L':
            context['AlgName'] = 'Lineal'
        case 'P':
            context['AlgName'] = 'Polinomial'
        case 'RBF':
            context['AlgName'] = 'RBF -Funci??n de Base Radial-'
        case 'SGM':
            context['AlgName'] = 'Sigmoide'

    return render(request, 'SVM/SVM.html', context)

def SVM_2(request,pk, algType):
    proyecto = Proyecto.objects.get(pk=pk)
    context = {}
    context['pk'] = pk
    context['type'] = algType

    #Obtencion de Var Cat y Pred
    predictoras = request.POST.getlist('predictora')
    pronosticar = request.POST['pronostico']
    
    #verificar que no hayas elegido la misma columna en ambos
    for i in range (len(predictoras)):
        if(predictoras[i] == pronosticar):
            return redirect('/SVMError/{}/{}'.format(pk,algType))
    
    if( len(predictoras) <= 0):
        return redirect('/SVMError/{}/{}'.format(pk,algType))

    #Paso usual
    source = proyecto.data
    df = pd.read_csv(source)
    for i in range(df.shape[1]):
        df.columns.values[i] = df.columns.values[i].replace(" ","_")
    NuevaMat = df.dropna() 


    #Seleccion variables Predictoras y de pronostico
    aux = pd.DataFrame(NuevaMat[predictoras])
    X = np.array(NuevaMat[predictoras])
    Xout = pd.DataFrame(data=X, columns=aux.columns.values)
    Xout.to_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/X.csv'), index=False)
    context['X'] = Xout.iloc[np.r_[0:5, -5:0]]

    Y = np.array(NuevaMat[pronosticar])
    Yout = pd.DataFrame(Y)
    
    Yout.to_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/Y.csv'), index=False)

    
    context['Y'] = Yout.iloc[np.r_[0:5, -5:0]]

    #Division de los datos
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)
    
    #Eleccion del tipo de algoritmo
    match algType:
        case 'L':
            context['AlgName'] = 'Lineal'
            ModeloSVM_1 = SVC(kernel='linear')
        case 'P':
            context['AlgName'] = 'Polinomial'
            ModeloSVM_1 = SVC(kernel='poly', degree=3)
        case 'RBF':
            context['AlgName'] = 'RBF -Funci??n de Base Radial-'
            ModeloSVM_1 = SVC(kernel='rbf')
        case 'SGM':
            context['AlgName'] = 'Sigmoide'
            ModeloSVM_1 = SVC(kernel='sigmoid')

    ModeloSVM_1.fit(X_train, Y_train)
    #Clasificaciones agregadas
    Clasificaciones_1 = ModeloSVM_1.predict(X_validation)
    context['Clas1']= Clasificaciones_1
    Clas1Df = pd.DataFrame(Clasificaciones_1)
    context['ClasDf1'] = Clas1Df.iloc[np.r_[0:5, -5:0]]

    #Coomparacion entre Yvalidation y Clasif1
    Clasificaciones = pd.DataFrame(Y_validation, Clasificaciones_1)
    context['Valores'] = Clasificaciones.iloc[np.r_[0:5, -5:0]]
    print(Clasificaciones)

    #Se calcula la exactitud promedio de la validaci??n
    Score = ModeloSVM_1.score(X_validation, Y_validation)
    context['Score'] = Score
    print(ModeloSVM_1.score(X_validation, Y_validation))

    #Matriz de clasificacion
    Clasificaciones_1 = ModeloSVM_1.predict(X_validation)
    Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                    Clasificaciones_1, 
                                    rownames=['Real'], 
                                    colnames=['Clasificaci??n']) 
    print(Matriz_Clasificacion)
    context['MClas']=Matriz_Clasificacion

    #Criterios
    criterios = []
    print("Exactitud", ModeloSVM_1.score(X_validation, Y_validation))
    criterios.append(ModeloSVM_1.score(X_validation, Y_validation))
    context['criterios'] = criterios
    ReporteC =classification_report(Y_validation, Clasificaciones_1, output_dict=True)
    RepClas = pd.DataFrame(ReporteC).transpose()
    context['ReporteClas'] = RepClas

    #Dataframe de los vectores de soporte
    VectoresSoporte_1 = ModeloSVM_1.support_vectors_
    VectSup = pd.DataFrame(VectoresSoporte_1)
    context['VectSup'] = VectSup.iloc[np.r_[0:5, -5:0]]

    #Vectores de soporte
    NSup = ModeloSVM_1.n_support_
    context['NSup'] = NSup
    VectList = ModeloSVM_1.support_
    context['VectList'] = VectList
    

    #NUeva clasificacion
    predictorasOut = NuevaMat[predictoras]
    context['Pred'] = predictorasOut.iloc[np.r_[0:5, -5:0]]


    return render(request, 'SVM/SVM2.html', context)

def SVM_3(request,pk, algType):
    proyecto = Proyecto.objects.get(pk=pk)
    context = {}
    context['pk'] = pk
    context['type'] = algType
    Val = request.POST.getlist('NClas')
    X = pd.read_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/X.csv'))
    Y = pd.read_csv(os.path.join(BASE_DIR, 'WebApp/data/Tmp/Y.csv'))

    #Division de los datos
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)
    
    #Eleccion del tipo de algoritmo
    match algType:
        case 'L':
            context['AlgName'] = 'Lineal'
            ModeloSVM_1 = SVC(kernel='linear')
        case 'P':
            context['AlgName'] = 'Polinomial'
            ModeloSVM_1 = SVC(kernel='poly', degree=3)
        case 'RBF':
            context['AlgName'] = 'RBF -Funci??n de Base Radial-'
            ModeloSVM_1 = SVC(kernel='rbf')
        case 'SGM':
            context['AlgName'] = 'Sigmoide'
            ModeloSVM_1 = SVC(kernel='sigmoid')
    ModeloSVM_1.fit(X_train, Y_train)
    
    col = list(X.columns)
    datoOut = {}
    for i in range(X.shape[1]):
        datoOut[col[i]] = float(Val[i])
    Npron = pd.DataFrame(datoOut, index=[0])
    context['DfN'] = Npron
    resultado = ModeloSVM_1.predict(Npron)
    print(resultado)
    context['resultado'] = resultado

    return render(request, 'SVM/SVM3.html', context)

def SVMError(request, pk, algType):
    context = showDatos(pk)
    context['type'] = algType

    match algType:
        case 'L':
            context['AlgName'] = 'Lineal'
        case 'P':
            context['AlgName'] = 'Polinomial'
        case 'RBF':
            context['AlgName'] = 'RBF -Funci??n de Base Radial-'
        case 'SGM':
            context['AlgName'] = 'Sigmoide'

    return render(request, 'SVM/SVMError.html', context)