from django.shortcuts import render, redirect
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


# Create your views here.
def home(request):
    return render(request,'home.html')

def pyscript(request):
    return render(request, 'pyscript.html')

def busqueda(request):
    return render(request, 'Busqueda.html')

def EDA(request):
    source = request.POST['fuente']
    #source = "WebApp/data/melb_data.csv"
    df = pd.read_csv(source)
    df2 = df[:10]
    size = df.shape
    cajas = []
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
                'height': 420,
                'width': 560,
            }
            # Getting HTML needed to render the plot.
            plot_div = plot({'data': fig, 'layout': layout}, 
                            output_type='div')
            histogramas.append(plot_div)
    
    for i in range(df.shape[1]):
        dataType = df.columns.values[i]
        if df[dataType].dtype != object:
            fig = px.box(df, x=df.columns[i])

            # Setting layout of the figure.
            layout = {
                'title': df.columns[i],
                'xaxis_title': 'X', 
                'yaxis_title': 'Y',
                'height': 420,
                'width': 560,
            }
            # Getting HTML needed to render the plot.
            plot_div = plot({'data': fig, 'layout': layout}, 
                            output_type='div')
            cajas.append(plot_div)

    return render(request, 'EDA.html', 
                  context={'plot_div': histogramas, 'df': df2, 'size' : size, 'diagramsCaja' : cajas})