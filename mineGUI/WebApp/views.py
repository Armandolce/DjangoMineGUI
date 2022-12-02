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

def EDA(request):
    return render(request, 'EDA.html')

def busqueda(request):
    return render(request, 'Busqueda.html')

def demo_plot_view(request):
    source = request.POST['fuente']
    #source = "WebApp/data/melb_data.csv"
    df = pd.read_csv(source)
    df2 = df[:10]
    size = df.shape
    fig = px.histogram(df, x=df.columns[2])
    #fig = px.box(df, x="Price") 

    # Setting layout of the figure.
    layout = {
        'title': 'Histogramas?',
        'xaxis_title': 'X',
        'yaxis_title': 'Y',
        'height': 420,
        'width': 560,
    }

    # Getting HTML needed to render the plot.
    plot_div = plot({'data': fig, 'layout': layout}, 
                    output_type='div')

    return render(request, 'demo-plot.html', 
                  context={'plot_div': plot_div, 'df': df2, 'size' : size})