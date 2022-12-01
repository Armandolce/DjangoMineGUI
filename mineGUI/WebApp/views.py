from django.shortcuts import render
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

def demo_plot_view(request):
    """ 
    View demonstrating how to display a graph object
    on a web page with Plotly. 
    """


    df = pd.read_csv('https://raw.githubusercontent.com/Armandolce/MineriaDatos/master/Proyecto/melb_data.csv')
    df2 = df[:10]
    json_records = df2.reset_index().to_json(orient ='records')
    arr = []
    arr = json.loads(json_records)
    contextt = {'d': arr}
    # Setting layout of the figure.
    layout = {
        'title': 'Histogramas?',
        'xaxis_title': 'X',
        'yaxis_title': 'Y',
        'height': 420,
        'width': 560,
    }

    fig = px.histogram(df, x="Price")
    #fig = px.box(df, x="Price") 

    # Getting HTML needed to render the plot.
    plot_div = plot({'data': fig, 'layout': layout}, 
                    output_type='div')

    return render(request, 'demo-plot.html', 
                  context={'plot_div': plot_div, 'df': df2})