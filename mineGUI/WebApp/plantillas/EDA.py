# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Get the data
from pyodide.http import open_url
url = 'https://raw.githubusercontent.com/Armandolce/MineriaDatos/master/Proyecto/melb_data.csv'
url_content = open_url(url)
df = pd.read_csv(url_content)

# Function to plot the chart
def plot(chart):
   fig, ax = plt.subplots()
   sns.lineplot(y=chart, x="Month", data=df, ax=ax)
   pyscript.write("chart1",fig)

# Set up a proxy to be called when a 'change'
# event occurs in the select control
from js import document
from pyodide import create_proxy
# Read the value of the select control
# and call 'plot'
def selectChange(event):
   choice = document.getElementById("select").value
   plot(choice)
# set the proxy
def setup():
   # Create a JsProxy for the callback function
   change_proxy = create_proxy(selectChange)
   e = document.getElementById("select")
   e.addEventListener("change", change_proxy)
setup()
# Intitially call plot with 'Tmax'
plot('Tmax')
pyscript.write("dataFrame1", df.describe)