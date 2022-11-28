from django.shortcuts import render


# Create your views here.
def home(request):
    return render(request,'home.html')

def pyscript(request):
    return render(request, 'pyscript.html')

def EDA(request):
    return render(request, 'EDA.html')