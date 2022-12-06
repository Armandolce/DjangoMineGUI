from django import forms
from .models import Proyecto

class ProjectForm(forms.ModelForm):
    class Meta:
        model = Proyecto
        fields = ('Nombre','descripcion','URL','data')