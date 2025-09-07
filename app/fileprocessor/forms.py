from django import forms
from .models import CSVFile

class CSVUploadForm(forms.ModelForm):
    class Meta:
        model = CSVFile
        fields = ('file',)

    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            if not file.name.endswith('.csv'):
                raise forms.ValidationError('File must be a CSV')
        return file

