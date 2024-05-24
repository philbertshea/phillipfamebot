from django import forms

class QueryList(forms.Form):
    query = forms.CharField(label='Ask me anything!', widget=forms.Textarea(attrs={'label': 'Ask me anything!', 'name':'body', 'rows':5, 'cols':100}), max_length=200)