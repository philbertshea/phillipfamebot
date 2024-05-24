from django.shortcuts import render
from django.http import HttpResponse
from .botfiles.chatbot import query
from .forms import QueryList
# Create your views here.

def say_hello(request):
    if request.method == "POST":
        form = QueryList(request.POST)
        if form.is_valid():
            message = form.cleaned_data['query']
            res = query(message)
            newform = QueryList()
            return render(request, 'hello.html', {"prev": res , "form": newform})
    else:
        form = QueryList()
        return render(request, 'hello.html', {"form": form})