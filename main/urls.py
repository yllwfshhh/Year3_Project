from django.urls import path
from . import views # from the current folder import views

urlpatterns = [
    path('index/', views.index, name='index'),# then back to main urls.py 
    path('predict/', views.predict, name='predict'),# then back to main urls.py 
    path('test/', views.test, name='test')
]