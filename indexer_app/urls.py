"""URL routing for indexer_app."""

from django.urls import path
from . import views

app_name = 'indexer_app'

urlpatterns = [
    path('', views.upload_index_view, name='upload_index'),
    path('view_index/', views.view_index_view, name='view_index'),
    path('search/', views.search_view, name='search'),
]
