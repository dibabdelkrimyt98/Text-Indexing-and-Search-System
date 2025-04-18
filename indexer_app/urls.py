"""URL configuration for indexer_app."""

from django.urls import path
from . import views

app_name = 'indexer_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('home/', views.home_view, name='home'),
    path('upload/', views.upload_document_view, name='upload'),
    path('process/', views.process_document_view, name='process_document'),
    path('search/', views.search_view, name='search'),
    path('similarity/', views.tfidf_similarity_view, name='similarity'),
]