"""URL configuration for indexer_app."""

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('upload/', views.upload_document_view, name='upload'),
    path('process/', views.process_document_view, name='process_document'),
    path('search/', views.search_view, name='search'),
    path(
        'tfidf-similarity/',
        views.tfidf_similarity_view,
        name='tfidf_similarity'
    ),
]
