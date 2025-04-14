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
    # It might be useful to add a path for the TF-IDF table page
    path('tfidf-table/', views.tfidf_table_page, name='tfidf_table'), 
]