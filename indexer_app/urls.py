"""URL configuration for indexer_app."""

from django.urls import path
from . import views

app_name = 'indexer_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('home/', views.home_view, name='home'),
    path('try/', views.try_view, name='try'),
    path('upload/', views.upload_document_view, name='upload_document'),
    path('process/', views.process_document_view, name='process'),
    path('search/', views.search_view, name='search'),
    path('api/search/', views.search_documents, name='search_documents'),
    path('api/tfidf-matrix/', views.get_tfidf_matrix, name='get_tfidf_matrix'),
    path('similarity/', views.tfidf_similarity_view, name='similarity'),
    path('results/', views.process_results, name='results'),
    path('test-json/', views.test_json_view, name='test_json'),
    path('test/', views.test_page_view, name='test'),
    path('test-upload/', views.test_upload_view, name='test_upload'),
    path('test-upload-page/', views.test_upload_page_view, name='test_upload_page'),
    path('test-plain/', views.test_plain_view, name='test_plain'),
    path('test-bg/', views.test_bg_view, name='test_bg'),
]