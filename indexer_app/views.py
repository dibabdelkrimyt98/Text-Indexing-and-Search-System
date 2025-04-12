from multiprocessing import context
from django.shortcuts import render, redirect
from django.http import HttpResponse

# Create your views here.
def upload_index_view(request):
    if request.method == 'POST':
        print("POST request received on upload/index view")
        pass 
    
    context = {
        'page title' :  'Upload and Index Documents',
    }
def view_index_view(request):
    index_data = { # Placeholder data
        'term1': {'doc1': 0.5, 'doc2': 0.1},
        'term2': {'doc1': 0.0, 'doc2': 0.8},
    }
    print("Request received for viewing index") # Placeholder

    context = {
        'page_title': 'View TF-IDF Index',
        'index_data': index_data, # Pass the loaded index to the template
    }
    return render(request, 'indexer_app/index_view.html', context)


def search_view(request):
    results = []
    query = ''
    similarity_method = ''

    if request.method == 'GET' and 'query' in request.GET: # Assuming search via GET query params
        query = request.GET.get('query', '')
        similarity_method = request.GET.get('similarity', 'cosine') # Default to cosine

        if query:
            print(f"Search request received: Query='{query}', Method='{similarity_method}'") # Placeholder
            results = [ # Placeholder results
                {'doc_name': 'document1.txt', 'score': 0.95},
                {'doc_name': 'document3.txt', 'score': 0.78},
            ]
        else:
            # Handle empty query if necessary
            pass
    context = {
        'page_title': 'Search Documents',
        'query': query,
        'similarity_method': similarity_method,
        'results': results,
    }
    return render(request, 'indexer_app/search.html', context)