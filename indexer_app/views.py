import os
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk


# Download necessary resources for nltk
nltk.download('punkt')


# Home page view
def index(request):
    return render(request, 'indexer_app/index.html')



# Upload a document and save it to the server
def upload_document(request):
    if request.method == 'POST' and request.FILES.get('document'):
        document = request.FILES['document']
        fs = FileSystemStorage()
        filename = fs.save(document.name, document)
        file_url = fs.url(filename)
        return JsonResponse({
            'file_url': file_url
        })

        return JsonResponse({'file_url': file_url})

    return JsonResponse({'error': 'No document uploaded'}, status=400)

# Process the uploaded document, calculate similarity, and return results
def process_document(request):
    if request.method == 'POST' and request.FILES.get('document'):
        document = request.FILES['document']
        fs = FileSystemStorage()
        filename = fs.save(document.name, document)
        file_url = fs.url(filename)

        # Read the document's content
        with open(os.path.join(fs.location, filename), 'r', encoding='utf-8') as file:
            text = file.read()

        # Compute the TF-IDF matrix and similarity score
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        return JsonResponse({
            'file_url': file_url,
            'similarity_score': cosine_sim[0][0]
        })

    return JsonResponse({'error': 'No document uploaded'}, status=400)

# Search for a query and return similar documents
def search(request):
    if request.method == 'GET':
        query = request.GET.get('query', '')
        results = []

        if query:
            document_list = os.listdir('documents/')
            for document_name in document_list:
                if document_name.endswith('.txt'):
                    with open(os.path.join('documents/', document_name), 'r', encoding='utf-8') as file:
                        text = file.read()
                    similarity = calculate_similarity(query, text)
                    results.append({
                        'document': document_name,
                        'similarity': similarity
                    })

        return JsonResponse({'results': results})

    return JsonResponse({'error': 'Invalid request'}, status=400)

# Calculate similarity score between query and document text
def calculate_similarity(query, text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([query, text])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return cosine_sim[0][0]
