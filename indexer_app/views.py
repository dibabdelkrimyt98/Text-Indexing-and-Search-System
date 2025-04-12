# indexer_app/views.py

import os
import json
import pickle # For saving/loading Python objects (like sklearn models)

from django.shortcuts import render, redirect
from django.contrib import messages # For user feedback
from django.core.files.storage import FileSystemStorage # For saving files
from django.conf import settings # To access settings like MEDIA_ROOT, INDEX_DATA_DIR

# Import your form
from .forms import DocumentUploadForm
# Import your model (optional)
# from .models import IndexedDocument

# Import ML/NLP libraries (install them first: pip install scikit-learn nltk)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import nltk
# Download necessary NLTK data (run once, e.g., in Django shell or startup)
# try:
#     nltk.data.find('corpora/stopwords')
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('stopwords', quiet=True)
#     nltk.download('punkt', quiet=True)
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer # Or another stemmer/lemmatizer

# --- Helper Functions (Placeholders - Implement these!) ---

def _process_text(text):
    """Placeholder: Clean text (tokenize, remove stop words, stem/lemmatize)."""
    # TODO: Implement text processing using nltk
    # Example (very basic):
    # tokens = nltk.word_tokenize(text.lower())
    # stop_words = set(stopwords.words('english')) # Or 'french'
    # filtered_tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
    # stemmer = PorterStemmer()
    # stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
    # return " ".join(stemmed_tokens)
    print(f"Processing text (length: {len(text)})...") # Placeholder
    return text # Return unprocessed for now

def _calculate_and_save_index(file_paths):
    """Placeholder: Calculate TF-IDF and save index components."""
    # TODO: Implement TF-IDF calculation and saving
    try:
        documents_content = []
        filenames = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f: # Specify encoding
                content = f.read()
                processed_content = _process_text(content)
                documents_content.append(processed_content)
                filenames.append(os.path.basename(file_path))

        if not documents_content:
            print("No content to index.")
            return False

        print(f"Calculating TF-IDF for {len(documents_content)} documents...") # Placeholder
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents_content)

        # Save the components
        with open(settings.VECTORIZER_FILE_PATH, 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(settings.INDEX_FILE_PATH, 'wb') as f:
            pickle.dump(tfidf_matrix, f)
        with open(settings.FILE_LIST_PATH, 'w') as f:
            json.dump(filenames, f)

        print("Index components saved successfully.") # Placeholder
        return True
    except Exception as e:
        print(f"Error calculating/saving index: {e}")
        return False

def _load_index_components():
    """Placeholder: Load saved index components."""
    # TODO: Implement loading with error handling
    try:
        if not os.path.exists(settings.INDEX_FILE_PATH):
            return None, None, None # Indicate index doesn't exist

        with open(settings.VECTORIZER_FILE_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(settings.INDEX_FILE_PATH, 'rb') as f:
            tfidf_matrix = pickle.load(f)
        with open(settings.FILE_LIST_PATH, 'r') as f:
            filenames = json.load(f)
        print("Index components loaded.") # Placeholder
        return vectorizer, tfidf_matrix, filenames
    except Exception as e:
        print(f"Error loading index components: {e}")
        return None, None, None

def _calculate_similarity(query_vector, index_matrix, method='cosine'):
    """Placeholder: Calculate similarity between query and documents."""
    # TODO: Implement similarity calculation
    print(f"Calculating similarity using {method}...") # Placeholder
    if method == 'cosine':
        # cosine_similarity returns shape (n_queries, n_documents)
        similarity_scores = cosine_similarity(query_vector, index_matrix)[0] # Get scores for the single query
    elif method == 'euclidean':
        # euclidean_distances returns distances, lower is better. Convert to similarity?
        # Simple inverse (handle division by zero if needed): 1 / (1 + distance)
        distances = euclidean_distances(query_vector, index_matrix)[0]
        similarity_scores = 1 / (1 + distances)
    else:
        similarity_scores = [0] * index_matrix.shape[0] # Default to zero score

    return similarity_scores

# --- Django Views ---

def upload_index_view(request):
    if request.method == 'POST':
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_files = form.cleaned_data['documents'] # Use cleaned_data
            saved_file_paths = []
            fs = FileSystemStorage() # Uses MEDIA_ROOT by default

            # Clear previous index data if desired (optional)
            # Be careful with deleting files! Add confirmation?
            # if os.path.exists(settings.INDEX_FILE_PATH): os.remove(settings.INDEX_FILE_PATH)
            # if os.path.exists(settings.VECTORIZER_FILE_PATH): os.remove(settings.VECTORIZER_FILE_PATH)
            # if os.path.exists(settings.FILE_LIST_PATH): os.remove(settings.FILE_LIST_PATH)

            for file in uploaded_files:
                # Save the file using Django's file storage system
                filename = fs.save(file.name, file) # Saves to MEDIA_ROOT
                file_path = fs.path(filename) # Gets the full path
                saved_file_paths.append(file_path)
                # Optional: Save metadata to database
                # IndexedDocument.objects.update_or_create(file_name=file.name)

            # Process and index the saved files
            success = _calculate_and_save_index(saved_file_paths)

            if success:
                messages.success(request, f'{len(uploaded_files)} document(s) uploaded and indexed successfully!')
                return redirect('indexer_app:view_index') # Redirect after successful POST
            else:
                 messages.error(request, 'Failed to calculate or save the index.')

        else:
            # Form is not valid, errors will be displayed automatically by Django template tags
             messages.error(request, 'Please correct the errors below.')
    else:
        # GET request, create a blank form
        form = DocumentUploadForm()

    context = {
        'form': form,
        'page_title': 'Upload and Index Documents',
    }
    return render(request, 'indexer_app/upload.html', context)

def view_index_view(request):
    vectorizer, tfidf_matrix, filenames = _load_index_components()
    display_data = {}

    if vectorizer is not None and tfidf_matrix is not None and filenames is not None:
        # Prepare data for display (this can be large!)
        # Consider showing only a summary or a portion if the index is big.
        # Example: Show terms and their scores for each document
        terms = vectorizer.get_feature_names_out()
        for i, filename in enumerate(filenames):
            # Get scores for this document (convert sparse matrix row to dense)
            doc_scores = tfidf_matrix[i].toarray().flatten()
            # Create pairs of (term, score) for scores > 0
            term_scores = {terms[j]: round(score, 4) for j, score in enumerate(doc_scores) if score > 0}
            # Sort by score descending for better readability (optional)
            sorted_term_scores = dict(sorted(term_scores.items(), key=lambda item: item[1], reverse=True))
            display_data[filename] = sorted_term_scores
        messages.info(request, 'Loaded existing index.')
    else:
        messages.warning(request, 'No index found. Please upload documents first.')

    context = {
        'page_title': 'View TF-IDF Index',
        'index_data': display_data, # Pass formatted data to template
        'has_index': bool(display_data) # Flag for template conditional display
    }
    return render(request, 'indexer_app/index.html', context) # Using index.html

def search_view(request):
    results = []
    query = request.GET.get('query', '')
    similarity_method = request.GET.get('similarity', 'cosine') # Default from Cahier des Charges [cite: 21]

    vectorizer, tfidf_matrix, filenames = _load_index_components()

    if vectorizer is None:
        messages.error(request, 'Index not found. Please index documents before searching.')
    elif query:
        try:
            # Process and vectorize the query
            processed_query = _process_text(query)
            query_vector = vectorizer.transform([processed_query]) # Use transform, not fit_transform

            # Calculate similarity
            similarity_scores = _calculate_similarity(query_vector, tfidf_matrix, similarity_method)

            # Combine filenames with scores and sort
            results_with_scores = zip(filenames, similarity_scores)
            # Filter out very low scores (optional threshold) and sort
            threshold = 0.01 # Example threshold
            filtered_results = [(doc, score) for doc, score in results_with_scores if score > threshold]
            sorted_results = sorted(filtered_results, key=lambda item: item[1], reverse=True)

            # Format results for template
            results = [{'doc_name': doc, 'score': round(score, 4)} for doc, score in sorted_results]

            if not results:
                messages.info(request, 'No relevant documents found for your query.')

        except Exception as e:
            messages.error(request, f'An error occurred during search: {e}')

    context = {
        'page_title': 'Search Documents',
        'query': query,
        'similarity_method': similarity_method,
        'results': results,
        'has_index': vectorizer is not None # Pass flag to template
    }
    return render(request, 'indexer_app/search.html', context)