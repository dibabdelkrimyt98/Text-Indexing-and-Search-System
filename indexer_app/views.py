"""
Views for the document indexing application.
"""
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.http import HttpResponse, JsonResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.core.exceptions import ValidationError, ObjectDoesNotExist
# pylint: disable=import-error
from django.conf import settings
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from django.db import transaction
from django.core.files.uploadedfile import UploadedFile
from django.utils import timezone
import numpy as np
from scipy.sparse import vstack

from .models import Document


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)


# Constants
MAX_FILE_SIZE = 40 * 1024 * 1024  # 40MB
ALLOWED_MIME_TYPES = {
    'text/plain',
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}
ALLOWED_EXTENSIONS = {'.txt', '.doc', '.docx', '.pdf'}

# Define upload directory
UPLOAD_DIR = Path(settings.MEDIA_ROOT) / 'documents'


def handle_uploaded_file(uploaded_file: UploadedFile) -> Tuple[str, str]:
    """
    Handle file upload and return content and file type.
    
    Args:
        uploaded_file: The uploaded file object
        
    Returns:
        Tuple containing (file_content, mime_type)
        
    Raises:
        ValidationError: If file type is not supported
    """
    try:
        logger.info(f"Starting to handle uploaded file: {uploaded_file.name}")
        logger.info(f"File size: {uploaded_file.size} bytes")
        logger.info(f"Content type: {uploaded_file.content_type}")
        
        content = ''
        file_path = UPLOAD_DIR / str(uploaded_file.name)
        file_ext = Path(str(uploaded_file.name)).suffix.lower()
        
        logger.info(f"File path: {file_path}")
        logger.info(f"File extension: {file_ext}")
        logger.info(f"UPLOAD_DIR absolute path: {UPLOAD_DIR.absolute()}")

        # Ensure upload directory exists
        try:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Upload directory exists: {UPLOAD_DIR.exists()}")
            logger.info(f"Upload directory is absolute: {UPLOAD_DIR.is_absolute()}")
        except Exception as e:
            logger.error(f"Error creating upload directory: {str(e)}")
            raise

        # Save the file to disk
        try:
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            logger.info(f"File saved successfully to disk at: {file_path}")
            logger.info(f"File exists after save: {file_path.exists()}")
            
            # Verify file was saved correctly
            if not file_path.exists():
                raise Exception(f"File was not saved correctly: {file_path}")
                
            # Check file size
            saved_size = file_path.stat().st_size
            logger.info(f"Saved file size: {saved_size} bytes")
            if saved_size != uploaded_file.size:
                logger.warning(f"File size mismatch: uploaded={uploaded_file.size}, saved={saved_size}")
        except Exception as e:
            logger.error(f"Error saving file to disk: {str(e)}")
            logger.error(f"File path that failed: {file_path}")
            raise

        # Determine mime type based on file extension
        if file_ext == '.txt':
            mime_type = 'text/plain'
            # Read text file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info("Text file read successfully with UTF-8 encoding")
            except UnicodeDecodeError:
                logger.info("UTF-8 encoding failed, trying latin-1")
                # Try with a different encoding if UTF-8 fails
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                logger.info("Text file read successfully with latin-1 encoding")
        elif file_ext in ['.doc', '.docx']:
            mime_type = 'application/msword'
            # For now, just extract text from the beginning of the file
            # In a real app, you'd use a library like python-docx
            content = f"[DOCUMENT CONTENT: {uploaded_file.name}]"
            logger.info("Word document placeholder content created")
        elif file_ext == '.pdf':
            mime_type = 'application/pdf'
            # For now, just extract text from the beginning of the file
            # In a real app, you'd use a library like PyPDF2
            content = f"[PDF CONTENT: {uploaded_file.name}]"
            logger.info("PDF document placeholder content created")
        else:
            error_msg = f"File type {file_ext} processing not implemented yet"
            logger.error(error_msg)
            raise ValidationError(error_msg)

        logger.info(f"File processing completed successfully. MIME type: {mime_type}")
        return content, mime_type

    except Exception as e:
        logger.error(f"Unexpected error in handle_uploaded_file: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error args: {e.args}")
        raise


@csrf_protect
def home_view(request: HttpRequest) -> HttpResponse:
    """
    Render the welcome page.
    
    Args:
        request: The HTTP request
        
    Returns:
        HttpResponse with rendered template
    """
    return render(
        request,
        "indexer_app/index.html",
        {'title': 'Welcome to AOS System'}
    )


@csrf_protect
def index(request: HttpRequest) -> HttpResponse:
    """
    Render the homepage with recent documents.
    
    Args:
        request: The HTTP request
        
    Returns:
        HttpResponse with rendered template
    """
    # Get the 10 most recent documents
    recent_documents = Document.objects.all().order_by('-uploaded_at')[:10]
    
    return render(
        request,
        "indexer_app/index.html",
        {'documents': recent_documents}
    )


@csrf_protect
def upload_document_view(request: HttpRequest) -> HttpResponse:
    """
    Handle document upload form display and submission.
    
    Args:
        request: The HTTP request
        
    Returns:
        HttpResponse with rendered template or redirect
    """
    # If it's a POST request, process the document
    if request.method == 'POST':
        return process_document_view(request)

    # Otherwise, display the upload form
    return render(
        request,
        "indexer_app/upload.html",
        {
            'max_file_size': MAX_FILE_SIZE,
            'allowed_types': [ext[1:] for ext in ALLOWED_EXTENSIONS]
        }
    )


@csrf_protect
def search_view(request: HttpRequest) -> HttpResponse:
    """
    Render the search interface with recent documents.
    
    Args:
        request: The HTTP request
        
    Returns:
        HttpResponse with rendered template
    """
    # Get the 5 most recent documents
    recent_docs = Document.objects.all().order_by('-uploaded_at')[:5]
    
    return render(
        request,
        "indexer_app/search.html",
        {'recent_documents': recent_docs}
    )


@require_http_methods(["POST"])
@csrf_protect
@transaction.atomic
def process_document_view(request: HttpRequest) -> HttpResponse:
    """
    Process uploaded documents and store them in the database.
    
    Args:
        request: The HTTP request
        
    Returns:
        HttpResponse with rendered template or JsonResponse with error
    """
    logger.info("Process document view called")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request FILES: {request.FILES}")
    logger.info(f"Request POST: {request.POST}")
    logger.info(f"Request META: {request.META}")
    logger.info(f"Request headers: {request.headers}")
    
    try:
        # Check if a document was uploaded
        if 'document' not in request.FILES:
            logger.error("No document provided in request.FILES")
            return JsonResponse(
                {"error": "No document provided"},
                status=400
            )

        # Get the uploaded file
        uploaded_file = request.FILES['document']
        logger.info(f"Processing file: {uploaded_file.name}")
        logger.info(f"File size: {uploaded_file.size}")
        logger.info(f"Content type: {uploaded_file.content_type}")
        
        # Validate that it's a proper UploadedFile
        if not isinstance(uploaded_file, UploadedFile):
            logger.error(f"Invalid file upload type: {type(uploaded_file)}")
            return JsonResponse(
                {"error": "Invalid file upload"},
                status=400
            )

        # Check file size
        file_size = uploaded_file.size
        if file_size is None or file_size > MAX_FILE_SIZE:
            size_mb = MAX_FILE_SIZE / (1024 * 1024)
            logger.error(f"File size {file_size} exceeds limit of {size_mb}MB")
            return JsonResponse(
                {"error": f"File size exceeds {size_mb}MB limit"},
                status=400
            )

        # Check file extension
        file_ext = Path(str(uploaded_file.name)).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            ext_list = ', '.join(ALLOWED_EXTENSIONS)
            logger.error(f"Invalid file extension: {file_ext}")
            return JsonResponse(
                {
                    "error": (
                        f"File type not allowed. "
                        f"Supported types: {ext_list}"
                    )
                },
                status=400
            )

        # Get the title from the form or use the filename
        base_title = request.POST.get('title', uploaded_file.name)
        logger.info(f"Processing document with base title: {base_title}")

        # Check if a document with this title already exists and add a timestamp if needed
        title = base_title
        counter = 1
        while Document.objects.filter(title=title).exists():
            # Add a timestamp to make the title unique
            timestamp = timezone.now().strftime("%Y%m%d_%H%M%S")
            title = f"{Path(base_title).stem}_{timestamp}{Path(base_title).suffix}"
            logger.info(f"Title already exists, using new title: {title}")
            counter += 1
            if counter > 10:  # Prevent infinite loop
                logger.error("Too many attempts to create a unique title")
                return JsonResponse(
                    {"error": "Could not create a unique title for the document"},
                    status=500
                )

        # Ensure upload directory exists
        try:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Upload directory exists: {UPLOAD_DIR.exists()}")
            logger.info(f"Upload directory is absolute: {UPLOAD_DIR.is_absolute()}")
            logger.info(f"Upload directory path: {UPLOAD_DIR}")
            
            # Test if directory is writable
            test_file = UPLOAD_DIR / 'test.txt'
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                test_file.unlink()  # Delete the test file
                logger.info("Upload directory is writable")
            except Exception as e:
                logger.error(f"Upload directory is not writable: {str(e)}")
                return JsonResponse(
                    {"error": f"Upload directory is not writable: {str(e)}"},
                    status=500
                )
        except Exception as e:
            logger.error(f"Error creating upload directory: {str(e)}")
            return JsonResponse(
                {"error": f"Error creating upload directory: {str(e)}"},
                status=500
            )

        # Process the file
        try:
            content, mime_type = handle_uploaded_file(uploaded_file)
            logger.info(f"File processed successfully. MIME type: {mime_type}")
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            return JsonResponse({"error": str(e)}, status=400)
        except UnicodeDecodeError:
            logger.error("File encoding error")
            return JsonResponse(
                {"error": "File encoding not supported. Use UTF-8."},
                status=400
            )
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error args: {e.args}")
            return JsonResponse(
                {"error": f"Error processing file: {str(e)}"},
                status=500
            )

        # Preprocess the text
        try:
            processed_content = preprocess_text(content)
            logger.info("Text preprocessing completed")
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return JsonResponse(
                {"error": "Error preprocessing text"},
                status=500
            )

        # Get file type without the dot
        file_type = file_ext[1:] if file_ext.startswith('.') else file_ext
        logger.info(f"File type: {file_type}")

        # Create the document in the database
        try:
            document = Document.objects.create(
                title=title,
                content=content,
                processed_content=processed_content,
                file_type=file_type,
                file_size=file_size
            )
            logger.info(f"Document created with ID: {document.id}")
        except Exception as e:
            logger.error(f"Error creating document: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error args: {e.args}")
            return JsonResponse(
                {"error": f"Error saving document to database: {str(e)}"},
                status=500
            )

        # Update TF-IDF vectors for all documents
        try:
            update_tfidf_vectors()
            logger.info("TF-IDF vectors updated successfully")
        except Exception as e:
            logger.error(f"Error updating TF-IDF vectors: {str(e)}")
            # Don't return error here, as the document was already saved

        # Return success response
        response_data = {
            'success': True,
            'message': 'Document processed successfully',
            'document_id': document.id,
            'title': title
        }
        logger.info(f"Returning success response: {response_data}")
        return JsonResponse(response_data)

    except Exception as e:
        # Log the error and return a generic error message
        logger.error(f"Unexpected error in process_document_view: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error args: {e.args}")
        return JsonResponse(
            {"error": f"An unexpected error occurred: {str(e)}"},
            status=500
        )


def preprocess_text(text: str) -> str:
    """
    Preprocess text by tokenizing, removing stopwords, and joining back.
    
    Args:
        text: The text to preprocess
        
    Returns:
        Preprocessed text string
    """
    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [t for t in tokens if t not in stop_words and t.isalnum()]
    
    # Join tokens back into text
    return ' '.join(filtered_tokens)


def find_best_context(content: str, query: str, window_size: int = 100) -> str:
    """Find the most relevant context around a search match."""
    content_lower = content.lower()
    query_lower = query.lower()
    
    # Find all occurrences of the query
    matches = []
    start = 0
    while True:
        index = content_lower.find(query_lower, start)
        if index == -1:
            break
        matches.append(index)
        start = index + 1
    
    if not matches:
        # If no exact matches, try to find partial matches
        query_words = query_lower.split()
        for word in query_words:
            index = content_lower.find(word)
            if index != -1:
                matches.append(index)
    
    if not matches:
        # If still no matches, return the beginning of the content
        return content[:200] + "..."
    
    # Find the best context window
    best_context = ""
    max_matches = 0
    
    for match in matches:
        start = max(0, match - window_size)
        end = min(len(content), match + len(query) + window_size)
        context = content[start:end]
        
        # Count how many query words appear in this context
        matches_in_context = sum(1 for word in query_lower.split() 
                               if word in context.lower())
        
        if matches_in_context > max_matches:
            max_matches = matches_in_context
            best_context = context
    
    return f"...{best_context}..."


@require_http_methods(["POST"])
@csrf_exempt  # Temporarily exempt from CSRF for testing
def search_documents(request: HttpRequest) -> JsonResponse:
    """Search for documents based on query parameters."""
    try:
        # Log the incoming request
        logger.info(f"Search request received - POST data: {request.POST}")
        
        # Get search parameters from request
        data = request.POST
        query = data.get('query', '').strip()
        method = data.get('method', 'cosine')
        file_type = data.get('fileType', 'all')
        date_range = data.get('dateRange', 'all')
        exact_match = data.get('exactMatch', 'false').lower() == 'true'

        logger.info(f"Search parameters: query='{query}', method={method}, file_type={file_type}, date_range={date_range}, exact_match={exact_match}")

        if not query:
            return JsonResponse({
                'error': 'Search query is required'
            }, status=400)

        # Start with all documents
        documents = Document.objects.all()
        logger.info(f"Total documents before filtering: {documents.count()}")

        # Apply file type filter
        if file_type != 'all':
            documents = documents.filter(file_type=file_type)
            logger.info(f"Documents after file type filter: {documents.count()}")

        # Apply date range filter
        if date_range != 'all':
            from django.utils import timezone
            from datetime import timedelta
            now = timezone.now()
            since = now  # Initialize with default value
            
            if date_range == 'day':
                since = now - timedelta(days=1)
            elif date_range == 'week':
                since = now - timedelta(weeks=1)
            elif date_range == 'month':
                since = now - timedelta(days=30)
            elif date_range == 'year':
                since = now - timedelta(days=365)
            
            documents = documents.filter(uploaded_at__gte=since)
            logger.info(f"Documents after date filter: {documents.count()}")

        # For exact match search
        if exact_match:
            matching_docs = []
            for doc in documents:
                # Search in both original content and processed content
                content_match = query.lower() in doc.content.lower()
                processed_match = query.lower() in doc.processed_content.lower()
                
                if content_match or processed_match:
                    # Find the best context that shows the match
                    context = find_best_context(doc.content, query)
                    
                    matching_docs.append({
                        'id': doc.id,
                        'title': doc.title,
                        'file_type': doc.file_type,
                        'uploaded_at': doc.uploaded_at.strftime("%Y-%m-%d %H:%M"),
                        'score': 100,  # Exact match gets 100%
                        'preview': context,
                        'size': f"{doc.file_size / 1024:.1f} KB" if doc.file_size else "N/A"
                    })
            logger.info(f"Exact match results: {len(matching_docs)}")
        else:
            # Create TF-IDF vectorizer with better parameters
            vectorizer = TfidfVectorizer(
                min_df=1,  # Include all terms
                max_df=0.9,  # Ignore terms that appear in more than 90% of docs
                stop_words='english',  # Remove English stop words
                ngram_range=(1, 2),  # Include both unigrams and bigrams
                strip_accents='unicode'  # Handle accented characters
            )
            
            # Get all document contents - combine original and processed content
            doc_contents = [f"{doc.content} {doc.processed_content}" for doc in documents]
            logger.info(f"Processing {len(doc_contents)} documents for TF-IDF")
            
            # Preprocess the query
            processed_query = preprocess_text(query)
            
            # Add query to the documents for vectorization
            all_texts = doc_contents + [processed_query]
            
            # Create TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Get query vector (last row of the matrix)
            query_vector = tfidf_matrix.getrow(-1)
            
            # Get document vectors (all rows except the last one)
            doc_vectors = vstack([tfidf_matrix.getrow(i) for i in range(tfidf_matrix.shape[0]-1)])
            
            # Calculate similarities
            if method == 'cosine':
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(doc_vectors.toarray(), query_vector.toarray())
            else:  # euclidean
                from sklearn.metrics.pairwise import euclidean_distances
                distances = euclidean_distances(doc_vectors.toarray(), query_vector.toarray())
                similarities = 1 / (1 + distances)  # Convert distance to similarity
            
            # Create results list
            matching_docs = []
            for doc, score in zip(documents, similarities):
                score_value = float(score[0])  # Get the score value
                if score_value > 0.1:  # Only include relevant results
                    # Find the best context that shows the match
                    context = find_best_context(doc.content, query)
                    
                    matching_docs.append({
                        'id': doc.id,
                        'title': doc.title,
                        'file_type': doc.file_type,
                        'uploaded_at': doc.uploaded_at.strftime("%Y-%m-%d %H:%M"),
                        'score': round(score_value * 100, 1),  # Convert to percentage
                        'preview': context,
                        'size': f"{doc.file_size / 1024:.1f} KB" if doc.file_size else "N/A"
                    })
            logger.info(f"TF-IDF search results: {len(matching_docs)}")

        # Sort results by score
        matching_docs.sort(key=lambda x: x['score'], reverse=True)
        
        response_data = {
            'results': matching_docs[:20],  # Limit to top 20 results
            'total': len(matching_docs),
            'query': query
        }
        logger.info(f"Returning {len(response_data['results'])} results")
        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error in search_documents: {str(e)}")
        logger.exception("Full traceback:")
        return JsonResponse({
            'error': 'An error occurred while searching documents'
        }, status=500)


@require_http_methods(["GET"])
def tfidf_similarity_view(_request: HttpRequest) -> JsonResponse:
    """
    Calculate and return TF-IDF similarity between documents.
    
    Args:
        _request: The HTTP request (unused)
        
    Returns:
        JsonResponse with similarity results or error
    """
    try:
        # Get all documents
        documents = Document.objects.all()
        
        # Check if there are any documents
        if not documents:
            return JsonResponse(
                {"error": "No documents available for processing."},
                status=404
            )

        # Extract processed content and titles
        processed_docs = [doc.processed_content for doc in documents]
        titles = [doc.title for doc in documents]

        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(min_df=1, max_df=0.9)
        tfidf_matrix = vectorizer.fit_transform(processed_docs)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Process results
        results = []
        for i, title in enumerate(titles):
            # Get similar documents for this document
            similar_docs = [
                {
                    "title": titles[j],
                    "similarity": round(float(score), 4),
                }
                for j, score in enumerate(similarity_matrix[i])
                if i != j and score > 0.0
            ]
            
            # Sort by similarity (highest first)
            similar_docs.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Add to results
            results.append({
                "title": title,
                "similar_documents": similar_docs[:5]  # Top 5 similar docs
            })

        # Return results
        return JsonResponse({"results": results})
    except Exception as e:
        # Log the error and return it
        logger.error("Error calculating similarity: %s", str(e))
        return JsonResponse({"error": str(e)}, status=500)


@transaction.atomic
def update_tfidf_vectors() -> None:
    """
    Update TF-IDF vectors for all documents.
    
    Raises:
        ValidationError: If updating vectors fails
    """
    try:
        # Get all documents
        documents = Document.objects.all()
        
        # If there are no documents, nothing to do
        if not documents:
            return

        # Extract processed content
        processed_docs = [doc.processed_content for doc in documents]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(min_df=1, max_df=0.9)
        tfidf_matrix = vectorizer.fit_transform(processed_docs)

        # Get feature names
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update each document's vector
        for i, doc in enumerate(documents):
            # Get the vector for this document
            vector = tfidf_matrix.getrow(i).toarray().flatten()
            
            # Update the document's vector
            doc.tfidf_vector = {
                'feature_names': feature_names,
                'vector': vector.tolist()
            }
            
            # Save the document
            doc.save(update_fields=['tfidf_vector'])
    except Exception as e:
        # Log the error and raise a ValidationError
        logger.error("Error updating TF-IDF vectors: %s", str(e))
        raise ValidationError("Failed to update TF-IDF vectors") from e


def process_results(request: HttpRequest) -> HttpResponse:
    """
    Display the results of document processing.
    
    Args:
        request: The HTTP request
        
    Returns:
        HttpResponse with rendered template
    """
    document_id = request.GET.get('id')
    
    if not document_id:
        return render(
            request,
            'indexer_app/process_results.html',
            {'error': 'No document ID provided'}
        )
    
    try:
        document = Document.objects.get(id=document_id)
        return render(
            request,
            'indexer_app/process_results.html',
            {'document': document}
        )
    except Document.DoesNotExist:
        return render(
            request,
            'indexer_app/process_results.html',
            {'error': 'Document not found'}
        )


def test_json_view(request: HttpRequest) -> JsonResponse:
    """
    A simple view that returns a JSON response for testing.
    
    Args:
        request: The HTTP request
        
    Returns:
        JsonResponse with test data
    """
    logger.info("Test JSON view called")
    return JsonResponse({
        'success': True,
        'message': 'Test JSON response',
        'timestamp': timezone.now().isoformat()
    })


def test_page_view(request: HttpRequest) -> HttpResponse:
    """
    Render the test page.
    
    Args:
        request: The HTTP request
        
    Returns:
        HttpResponse with rendered template
    """
    return render(request, 'indexer_app/test.html')


@require_http_methods(["GET"])
def test_upload_view(request: HttpRequest) -> JsonResponse:
    """
    A simple test endpoint to verify that the server is working correctly.
    
    Args:
        request: The HTTP request
        
    Returns:
        JsonResponse with test data
    """
    logger.info("Test upload view called")
    
    # Check if the upload directory exists
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        upload_dir_exists = UPLOAD_DIR.exists()
        upload_dir_absolute = UPLOAD_DIR.is_absolute()
        upload_dir_path = str(UPLOAD_DIR)
        
        # Try to create a test file to verify write permissions
        test_file_path = UPLOAD_DIR / 'test.txt'
        with open(test_file_path, 'w') as f:
            f.write('Test file')
        test_file_created = test_file_path.exists()
        if test_file_created:
            test_file_path.unlink()  # Delete the test file
    except Exception as e:
        upload_dir_exists = False
        upload_dir_absolute = False
        upload_dir_path = str(e)
        test_file_created = False
    
    # Check if the Document model is properly configured
    try:
        document_count = Document.objects.count()
        model_working = True
    except Exception as e:
        document_count = 0
        model_working = False
        logger.error(f"Error checking Document model: {str(e)}")
    
    # Return test data
    return JsonResponse({
        'success': True,
        'message': 'Test upload endpoint working',
        'timestamp': timezone.now().isoformat(),
        'upload_dir': {
            'exists': upload_dir_exists,
            'is_absolute': upload_dir_absolute,
            'path': upload_dir_path,
            'writable': test_file_created
        },
        'database': {
            'working': model_working,
            'document_count': document_count
        }
    })


def test_upload_page_view(request):
    """
    View function for the test upload page.
    This page provides diagnostic information and a simple file upload form.
    """
    return render(request, 'indexer_app/test_upload.html')


@require_http_methods(["GET"])
def test_plain_view(request: HttpRequest) -> HttpResponse:
    """
    A simple test view that returns a plain text response.
    
    Args:
        request: The HTTP request
        
    Returns:
        HttpResponse with plain text
    """
    return HttpResponse("Test view is working!", content_type="text/plain")


def test_bg_view(request: HttpRequest) -> HttpResponse:
    """
    View function for the background image test page.
    This page provides a simple test to verify that the background image is loading correctly.
    
    Args:
        request: The HTTP request
        
    Returns:
        HttpResponse with rendered template
    """
    return render(request, 'indexer_app/test_bg.html')


@csrf_protect
def try_view(request: HttpRequest) -> HttpResponse:
    """
    Render the try page.
    
    Args:
        request: The HTTP request
        
    Returns:
        HttpResponse with rendered template
    """
    logger.info("Rendering try page")
    return render(
        request,
        'indexer_app/try.html',
        {'title': 'Try AOS System'}
    )