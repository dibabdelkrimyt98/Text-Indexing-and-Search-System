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
from django.views.decorators.csrf import csrf_protect
from django.db import transaction
from django.core.files.uploadedfile import UploadedFile

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


# Create upload directory if it doesn't exist
UPLOAD_DIR = Path(settings.MEDIA_ROOT) / 'documents'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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
    content = ''
    file_path = UPLOAD_DIR / str(uploaded_file.name)

    # Save the file to disk
    with open(file_path, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

    # For simplicity, we'll just handle text files for now
    mime_type = 'text/plain'

    # Read the file content
    if mime_type == 'text/plain':
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        raise ValidationError(
            f"File type {mime_type} processing not implemented yet"
        )

    return content, mime_type


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
    try:
        # Check if a document was uploaded
        if 'document' not in request.FILES:
            return JsonResponse(
                {"error": "No document provided"},
                status=400
            )

        # Get the uploaded file
        uploaded_file = request.FILES['document']
        
        # Validate that it's a proper UploadedFile
        if not isinstance(uploaded_file, UploadedFile):
            return JsonResponse(
                {"error": "Invalid file upload"},
                status=400
            )

        # Check file size
        file_size = uploaded_file.size
        if file_size is None or file_size > MAX_FILE_SIZE:
            size_mb = MAX_FILE_SIZE / (1024 * 1024)
            return JsonResponse(
                {"error": f"File size exceeds {size_mb}MB limit"},
                status=400
            )

        # Check file extension
        file_ext = Path(str(uploaded_file.name)).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            ext_list = ', '.join(ALLOWED_EXTENSIONS)
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
        title = request.POST.get('title', uploaded_file.name)

        # Check if a document with this title already exists
        if Document.objects.filter(title=title).exists():
            return JsonResponse(
                {"error": "A document with this title already exists"},
                status=400
            )

        # Process the file
        try:
            content, _ = handle_uploaded_file(uploaded_file)
        except ValidationError as e:
            return JsonResponse({"error": str(e)}, status=400)
        except UnicodeDecodeError:
            return JsonResponse(
                {"error": "File encoding not supported. Use UTF-8."},
                status=400
            )

        # Preprocess the text
        processed_content = preprocess_text(content)

        # Create the document in the database
        document = Document.objects.create(
            title=title,
            content=content,
            processed_content=processed_content,
            file_type=file_ext[1:],
            file_size=file_size
        )

        # Update TF-IDF vectors for all documents
        update_tfidf_vectors()

        # Return success response
        return JsonResponse({
            'success': True,
            'message': 'Document processed successfully',
            'document_id': document.id
        })

    except Exception as e:
        # Log the error and return a generic error message
        logger.error("Error processing document: %s", str(e))
        return JsonResponse(
            {"error": "An error occurred while processing the document"},
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
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [t for t in tokens if t not in stop_words]
    
    # Join tokens back into text
    return ' '.join(filtered_tokens)


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
    Handle document processing results.
    
    Args:
        request: The HTTP request
        
    Returns:
        HttpResponse with rendered template
    """
    # Get the document ID from the query parameters
    document_id = request.GET.get('id')
    
    # Check if the ID is valid
    if not document_id or not document_id.isdigit():
        return render(
            request,
            "indexer_app/process_results.html",
            {
                'success': False,
                'error_message': 'Invalid or missing document ID.',
            }
        )

    try:
        # Get the document
        document = Document.objects.get(id=document_id)
        
        # Return success response
        return render(
            request,
            'indexer_app/process_results.html',
            {
                'success': True,
                'document': document,
            }
        )
    except ObjectDoesNotExist:
        # Document not found
        return render(
            request,
            'indexer_app/process_results.html',
            {
                'success': False,
                'error_message': 'Document not found.',
            }
        )