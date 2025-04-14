from typing import List, Dict, Any
from django.http import HttpResponse, JsonResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.core.exceptions import ValidationError
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.views.decorators.csrf import csrf_protect
from django.db import transaction

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os
import logging
import magic  # for file type detection
from pathlib import Path

from .models import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLTK - Put this in a try-except block to handle first-time setup
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

# Constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_FILE_TYPES = {'text/plain', 'application/pdf', 'application/msword', 
                     'application/vnd.openxmlformats-officedocument.wordprocessingml.document'}
ALLOWED_EXTENSIONS = {'.txt', '.doc', '.docx', '.pdf'}

# Create upload directory if it doesn't exist
UPLOAD_DIR = Path(settings.MEDIA_ROOT) / 'documents'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def handle_uploaded_file(uploaded_file) -> tuple[str, str]:
    """
    Handle file upload and return content and file type.
    """
    # Read file content
    content = ''
    file_path = UPLOAD_DIR / uploaded_file.name
    
    with open(file_path, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)
    
    # Detect file type
    mime_type = magic.from_file(str(file_path), mime=True)
    
    # Read content based on file type
    if mime_type == 'text/plain':
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        # For PDF and DOC files, you'll need to implement appropriate readers
        # This is a placeholder - you should implement proper document parsing
        raise ValidationError(f"File type {mime_type} processing not implemented yet")
    
    return content, mime_type


@csrf_protect
def home_view(request: HttpRequest) -> HttpResponse:
    """Render the welcome page for AOS System."""
    try:
        return render(request, "indexer_app/index.html", {
            'title': 'Welcome to AOS System'
        })
    except Exception as e:
        logger.error(f"Error in home view: {e}")
        return HttpResponse("System Error", status=500)


@csrf_protect
def index(request: HttpRequest) -> HttpResponse:
    """Render the homepage with recent documents."""
    try:
        recent_documents = Document.objects.all().order_by('-uploaded_at')[:10]
        return render(request, "indexer_app/index.html", {
            'documents': recent_documents
        })
    except Exception as e:
        logger.error(f"Error in index view: {e}")
        return HttpResponse("System Error", status=500)


@csrf_protect
def upload_document_view(request: HttpRequest) -> HttpResponse:
    """Handle document upload form display and submission."""
    try:
        if request.method == 'POST':
            return process_document_view(request)
        
        return render(request, "indexer_app/upload.html", {
            'max_file_size': MAX_FILE_SIZE,
            'allowed_types': [ext[1:] for ext in ALLOWED_EXTENSIONS]
        })
    except Exception as e:
        logger.error(f"Error in upload view: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_protect
def search_view(request: HttpRequest) -> HttpResponse:
    """Render the search interface with recent documents."""
    try:
        recent_docs = Document.objects.all().order_by('-uploaded_at')[:5]
        return render(request, "indexer_app/search.html", {
            'recent_documents': recent_docs
        })
    except Exception as e:
        logger.error(f"Error in search view: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["POST"])
@csrf_protect
@transaction.atomic
def process_document_view(request: HttpRequest) -> HttpResponse:
    """Process uploaded documents and store them in the database."""
    try:
        # Validate file upload
        if 'document' not in request.FILES:
            return JsonResponse({"error": "No document provided"}, status=400)

        uploaded_file = request.FILES['document']
        
        # Validate file size
        if uploaded_file.size > MAX_FILE_SIZE:
            return JsonResponse({
                "error": f"File size exceeds {MAX_FILE_SIZE/1024/1024}MB limit"
            }, status=400)
            
        # Validate file extension
        file_ext = Path(uploaded_file.name).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return JsonResponse({
                "error": f"File type not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"
            }, status=400)

        title = request.POST.get('title', uploaded_file.name)
        
        # Check for duplicate titles
        if Document.objects.filter(title=title).exists():
            return JsonResponse({
                "error": "A document with this title already exists"
            }, status=400)

        # Process the uploaded file
        try:
            content, mime_type = handle_uploaded_file(uploaded_file)
        except ValidationError as e:
            return JsonResponse({"error": str(e)}, status=400)
        except UnicodeDecodeError:
            return JsonResponse({
                "error": "File encoding not supported. Please upload UTF-8 encoded files."
            }, status=400)

        # Process document content
        processed_content = preprocess_text(content)
        
        # Create document
        document = Document.objects.create(
            title=title,
            content=content,
            processed_content=processed_content,
            file_type=file_ext[1:],  # Remove the dot from extension
            file_size=uploaded_file.size
        )

        # Update TF-IDF vectors
        update_tfidf_vectors()

        return render(request, "indexer_app/process_results.html", {
            'document': document,
            'success': True,
            'message': 'Document processed successfully'
        })

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return JsonResponse({
            "error": "An error occurred while processing the document"
        }, status=500)


def preprocess_text(text: str) -> str:
    """Preprocess text by removing stopwords and normalizing."""
    try:
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(text)
        filtered = [
            w.lower()
            for w in words
            if w.isalpha() and w.lower() not in stop_words
        ]
        return " ".join(filtered)
    except Exception as e:
        logger.error(f"Text preprocessing failed: {str(e)}", exc_info=True)
        raise ValidationError(f"Text preprocessing failed: {str(e)}")


@require_http_methods(["GET"])
def tfidf_similarity_view(request: HttpRequest) -> JsonResponse:
    """Calculate and return TF-IDF similarity between documents."""
    try:
        documents = Document.objects.all()
        if not documents:
            return JsonResponse({
                "error": "No documents available for processing."
            }, status=404)

        processed_docs = [doc.processed_content for doc in documents]
        titles = [doc.title for doc in documents]

        vectorizer = TfidfVectorizer(min_df=1, max_df=0.9)
        tfidf_matrix = vectorizer.fit_transform(processed_docs)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        results: List[Dict[str, Any]] = []
        for i, title in enumerate(titles):
            similar_docs = [
                {
                    "title": titles[j],
                    "similarity": round(float(score), 4),
                }
                for j, score in enumerate(similarity_matrix[i])
                if i != j and score > 0.0  # Only include non-zero similarities
            ]
            similar_docs.sort(key=lambda x: x["similarity"], reverse=True)
            results.append({
                "title": title,
                "similar_documents": similar_docs[:5]  # Limit to top 5 similar docs
            })

        return JsonResponse({"results": results})
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)


@transaction.atomic
def update_tfidf_vectors() -> None:
    """Update TF-IDF vectors for all documents."""
    try:
        documents = Document.objects.all()
        if not documents:
            return

        processed_docs = [doc.processed_content for doc in documents]
        vectorizer = TfidfVectorizer(min_df=1, max_df=0.9)
        tfidf_matrix = vectorizer.fit_transform(processed_docs)
        
        feature_names = vectorizer.get_feature_names_out().tolist()
        
        for i, doc in enumerate(documents):
            doc.tfidf_vector = {
                'feature_names': feature_names,
                'vector': tfidf_matrix[i].toarray().tolist()[0]
            }
            doc.save()
    except Exception as e:
        logger.error(f"Error updating TF-IDF vectors: {str(e)}", exc_info=True)
        raise