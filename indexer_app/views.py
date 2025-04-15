"""
Views for the document indexing application.
Handles document upload, processing, and similarity calculations.
"""

# Standard library imports
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, cast

# Third-party imports
import magic
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import spmatrix

# Django imports
from django.http import HttpResponse, JsonResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.core.exceptions import ValidationError
from django.conf import settings
from django.views.decorators.csrf import csrf_protect
from django.db import transaction
from django.core.files.uploadedfile import UploadedFile
from django.db.models.query import QuerySet
from django.db.models import Manager

# Local imports
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
MAX_FILE_SIZE: int = 5 * 1024 * 1024  # 5MB
ALLOWED_MIME_TYPES: set[str] = {
    'text/plain',
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}
ALLOWED_EXTENSIONS: set[str] = {'.txt', '.doc', '.docx', '.pdf'}

# Create upload directory if it doesn't exist
UPLOAD_DIR: Path = Path(settings.MEDIA_ROOT) / 'documents'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Type alias for Document manager
DocumentManager = Manager[Document]


def get_document_manager() -> DocumentManager:
    """Get the Document manager with proper typing."""
    return cast(DocumentManager, Document._default_manager)


def handle_uploaded_file(uploaded_file: UploadedFile) -> Tuple[str, str]:
    """
    Handle file upload and return content and file type.

    Args:
        uploaded_file: The uploaded file object

    Returns:
        Tuple containing (file_content, mime_type)

    Raises:
        ValidationError: If file type is not supported
        UnicodeDecodeError: If file encoding is not supported
    """
    content: str = ''
    file_path: Path = UPLOAD_DIR / str(uploaded_file.name)

    with open(file_path, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

    mime_type: str = magic.from_file(str(file_path), mime=True)

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
    """Render the welcome page."""
    try:
        return render(
            request,
            "indexer_app/index.html",
            {'title': 'Welcome to AOS System'}
        )
    except Exception as e:
        logger.error("Error in home view: %s", str(e))
        return HttpResponse("System Error", status=500)


@csrf_protect
def index(request: HttpRequest) -> HttpResponse:
    """Render the homepage with recent documents."""
    try:
        recent_documents: QuerySet[Document] = get_document_manager().all(
        ).order_by('-uploaded_at')[:10]
        return render(
            request,
            "indexer_app/index.html",
            {'documents': recent_documents}
        )
    except Exception as e:
        logger.error("Error in index view: %s", str(e))
        return HttpResponse("System Error", status=500)


@csrf_protect
def upload_document_view(request: HttpRequest) -> HttpResponse:
    """Handle document upload form display and submission."""
    try:
        if request.method == 'POST':
            return process_document_view(request)

        return render(
            request,
            "indexer_app/upload.html",
            {
                'max_file_size': MAX_FILE_SIZE,
                'allowed_types': [ext[1:] for ext in ALLOWED_EXTENSIONS]
            }
        )
    except Exception as e:
        logger.error("Error in upload view: %s", str(e))
        return JsonResponse({"error": str(e)}, status=500)


@csrf_protect
def search_view(request: HttpRequest) -> HttpResponse:
    """Render the search interface with recent documents."""
    try:
        recent_docs: QuerySet[Document] = get_document_manager().all(
        ).order_by('-uploaded_at')[:5]
        return render(
            request,
            "indexer_app/search.html",
            {'recent_documents': recent_docs}
        )
    except Exception as e:
        logger.error("Error in search view: %s", str(e))
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["POST"])
@csrf_protect
@transaction.atomic
def process_document_view(request: HttpRequest) -> HttpResponse:
    """Process uploaded documents and store them in the database."""
    try:
        if 'document' not in request.FILES:
            return JsonResponse(
                {"error": "No document provided"},
                status=400
            )

        uploaded_file = cast(UploadedFile, request.FILES['document'])
        file_size = getattr(uploaded_file, 'size', 0)

        if file_size > MAX_FILE_SIZE:
            size_mb = MAX_FILE_SIZE / (1024 * 1024)
            return JsonResponse(
                {"error": f"File size exceeds {size_mb}MB limit"},
                status=400
            )

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

        title = request.POST.get('title', uploaded_file.name)

        if get_document_manager().filter(title=title).exists():
            return JsonResponse(
                {"error": "A document with this title already exists"},
                status=400
            )

        try:
            content, _ = handle_uploaded_file(uploaded_file)
        except ValidationError as e:
            return JsonResponse({"error": str(e)}, status=400)
        except UnicodeDecodeError:
            return JsonResponse(
                {"error": "File encoding not supported. Use UTF-8."},
                status=400
            )

        processed_content = preprocess_text(content)

        document = get_document_manager().create(
            title=title,
            content=content,
            processed_content=processed_content,
            file_type=file_ext[1:],
            file_size=file_size
        )

        update_tfidf_vectors()

        return render(
            request,
            "indexer_app/process_results.html",
            {
                'document': document,
                'success': True,
                'message': 'Document processed successfully'
            }
        )

    except Exception as e:
        logger.error("Error processing document: %s", str(e))
        return JsonResponse(
            {"error": "An error occurred while processing the document"},
            status=500
        )


def preprocess_text(text: str) -> str:
    """
    Preprocess text by removing stopwords and normalizing.

    Args:
        text: The input text to process

    Returns:
        str: The processed text
    """
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
        logger.error("Text preprocessing failed: %s", str(e))
        raise ValidationError(f"Text preprocessing failed: {str(e)}") from e


@require_http_methods(["GET"])
def tfidf_similarity_view(_request: HttpRequest) -> JsonResponse:
    """Calculate and return TF-IDF similarity between documents."""
    try:
        documents: QuerySet[Document] = get_document_manager().all()
        if not documents:
            return JsonResponse(
                {"error": "No documents available for processing."},
                status=404
            )

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
                if i != j and score > 0.0
            ]
            similar_docs.sort(key=lambda x: x["similarity"], reverse=True)
            results.append({
                "title": title,
                "similar_documents": similar_docs[:5]
            })

        return JsonResponse({"results": results})
    except Exception as e:
        logger.error("Error calculating similarity: %s", str(e))
        return JsonResponse({"error": str(e)}, status=500)


@transaction.atomic
def update_tfidf_vectors() -> None:
    """Update TF-IDF vectors for all documents."""
    try:
        documents: QuerySet[Document] = get_document_manager().all()
        if not documents:
            return

        processed_docs = [doc.processed_content for doc in documents]
        vectorizer = TfidfVectorizer(min_df=1, max_df=0.9)
        tfidf_matrix: spmatrix = vectorizer.fit_transform(processed_docs)

        feature_names = vectorizer.get_feature_names_out().tolist()

        for i, doc in enumerate(documents):
            vector = tfidf_matrix.getrow(i).toarray().flatten()
            doc.tfidf_vector = {
                'feature_names': feature_names,
                'vector': vector.tolist()
            }
            doc.save(update_fields=['tfidf_vector'])
    except Exception as e:
        logger.error("Error updating TF-IDF vectors: %s", str(e))
        raise ValidationError("Failed to update TF-IDF vectors") from e
    
    
def process_results(request: HttpRequest) -> HttpResponse:
    document_id = request.GET.get('id')
    if not document_id or not document_id.isdigit():
        return render(request, 'indexer_app/process_results.html', {
            'success': False,
            'error_message': 'Invalid or missing document ID.',
        })

    try:
        document = Document.objects.get(id=document_id)
        return render(request, 'indexer_app/process_results.html', {
            'success': True,
            'document': document,
        })
    except Document.DoesNotExist:
        return render(request, 'indexer_app/process_results.html', {
            'success': False,
            'error_message': 'Document not found.',
        })