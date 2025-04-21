"""
Views for the document indexing application.
"""
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import nltk
from nltk.corpus import stopwords, wordnet
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
from scipy.sparse import vstack, csr_matrix
from nltk.stem import PorterStemmer
from collections import Counter
import math
from difflib import SequenceMatcher
from nltk.corpus.reader.wordnet import Synset

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
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
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
    """Handle uploaded file and extract its content."""
    file_path = UPLOAD_DIR / uploaded_file.name
    try:
        # Save the file temporarily
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        # Determine mime type and extract content based on file extension
        if file_ext == '.txt':
            mime_type = 'text/plain'
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                raise ValidationError("File must be UTF-8 encoded")
        elif file_ext in ['.doc', '.docx']:
            mime_type = 'application/msword'
            try:
                import docx
                doc = docx.Document(str(file_path))
                content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            except ImportError:
                raise ValidationError("python-docx package is required for Word documents")
        elif file_ext == '.pdf':
            mime_type = 'application/pdf'
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    content = '\n'.join(page.extract_text() for page in pdf_reader.pages)
            except ImportError:
                raise ValidationError("PyPDF2 package is required for PDF documents")
        else:
            raise ValidationError(f"File type {file_ext} not supported")

        # Clean up temporary file
        file_path.unlink()
        
        # Clean and normalize the content
        content = ' '.join(content.split())
        return content, mime_type

    except Exception as e:
        # Clean up temporary file if it exists
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        raise ValidationError(f"Error processing file: {str(e)}")


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
    """Process uploaded documents and store them in the database."""
    logger.info("=== Starting document processing ===")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request FILES: {request.FILES}")
    logger.info(f"Request POST: {request.POST}")
    logger.info(f"Content Type: {request.content_type}")
    logger.info(f"Headers: {request.headers}")
    
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
                    "error": f"File type not allowed. Supported types: {ext_list}"
                },
                status=400
            )

        # Ensure upload directory exists
        try:
            logger.info(f"Creating upload directory: {UPLOAD_DIR}")
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
            logger.info("Starting file processing")
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

        # Get the title from the form or use the filename
        base_title = request.POST.get('title', uploaded_file.name)
        logger.info(f"Using base title: {base_title}")

        # Create unique title if needed
        title = base_title
        counter = 1
        while Document.objects.filter(title=title).exists():
            timestamp = timezone.now().strftime("%Y%m%d_%H%M%S")
            title = f"{Path(base_title).stem}_{timestamp}{Path(base_title).suffix}"
            logger.info(f"Title already exists, using new title: {title}")
            counter += 1
            if counter > 10:
                logger.error("Too many attempts to create a unique title")
                return JsonResponse(
                    {"error": "Could not create a unique title for the document"},
                    status=500
                )

        # Create the document
        try:
            logger.info("Creating document in database")
            # Calculate word frequencies from the content
            word_frequencies = dict(extract_document_vocabulary(content))
            
            document = Document.objects.create(
                title=title,
                content=content,
                processed_content=preprocess_text(content),
                file_type=file_ext[1:] if file_ext.startswith('.') else file_ext,
                file_size=file_size,
                word_frequencies=word_frequencies
            )
            document.save()
            logger.info(f"Document created with ID: {document.id}")
        except Exception as e:
            logger.error(f"Error creating document: {str(e)}")
            return JsonResponse(
                {"error": f"Error saving document to database: {str(e)}"},
                status=500
            )

        # Return success response
        response_data = {
            'success': True,
            'message': 'Document processed successfully',
            'document_id': document.id,
            'title': title
        }
        logger.info(f"Upload successful: {response_data}")
        return JsonResponse(response_data)

    except Exception as e:
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
    word_frequencies = {
        word: count for word, count in Counter(tokens).items()
        if word.isalnum()
    }
    
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
        # If no exact matches, return the beginning of the content
        return content[:200] + "..."
    
    # Find the best context window
    best_context = ""
    max_matches = 0
    best_start = 0
    best_end = 0
    
    for match in matches:
        start = max(0, match - window_size)
        end = min(len(content), match + len(query) + window_size)
        context = content[start:end]
        context_lower = context.lower()
        
        # Count how many query occurrences appear in this context
        matches_in_context = context_lower.count(query_lower)
        
        if matches_in_context > max_matches:
            max_matches = matches_in_context
            best_context = context
            best_start = start
            best_end = end
    
    # Highlight the matches in the context
    highlighted_context = best_context
    for match in matches:
        if match >= best_start and match < best_end:
            # Calculate the position in the context
            context_pos = match - best_start
            # Add HTML highlighting
            highlighted_context = (
                highlighted_context[:context_pos] +
                f"<mark>{highlighted_context[context_pos:context_pos + len(query)]}</mark>" +
                highlighted_context[context_pos + len(query):]
            )
    
    return f"...{highlighted_context}..."


@require_http_methods(["POST"])
@csrf_exempt
def search_documents(request: HttpRequest) -> JsonResponse:
    """Search documents using case-insensitive substring matching."""
    try:
        data = request.POST
        query = data.get('query', '').strip()
        file_type = data.get('fileType', 'all')
        date_range = data.get('dateRange', 'all')
        exact_match = data.get('exactMatch', 'false').lower() == 'true'

        if not query:
            return JsonResponse({'error': 'Search query is required'}, status=400)

        # Debug log the search parameters
        logger.info(f"Search query: {query}")
        logger.info(f"File type filter: {file_type}")
        logger.info(f"Date range filter: {date_range}")
        logger.info(f"Exact match: {exact_match}")

        # Start with all documents
        documents = Document.objects.all()
        logger.info(f"Total documents in database: {documents.count()}")

        # Apply filters
        if file_type != 'all':
            documents = documents.filter(file_type=file_type)
        if date_range != 'all':
            from django.utils import timezone
            from datetime import timedelta
            now = timezone.now()
            date_filters = {
                'day': timedelta(days=1),
                'week': timedelta(weeks=1),
                'month': timedelta(days=30),
                'year': timedelta(days=365)
            }
            if date_range in date_filters:
                documents = documents.filter(
                    uploaded_at__gte=now - date_filters[date_range]
                )
        
        logger.info(f"Documents after filtering: {documents.count()}")

        # Calculate scores for each document
        results = []
        query_lower = query.lower()
        
        for doc in documents:
            logger.info(f"\nChecking document: {doc.title}")
            
            score = 0
            matching_terms = []
            
            # Check title for matches (higher weight for title matches)
            title_lower = doc.title.lower()
            if query_lower in title_lower:
                # Count occurrences in title
                title_count = title_lower.count(query_lower)
                score += title_count * 10  # Higher weight for title matches
                logger.info(f"Title matches: {title_count} occurrences")
                
                matching_terms.append({
                    'term': query,
                    'count': title_count,
                    'in_title': True,
                    'score': title_count * 10
                })
            
            # Check content for matches
            content_lower = doc.content.lower()
            if query_lower in content_lower:
                # Count occurrences in content
                content_count = content_lower.count(query_lower)
                score += content_count  # Lower weight for content matches
                logger.info(f"Content matches: {content_count} occurrences")
                
                # Add to matching terms if not already added
                if not matching_terms:
                    matching_terms.append({
                        'term': query,
                        'count': content_count,
                        'in_title': False,
                        'score': content_count
                    })
                else:
                    # Update the existing matching term
                    matching_terms[0]['count'] += content_count
                    matching_terms[0]['score'] += content_count
            
            logger.info(f"Final score: {score}")
            
            if score > 0:
                # Find context showing matches
                context = find_best_context(doc.content, query)
                
                results.append({
                    'id': doc.id,
                    'title': doc.title,
                    'score': round(score, 2),
                    'matching_terms': matching_terms,
                    'context': context,
                    'file_type': doc.file_type,
                    'size': format_file_size(doc.file_size),
                    'uploaded_at': doc.uploaded_at.strftime("%Y-%m-%d %H:%M")
                })

        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Log final results
        logger.info(f"Total results found: {len(results)}")
        
        # Prepare response
        response_data = {
            'results': results[:20],  # Top 20 results
            'total': len(results),
            'query': query,
            'query_terms': [query]  # Simplified for substring matching
        }
        
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


def extract_document_vocabulary(document_content: str) -> Counter:
    """
    Extract vocabulary and word frequencies from document content.
    
    Args:
        document_content: The text content of the document
        
    Returns:
        Counter object with word frequencies
    """
    # Tokenize and clean the text
    tokens = word_tokenize(document_content.lower())
    # Remove stopwords and non-alphanumeric tokens
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.isalnum() and word not in stop_words]
    # Count word frequencies
    return Counter(words)


def tokenize_and_stem(text: str) -> List[Tuple[str, str]]:
    """
    Tokenize text and get word stems.
    
    Args:
        text: Input text to process
        
    Returns:
        List of tuples containing (original_word, stemmed_word)
    """
    stemmer = PorterStemmer()
    # Tokenize and clean
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    # Get both original and stemmed forms
    word_stems = [
        (word, stemmer.stem(word))
        for word in tokens
        if word.isalnum() and word not in stop_words
    ]
    return word_stems


def format_file_size(size_in_bytes: float) -> str:
    """
    Format file size from bytes to human readable format.
    
    Args:
        size_in_bytes: File size in bytes
        
    Returns:
        Formatted string (e.g., '1.5 MB')
    """
    size = float(size_in_bytes)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def get_related_terms(word: str) -> set:
    """Get related terms using WordNet synonyms and similar words."""
    related_terms = {word.lower()}  # Initialize with original word
    
    try:
        for syn in wordnet.synsets(word):
            if isinstance(syn, Synset):
                # Add synonyms
                lemmas = syn.lemmas()
                if lemmas:
                    related_terms.update(
                        lemma.name().lower() 
                        for lemma in lemmas
                        if lemma is not None
                    )
                
                # Add hypernyms
                hypernyms = syn.hypernyms()
                if hypernyms:
                    related_terms.update(
                        lemma.name().lower()
                        for h in hypernyms
                        if isinstance(h, Synset)
                        for lemma in (h.lemmas() or [])
                        if lemma is not None
                    )
                
                # Add hyponyms
                hyponyms = syn.hyponyms()
                if hyponyms:
                    related_terms.update(
                        lemma.name().lower()
                        for h in hyponyms
                        if isinstance(h, Synset)
                        for lemma in (h.lemmas() or [])
                        if lemma is not None
                    )
    except Exception as e:
        logger.warning(f"Error getting related terms for '{word}': {str(e)}")
    
    return related_terms


def calculate_word_similarity(word1: str, word2: str) -> float:
    """
    Calculate similarity between two words using sequence matching.
    
    Args:
        word1: First word
        word2: Second word
        
    Returns:
        Similarity score between 0 and 1
    """
    return SequenceMatcher(None, word1.lower(), word2.lower()).ratio()


def calculate_tfidf_scores(query: str, documents: List[Document]) -> List[Dict]:
    """
    Calculate TF-IDF scores and rank documents using enhanced matching.
    
    Args:
        query: Search query
        documents: List of Document objects
        
    Returns:
        List of dictionaries containing document info and scores
    """
    # Get document vocabularies
    doc_vocabularies = {
        doc.id: extract_document_vocabulary(doc.content)
        for doc in documents
    }
    
    # Calculate document frequency for each term
    term_doc_freq = Counter()
    for vocab in doc_vocabularies.values():
        term_doc_freq.update(set(vocab.keys()))
    
    # Get query terms and their related terms
    query_terms = set()
    for word, _ in tokenize_and_stem(query):
        query_terms.add(word)
        # Add related terms
        query_terms.update(get_related_terms(word))
    
    # Calculate scores for each document
    results = []
    total_docs = len(documents)
    
    for doc in documents:
        doc_vocab = doc_vocabularies[doc.id]
        score = 0
        matching_terms = []
        
        # Check each word in document against query terms
        for doc_word in doc_vocab:
            # Find best matching query term
            best_match_score = 0
            best_match_term = None
            
            for query_term in query_terms:
                # Calculate similarity between document word and query term
                similarity = calculate_word_similarity(doc_word, query_term)
                if similarity > 0.8:  # Threshold for similarity
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_term = query_term
            
            if best_match_term:
                # Calculate TF-IDF score with similarity factor
                tf = doc_vocab[doc_word] / sum(doc_vocab.values())
                idf = math.log(total_docs / (term_doc_freq[doc_word] + 1))
                term_score = tf * idf * best_match_score
                score += term_score
                
                # Store matching term info
                matching_terms.append({
                    'term': doc_word,
                    'matched_query': best_match_term,
                    'similarity': round(best_match_score * 100, 1),
                    'count': doc_vocab[doc_word],
                    'score': round(term_score, 4)
                })
        
        if score > 0:
            # Find best context showing matches
            context = find_best_context(doc.content, ' '.join(query_terms))
            
            # Calculate final document score
            final_score = score * 100
            
            results.append({
                'id': doc.id,
                'title': doc.title,
                'score': round(final_score, 2),
                'matching_terms': matching_terms,
                'context': context,
                'file_type': doc.file_type,
                'size': format_file_size(doc.file_size),
                'uploaded_at': doc.uploaded_at.strftime("%Y-%m-%d %H:%M")
            })
    
    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


@require_http_methods(["POST"])
@csrf_exempt
def analyze_document_vocabulary(request: HttpRequest) -> JsonResponse:
    """
    Analyze document vocabulary and return detailed statistics.
    
    Args:
        request: HTTP request with document ID
        
    Returns:
        JsonResponse with vocabulary analysis
    """
    try:
        doc_id = request.POST.get('document_id')
        if not doc_id:
            return JsonResponse({'error': 'Document ID required'}, status=400)
            
        document = Document.objects.get(id=doc_id)
        
        # Get vocabulary statistics
        vocab = extract_document_vocabulary(document.content)
        
        # Get stemmed forms
        word_stems = tokenize_and_stem(document.content)
        stem_groups = {}
        for word, stem in word_stems:
            if stem not in stem_groups:
                stem_groups[stem] = []
            stem_groups[stem].append(word)
        
        # Prepare response
        analysis = {
            'document_id': document.id,
            'title': document.title,
            'total_words': sum(vocab.values()),
            'unique_words': len(vocab),
            'vocabulary': [
                {'word': word, 'frequency': freq}
                for word, freq in vocab.most_common(50)
            ],
            'word_stems': [
                {
                    'stem': stem,
                    'variations': list(set(words)),
                    'total_occurrences': sum(vocab[word] for word in words)
                }
                for stem, words in stem_groups.items()
            ]
        }
        
        return JsonResponse({'analysis': analysis})
        
    except Document.DoesNotExist:
        return JsonResponse({'error': 'Document not found'}, status=404)
    except Exception as e:
        logger.error(f"Error analyzing document: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["POST"])
@csrf_exempt
def get_tfidf_matrix(request: HttpRequest) -> JsonResponse:
    """Calculate and return the TF-IDF matrix for a given query."""
    try:
        query = request.POST.get('query', '').strip()
        top_n = int(request.POST.get('top_n', 10))  # Number of top keywords to show
        
        if not query:
            return JsonResponse({'error': 'Search query is required'}, status=400)
            
        # Get all documents
        documents = Document.objects.all()
        
        if not documents.exists():
            return JsonResponse({
                'matrix': [],
                'documents': [],
                'keywords': [],
                'query': query
            })
            
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000
        )
        
        # Prepare documents for vectorization
        doc_contents = [doc.processed_content for doc in documents]
        doc_contents.append(query)  # Add query to the corpus
        
        # Fit and transform the documents
        tfidf_matrix = vectorizer.fit_transform(doc_contents)
        
        # Get feature names (keywords)
        feature_names = vectorizer.get_feature_names_out().tolist()
        
        # Get top N keywords based on query vector
        query_vector = tfidf_matrix.getrow(-1).toarray()[0]
        top_keyword_indices = np.argsort(query_vector)[-top_n:][::-1]
        top_keywords = [feature_names[i] for i in top_keyword_indices]
        
        # Create a matrix of document vectors with only the top keywords
        matrix_data = []
        for i, doc in enumerate(documents):
            doc_vector = tfidf_matrix.getrow(i).toarray()[0]
            doc_keyword_values = []
            for kw in top_keywords:
                try:
                    idx = feature_names.index(kw)
                    doc_keyword_values.append(float(doc_vector[idx]))
                except (ValueError, IndexError):
                    doc_keyword_values.append(0.0)
                    
            matrix_data.append({
                'id': doc.id,
                'title': doc.title,
                'values': doc_keyword_values
            })
            
        # Get query vector values for top keywords
        query_values = []
        for kw in top_keywords:
            try:
                idx = feature_names.index(kw)
                query_values.append(float(query_vector[idx]))
            except (ValueError, IndexError):
                query_values.append(0.0)
        
        return JsonResponse({
            'matrix': matrix_data,
            'documents': [{'id': doc.id, 'title': doc.title} for doc in documents],
            'keywords': top_keywords,
            'query_values': query_values,
            'query': query
        })
        
    except Exception as e:
        logger.error(f"Error in get_tfidf_matrix: {str(e)}")
        logger.exception("Full traceback:")
        return JsonResponse({
            'error': 'An error occurred while calculating the TF-IDF matrix'
        }, status=500)