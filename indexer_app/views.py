from typing import List, Dict, Any
from django.http import HttpResponse, JsonResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from django.core.exceptions import ValidationError

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")


def home_view(request: HttpRequest) -> HttpResponse:
    """
    Render the welcome page for AOS System.
    """
    return HttpResponse("Welcome to AOS System!")


def index(request: HttpRequest) -> HttpResponse:
    """
    Render the homepage.
    """
    return render(request, "indexer_app/index.html")


def upload_document_view(request: HttpRequest) -> HttpResponse:
    """
    Render the upload page for document submission.
    """
    return render(request, "indexer_app/upload.html")


def search_view(request: HttpRequest) -> HttpResponse:
    """
    Render the search interface page.
    """
    return render(request, "indexer_app/search.html")


def process_document_view(request: HttpRequest) -> HttpResponse:
    """
    Process uploaded documents and render the results page.
    
    Returns:
        HttpResponse: Rendered template with processing results
    """
    try:
        # Add document processing logic here
        return render(request, "indexer_app/process_results.html")
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def tfidf_table_page(request: HttpRequest) -> HttpResponse:
    """
    Render the TF-IDF table display page.
    """
    return render(request, "indexer_app/tfidf_table.html")


def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by removing stopwords 
    and converting to lowercase.
    
    Args:
        text (str): Input text to be preprocessed
        
    Returns:
        str: Preprocessed text
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
        raise ValidationError(f"Text preprocessing failed: {str(e)}")


@require_http_methods(["GET"])
def tfidf_similarity_view(request: HttpRequest) -> JsonResponse:
    """
    Calculate TF-IDF similarity between documents and return results.
    
    Returns:
        JsonResponse: JSON containing similarity results or error message
    """
    try:
        # In a real application, this data should come from a database
        dummy_titles: List[str] = ["Doc A", "Doc B", "Doc C"]
        dummy_contents: List[str] = [
            "Resistance is a right.",
            "The Palestinian struggle is about land and justice.",
            "Truth and resilience define our spirit.",
        ]

        processed_docs: List[str] = [
            preprocess_text(doc)
            for doc in dummy_contents
        ]

        if not processed_docs:
            return JsonResponse(
                {"error": "No documents available for processing."}, 
                status=404
            )

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_docs)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        results: List[Dict[str, Any]] = []
        for i, title in enumerate(dummy_titles):
            similar_docs: List[Dict[str, Any]] = [
                {
                    "title": dummy_titles[j],
                    "similarity": round(float(score), 4),
                }
                for j, score in enumerate(similarity_matrix[i])
                if i != j
            ]
            similar_docs.sort(key=lambda x: x["similarity"], reverse=True)
            results.append({
                "title": title,
                "similar_documents": similar_docs,
            })

        return JsonResponse({"results": results})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)