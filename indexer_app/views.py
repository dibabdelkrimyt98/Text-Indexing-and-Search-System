from typing import List, Dict, Any
from django.http import HttpResponse, JsonResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk

nltk.download("punkt")
nltk.download("stopwords")


def index(request: HttpRequest) -> HttpResponse:
    """Render the homepage."""
    return render(request, "indexer_app/index.html")


def upload_page(request: HttpRequest) -> HttpResponse:
    """Render the upload page."""
    return render(request, "indexer_app/upload.html")


def search_page(request: HttpRequest) -> HttpResponse:
    """Render the search page."""
    return render(request, "indexer_app/search.html")


def tfidf_table_page(request: HttpRequest) -> HttpResponse:
    """Render the TF-IDF table display page."""
    return render(request, "indexer_app/tfidf_table.html")


def preprocess_text(text: str) -> str:
    """Remove stopwords and tokenize."""
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered = [
        w.lower()
        for w in words
        if w.isalpha() and w.lower() not in stop_words
    ]   
    return " ".join(filtered)


@require_http_methods(["GET"])
def tfidf_similarity_view(request: HttpRequest) -> JsonResponse:
    """
    """
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
        return JsonResponse({"error": "No documents available."}, status=404)

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