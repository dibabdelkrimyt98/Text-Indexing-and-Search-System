"""Views for computing TF-IDF and document similarity in AOS System."""

from typing import List, Dict, Any
from django.http import JsonResponse, HttpRequest
from django.views.decorators.http import require_http_methods

# These packages raise Mypy import warnings due to lack of type stubs
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore

import nltk
nltk.download('punkt')
# Download NLTK corpora (should ideally be done once globally, not at runtime)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


def preprocess_text(text: str) -> str:
    """

    Args:
        text (str): Raw input string.

    Returns:
        str: Cleaned string with only relevant tokens.
    """
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)

    # Filter out stopwords and punctuation
    filtered = [
        word.lower()
        for word in words
        if word.isalpha() and word.lower() not in stop_words
    ]
    return " ".join(filtered)

@require_http_methods(["GET"])
def tfidf_similarity_view(request):
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method."}, status=405)

 
    dummy_titles: List[str] = ["Doc A", "Doc B", "Doc C"]
    dummy_contents: List[str] = [
        "Resistance is a right.",
        "The Palestinian struggle is about land and justice.",
        "Truth and resilience define our spirit.",
    ]

    # Step 1: Preprocess documents
    processed_docs: List[str] = [
        preprocess_text(doc)
            for doc in dummy_contents
    ]


    if not processed_docs:
        return JsonResponse({"error": "No documents to process."}, status=404)

    # Step 2: Generate TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_docs)

    # Step 3: Compute cosine similarity between all docs
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Step 4: Build similarity result
    results: List[Dict[str, Any]] = []
    for i, title in enumerate(dummy_titles):
        similar_docs = [
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
