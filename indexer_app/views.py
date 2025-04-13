from typing import List, Dict, Any
from django.http import JsonResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are available
import nltk

nltk.download("punkt")
nltk.download("stopwords")


# Dummy document model (you need to create this in models.py or remove this import if unused)
# from .models import Document


def preprocess_text(text: str) -> str:
    """
    Preprocesses input text: lowercasing, removing stopwords and non-alpha tokens.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned and preprocessed text.
    """
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered = [w.lower() for w in words if w.isalpha() and w.lower() not in stop_words]
    return " ".join(filtered)


def tfidf_similarity_view(request):
    """
    View that simulates TF-IDF computation and similarity for dummy content.

    Returns:
        JsonResponse: TF-IDF similarity data or error message.
    """
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method."}, status=405)

    dummy_titles = ["Doc A", "Doc B", "Doc C"]
    dummy_contents = [
        "Resistance is a right.",
        "The Palestinian struggle is about land and justice.",
        "Truth and resilience define our spirit.",
    ]

    # Apply preprocessing
    processed_docs: List[str] = [preprocess_text(doc) for doc in dummy_contents]

    if not processed_docs:
        return JsonResponse({"error": "No documents available."}, status=404)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    results: List[Dict[str, Any]] = []
    for i, title in enumerate(dummy_titles):
        similar_docs = []
        for j, score in enumerate(similarity_matrix[i]):
            if i != j:
                similar_docs.append({
                    "title": dummy_titles[j],
                    "similarity": round(float(score), 4),
                })
        similar_docs.sort(key=lambda x: x["similarity"], reverse=True)
        results.append({
            "title": title,
            "similar_documents": similar_docs
        })

    return JsonResponse({"results": results})
