from indexer_app.models import Document

# Get all documents
documents = Document.objects.all()
print(f"Total documents: {documents.count()}")

# Check vocabulary fields for the first 5 documents
for i, doc in enumerate(documents[:5]):
    print(f"\nDocument {i+1}: {doc.title}")
    print(f"Title vocabulary: {doc.title_vocabulary if doc.title_vocabulary else 'None'}")
    
    # Handle content vocabulary
    if doc.content_vocabulary:
        print(f"Content vocabulary: {doc.content_vocabulary[:20]}...")  # Show first 20 words
    else:
        print("Content vocabulary: None")
    
    # Handle word frequencies
    if doc.word_frequencies:
        print(f"Word frequencies: {list(doc.word_frequencies.items())[:10]}...")  # Show first 10 frequencies
    else:
        print("Word frequencies: None") 