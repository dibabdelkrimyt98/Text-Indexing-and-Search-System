from indexer_app.models import Document

# Get all documents
documents = Document.objects.all()
print(f"Total documents: {documents.count()}")

# Update vocabulary fields for all documents
for i, doc in enumerate(documents):
    print(f"Updating document {i+1}: {doc.title}")
    doc.update_vocabulary()
    doc.save()
    print(f"Updated vocabulary for document: {doc.title}")

print("All documents updated successfully.")