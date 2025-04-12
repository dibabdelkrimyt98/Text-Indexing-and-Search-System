from django.db import models

class IndexedDocument(models.Model):
    """
    A simple model to keep track of documents that have been indexed.
    """
    file_name = models.CharField(max_length=255, unique=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # You could add more metadata if needed, e.g., file size, path

    def __str__(self):
        return self.file_name