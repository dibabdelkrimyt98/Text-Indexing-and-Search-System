"""Model definitions for indexed documents."""

from django.db import models


class IndexedDocument(models.Model):
    file_name: str = models.CharField(max_length=255, unique=True)
    uploaded_at: models.DateTimeField = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return str(self.file_name)
