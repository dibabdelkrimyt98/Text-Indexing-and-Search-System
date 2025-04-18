"""
Document model for the indexing application.
"""

from django.db import models
from django.utils import timezone
from django.core.exceptions import ValidationError
from django.core.validators import MinValueValidator
from typing import Dict, Any, cast, ClassVar
from django.db.models.fields import (
    CharField, TextField, DateTimeField, IntegerField
)


def validate_file_size(value: int) -> None:
    """
    Validate file size is under 5MB.
    
    Args:
        value: File size in bytes
    Raises:
        ValidationError: If file size exceeds limit
    """
    max_size = 5 * 1024 * 1024  # 5MB
    max_size_mb = max_size / 1024 / 1024
    if value > max_size:
        raise ValidationError(
            f'File size cannot exceed {max_size_mb}MB'
        )


class Document(models.Model):
    """
    Document model for storing text documents and their processed data.
    
    Attributes:
        title (str): The document title (unique)
        content (str): The original document content
        processed_content (str): The preprocessed content for TF-IDF
        uploaded_at (datetime): Timestamp of upload
        tfidf_vector (dict): JSON field storing TF-IDF vector data
        file_type (str): Type of the document file
        file_size (int): Size of the document in bytes
    """
    
    # Type hint for the objects manager
    objects: ClassVar[models.Manager]
    
    # File type choices
    FILE_TYPES = [
        ('txt', 'Text File'),
        ('pdf', 'PDF Document'),
        ('doc', 'Word Document'),
        ('docx', 'Word Document (DOCX)'),
    ]
    
    # Fields
    title: CharField = models.CharField(
        max_length=255,
        unique=True,
        db_index=True,
        help_text="Unique title for the document"
    )
    
    content: TextField = models.TextField(
        help_text="Original document content"
    )
    
    processed_content: TextField = models.TextField(
        blank=True,
        help_text="Preprocessed content for TF-IDF calculation"
    )
    
    uploaded_at: DateTimeField = models.DateTimeField(
        default=timezone.now,
        db_index=True,
        help_text="Timestamp of document upload"
    )
    
    tfidf_vector: models.JSONField = models.JSONField(
        null=True,
        blank=True,
        help_text="TF-IDF vector data stored as JSON"
    )
    
    file_type: CharField = models.CharField(
        max_length=10,
        choices=FILE_TYPES,
        default='txt',
        help_text="Type of the document file"
    )
    
    file_size: IntegerField = models.IntegerField(
        validators=[MinValueValidator(1), validate_file_size],
        help_text="Size of the document in bytes"
    )
    
    class Meta:
        ordering = ['-uploaded_at']
        indexes = [
            models.Index(fields=['title']),
            models.Index(fields=['uploaded_at']),
            models.Index(fields=['file_type']),
        ]
        verbose_name = 'Document'
        verbose_name_plural = 'Documents'
    
    def __str__(self) -> str:
        """Return string representation of the document."""
        return str(self.title)
    
    def clean(self) -> None:
        """Validate the model before saving."""
        super().clean()
        if self.file_size and self.file_size > 5 * 1024 * 1024:  # 5MB
            raise ValidationError({
                'file_size': 'File size cannot exceed 5MB'
            })
    
    def save(self, *args: Any, **kwargs: Any) -> None:
        """Save the model with validation."""
        self.full_clean()
        super().save(*args, **kwargs)
    
    def get_vector_data(self) -> Dict[str, Any]:
        """Get the TF-IDF vector data."""
        if not self.tfidf_vector:
            return {}
        return cast(Dict[str, Any], self.tfidf_vector)
    
    def get_file_extension(self) -> str:
        """Get the file extension."""
        return f".{self.file_type}"
    
    def get_file_size_display(self) -> str:
        """Get a human-readable file size."""
        size_bytes = self.file_size
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    
    @property
    def is_processed(self) -> bool:
        """Check if the document has been processed."""
        return bool(
            self.processed_content and
            self.tfidf_vector is not None
        )
    
    @classmethod
    def get_supported_file_types(cls) -> list[str]:
        """Get list of supported file extensions."""
        return [f".{ft[0]}" for ft in cls.FILE_TYPES]
    
    def update_tfidf_vector(self, vector_data: Dict[str, Any]) -> None:
        """Update the TF-IDF vector data."""
        self.tfidf_vector = vector_data
        self.save(update_fields=['tfidf_vector'])