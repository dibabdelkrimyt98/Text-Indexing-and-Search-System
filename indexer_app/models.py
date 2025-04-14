from django.db import models
from django.utils import timezone
from django.core.exceptions import ValidationError
from django.core.validators import MinValueValidator
from typing import Dict, Any, cast
import json


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
    
    # File type choices
    FILE_TYPES: list[tuple[str, str]] = [
        ('txt', 'Text File'),
        ('pdf', 'PDF Document'),
        ('doc', 'Word Document'),
        ('docx', 'Word Document (DOCX)'),
    ]
    
    title: models.CharField = models.CharField(
        max_length=255,
        unique=True,
        db_index=True,
        help_text="Unique title for the document"
    )
    
    content: models.TextField = models.TextField(
        help_text="Original document content"
    )
    
    processed_content: models.TextField = models.TextField(
        blank=True,
        help_text="Preprocessed content for TF-IDF calculation"
    )
    
    uploaded_at: models.DateTimeField = models.DateTimeField(
        default=timezone.now,
        db_index=True,
        help_text="Timestamp of document upload"
    )
    
    tfidf_vector: models.JSONField = models.JSONField(
        null=True,
        blank=True,
        help_text="TF-IDF vector data stored as JSON"
    )
    
    file_type: models.CharField = models.CharField(
        max_length=10,
        choices=FILE_TYPES,
        default='txt',
        help_text="Type of the document file"
    )
    
    file_size: models.IntegerField = models.IntegerField(
        default=0,
        validators=[
            MinValueValidator(0),
            validate_file_size
        ],
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
        """String representation of the document."""
        return f"{self.title} ({self.file_type})"

    def clean(self) -> None:
        """
        Validate the model instance.
        Raises ValidationError for invalid data.
        """
        super().clean()
        if not self.title:
            raise ValidationError("Title is required")
        if not self.content:
            raise ValidationError("Content is required")
        if self.file_size < 0:
            raise ValidationError("File size cannot be negative")

    def save(self, *args: Any, **kwargs: Any) -> None:
        """Override save method to perform cleaning."""
        self.full_clean()
        super().save(*args, **kwargs)

    def get_vector_data(self) -> Dict[str, Any]:
        empty_dict: Dict[str, Any] = {}
        
        try:
            if isinstance(self.tfidf_vector, str):
                return cast(Dict[str, Any], json.loads(self.tfidf_vector))
            if isinstance(self.tfidf_vector, dict):
                return cast(Dict[str, Any], self.tfidf_vector)
            return empty_dict
        except (json.JSONDecodeError, TypeError):
            return empty_dict

    def get_file_extension(self) -> str:
        """
        Get the file extension for the document.
        
        Returns:
            str: File extension (e.g., '.txt', '.pdf')
        """
        return f".{self.file_type}"

    def get_file_size_display(self) -> str:
        """
        Get human-readable file size.
        
        Returns:
            str: Formatted file size (e.g., '2.5 MB')
        """
        size = float(self.file_size)  # Convert to float for division
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    @property
    def is_processed(self) -> bool:
        """
        Check if document has been processed.
        
        Returns:
            bool: True if document has processed content and TF-IDF vector
        """
        return bool(self.processed_content and self.tfidf_vector)

    @classmethod
    def get_supported_file_types(cls) -> list[str]:
        """
        Get list of supported file types.
        
        Returns:
            list[str]: List of supported file extensions
        """
        return [ext for ext, _ in cls.FILE_TYPES]

    def update_tfidf_vector(self, vector_data: Dict[str, Any]) -> None:
        """
        Update the TF-IDF vector data.
        
        Args:
            vector_data: Dictionary containing vector data
        Raises:
            ValidationError: If vector data is invalid
        """
        if not isinstance(vector_data, dict):
            raise ValidationError("Vector data must be a dictionary")
        
        try:
            self.tfidf_vector = vector_data
            self.save(update_fields=['tfidf_vector'])
        except Exception as e:
            raise ValidationError(
                f"Failed to update TF-IDF vector: {str(e)}"
            )