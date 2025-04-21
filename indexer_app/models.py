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
from django.contrib.postgres.fields import ArrayField
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from django.db.models import JSONField

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
        title_vocabulary (list): List of processed words from the title
        content_vocabulary (list): List of processed words from the content
        word_frequencies (dict): Word frequencies in the document
    """
    
    # File type choices
    FILE_TYPES = [
        ('txt', 'Text File'),
        ('pdf', 'PDF Document'),
        ('doc', 'Word Document'),
        ('docx', 'Word Document (DOCX)'),
    ]
    
    # Fields
    id: models.AutoField = models.AutoField(primary_key=True)
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
    
    # Modified vocabulary fields to use JSONField
    title_vocabulary = models.JSONField(
        default=list,
        blank=True,
        null=True,
        help_text="Processed words from the title"
    )
    content_vocabulary = models.JSONField(
        default=list,
        blank=True,
        null=True,
        help_text="Processed words from the content"
    )
    word_frequencies = models.JSONField(
        default=dict,
        help_text="Word frequencies in the document"
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
        try:
            self.full_clean()
            # Update vocabulary before saving
            self.update_vocabulary()
            super().save(*args, **kwargs)
        except Exception as e:
            raise ValidationError(f"Error saving document: {str(e)}")
    
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

    def extract_vocabulary(self, text: str) -> list:
        """Extract vocabulary from text, removing stopwords."""
        try:
            # Tokenize and clean
            tokens = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            # Keep only alphanumeric words that aren't stopwords
            words = [
                word for word in tokens 
                if word.isalnum() and word not in stop_words
            ]
            return list(set(words))  # Remove duplicates
        except Exception as e:
            print(f"Error extracting vocabulary: {e}")
            return []

    def update_vocabulary(self):
        """Update document vocabulary fields."""
        # Process title vocabulary
        self.title_vocabulary = self.extract_vocabulary(self.title)
        
        # Process content vocabulary
        content_words = self.extract_vocabulary(self.content)
        self.content_vocabulary = content_words
        
        # Update word frequencies
        from collections import Counter
        word_counts = Counter(
            word_tokenize(self.content.lower())
        )
        self.word_frequencies = {
            word: count for word, count in word_counts.items()
            if word.isalnum()
        }