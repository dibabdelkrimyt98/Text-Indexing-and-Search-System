from django import forms

class DocumentUploadForm(forms.Form):
    """
    Form for uploading multiple text documents.
    """
    # Allows multiple files to be uploaded via one input field
    documents = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}),
        label='Select .txt documents to index',
        required=True
    )

    # Optional: Add validation for file types (e.g., only allow .txt)
    def clean_documents(self):
        files = self.files.getlist('documents') # Use getlist for multiple files
        for file in files:
            if not file.name.lower().endswith('.txt'):
                raise forms.ValidationError(f"Only .txt files are allowed. Invalid file: {file.name}")
            # You could add size validation here too
        return files # Return the list of cleaned files