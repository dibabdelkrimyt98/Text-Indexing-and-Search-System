from django import forms


class DocumentUploadForm(forms.Form):
    """
    Form for uploading multiple text documents.
    """
    documents = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}),
        label='Select .txt documents to index',
        required=True
    )

    def clean_documents(self):
        files = self.files.getlist('documents')
        if not files:
            raise forms.ValidationError("No files uploaded.")

        for file in files:
            if not file.name.lower().endswith('.txt'):
                raise forms.ValidationError(
                    f"Only .txt files are allowed: {file.name}"
                )
        return files
