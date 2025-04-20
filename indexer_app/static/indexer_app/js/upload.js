// Show selected files
document.getElementById('fileInput').addEventListener('change', function () {
    const fileList = document.getElementById('fileList');
    fileList.innerHTML = '';
    for (let file of this.files) {
      const li = document.createElement('li');
      li.textContent = file.name;
      fileList.appendChild(li);
    }
    document.getElementById('indexButton').disabled = this.files.length === 0;
});

// Handle file upload and indexing
document.getElementById('indexButton').addEventListener('click', async () => {
    const fileInput = document.getElementById('fileInput');
    const files = fileInput.files;
    
    if (files.length === 0) {
        alert('Please select files to upload');
        return;
    }

    const formData = new FormData();
    for (let file of files) {
        formData.append('document', file);
        formData.append('title', file.name);
    }

    try {
        const csrfToken = getCookie('csrftoken');
        if (!csrfToken) {
            throw new Error('CSRF token not found. Please refresh the page and try again.');
        }

        const response = await fetch('/process/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': csrfToken
            },
            credentials: 'same-origin'
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        if (result.success) {
            const dialog = document.getElementById('successDialog');
            dialog.classList.add('show');
            dialog.dataset.documentId = result.document_id;
        } else {
            throw new Error(result.error || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Error uploading files: ' + error.message);
    }
});

// Close dialog
document.getElementById('closeDialog').addEventListener('click', () => {
    const dialog = document.getElementById('successDialog');
    dialog.classList.remove('show');
    const documentId = dialog.dataset.documentId;
    if (documentId) {
        window.location.href = `/results/?id=${documentId}`;
    }
});

// Helper function to get CSRF token
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
  
  