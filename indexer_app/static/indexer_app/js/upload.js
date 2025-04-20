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
        console.log('CSRF Token:', csrfToken);
        
        // Log the form data for debugging
        console.log('Form data entries:');
        for (let pair of formData.entries()) {
            console.log(pair[0] + ': ' + pair[1]);
        }
        
        const response = await fetch('/process/', {
            method: 'POST',
            body: formData,
            // Temporarily disable CSRF for testing
            // headers: {
            //     'X-CSRFToken': csrfToken
            headers: {
                'X-CSRFToken': csrfToken
            }
        });

        // Debug: Log the response text
        const responseText = await response.text();
        console.log('Server response:', responseText);
        
        // Try to parse as JSON
        let result;
        try {
            result = JSON.parse(responseText);
        } catch (e) {
            console.error('JSON parse error:', e);
            throw new Error('Server returned invalid JSON: ' + responseText.substring(0, 100));
        }

        if (!response.ok) {
            throw new Error(result.error || 'Upload failed');
        }

        if (result.success) {
            const dialog = document.getElementById('successDialog');
            dialog.classList.add('show');
            // Store the document ID for redirection
            dialog.dataset.documentId = result.document_id;
        } else {
            throw new Error('Upload failed');
        }
    } catch (error) {
        alert(error.message);
    }
});

// Close dialog
document.getElementById('closeDialog').addEventListener('click', () => {
    const dialog = document.getElementById('successDialog');
    dialog.classList.remove('show');
    // Redirect to results page with document ID
    const documentId = dialog.dataset.documentId;
    window.location.href = `/results/?id=${documentId}`;
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
  
  