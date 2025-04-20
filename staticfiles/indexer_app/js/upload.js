// Store selected files
let selectedFiles = [];

// Get DOM elements
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const indexButton = document.getElementById('indexButton');
const fileCount = document.querySelector('.file-count');
const uploadBox = document.getElementById('uploadBox');
const successDialog = document.getElementById('successDialog');
const closeDialog = document.getElementById('closeDialog');

// Handle file selection
fileInput.addEventListener('change', (e) => {
  const files = Array.from(e.target.files);
  
  // Add new files to selectedFiles array
  files.forEach(file => {
    // Check if file is already selected
    if (!selectedFiles.some(f => f.name === file.name)) {
      selectedFiles.push(file);
    }
  });
  
  updateFileList();
  
  // Clear the file input so the same file can be selected again
  fileInput.value = '';
});

// Update file list display
function updateFileList() {
  fileList.innerHTML = '';
  fileCount.textContent = `${selectedFiles.length} file${selectedFiles.length !== 1 ? 's' : ''} selected`;
  
  selectedFiles.forEach((file, index) => {
    const li = document.createElement('li');
    li.className = 'file-item';
    li.dataset.filename = file.name; // Add filename as data attribute for easier selection
    
    const fileName = document.createElement('span');
    fileName.className = 'file-name';
    fileName.textContent = file.name;
    
    const fileSize = document.createElement('span');
    fileSize.className = 'file-size';
    fileSize.textContent = formatFileSize(file.size);
    
    const deleteButton = document.createElement('button');
    deleteButton.className = 'delete-file';
    deleteButton.innerHTML = '<i class="fas fa-times"></i>';
    deleteButton.onclick = () => removeFile(index);
    
    li.appendChild(fileName);
    li.appendChild(fileSize);
    li.appendChild(deleteButton);
    fileList.appendChild(li);
  });
  
  // Enable/disable index button
  indexButton.disabled = selectedFiles.length === 0;
}

// Remove file from selection
function removeFile(index) {
  selectedFiles.splice(index, 1);
  updateFileList();
}

// Format file size
function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Handle file indexing
indexButton.addEventListener('click', async () => {
  if (selectedFiles.length === 0) return;
  
  console.log('Index button clicked');
  console.log('Selected files:', selectedFiles);
  
  indexButton.disabled = true;
  indexButton.textContent = 'Indexing...';
  
  // Get CSRF token from the meta tag
  const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
  console.log('CSRF Token:', csrfToken);
  
  if (!csrfToken) {
    console.error('CSRF token not found');
    alert('Security token not found. Please refresh the page and try again.');
    indexButton.disabled = false;
    indexButton.textContent = 'Index Files';
    return;
  }
  
  const results = {
    success: 0,
    failed: 0
  };
  
  // Process files sequentially to avoid overwhelming the server
  for (const file of selectedFiles) {
    const formData = new FormData();
    formData.append('document', file);
    
    try {
      console.log('Uploading file:', file.name);
      console.log('File size:', file.size);
      console.log('File type:', file.type);
      
      // Use the correct URL with the app prefix
      const response = await fetch('/indexer_app/process/', {
        method: 'POST',
        body: formData,
        headers: {
          'X-CSRFToken': csrfToken
        }
      });
      
      console.log('Response status:', response.status);
      
      let data;
      try {
        data = await response.json();
        console.log('Server response:', data);
      } catch (jsonError) {
        console.error('Error parsing JSON response:', jsonError);
        const textResponse = await response.text();
        console.log('Raw response text:', textResponse);
        throw new Error('Invalid JSON response from server');
      }
      
      // Find the file item using the data attribute
      const fileItem = fileList.querySelector(`li[data-filename="${file.name}"]`);
      
      if (response.ok && data.success) {
        results.success++;
        if (fileItem) {
          fileItem.classList.add('success');
          const successIcon = document.createElement('i');
          successIcon.className = 'fas fa-check success-icon';
          fileItem.appendChild(successIcon);
          
          // Add the actual title used by the server
          if (data.title && data.title !== file.name) {
            const titleInfo = document.createElement('span');
            titleInfo.className = 'title-info';
            titleInfo.textContent = `Saved as: ${data.title}`;
            fileItem.appendChild(titleInfo);
          }
        }
      } else {
        results.failed++;
        if (fileItem) {
          fileItem.classList.add('error');
          const errorIcon = document.createElement('i');
          errorIcon.className = 'fas fa-times error-icon';
          const errorMessage = document.createElement('span');
          errorMessage.className = 'error-message';
          errorMessage.textContent = data.error || 'Upload failed';
          fileItem.appendChild(errorIcon);
          fileItem.appendChild(errorMessage);
        }
      }
    } catch (error) {
      results.failed++;
      console.error('Error uploading file:', error);
      
      const fileItem = fileList.querySelector(`li[data-filename="${file.name}"]`);
      if (fileItem) {
        fileItem.classList.add('error');
        const errorIcon = document.createElement('i');
        errorIcon.className = 'fas fa-times error-icon';
        const errorMessage = document.createElement('span');
        errorMessage.className = 'error-message';
        errorMessage.textContent = 'Network error or server issue';
        fileItem.appendChild(errorIcon);
        fileItem.appendChild(errorMessage);
      }
    }
  }
  
  // Show completion dialog
  if (results.success > 0) {
    successDialog.classList.add('show');
  }
  
  // Reset button state
  indexButton.disabled = false;
  indexButton.textContent = 'Index Files';
});

// Close success dialog
closeDialog.addEventListener('click', () => {
  successDialog.classList.remove('show');
  // Clear selected files
  selectedFiles = [];
  updateFileList();
});

// Drag and drop functionality
uploadBox.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
  uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadBox.classList.remove('dragover');
  
  const files = Array.from(e.dataTransfer.files);
  files.forEach(file => {
    if (!selectedFiles.some(f => f.name === file.name)) {
      selectedFiles.push(file);
    }
  });
  
  updateFileList();
});
  