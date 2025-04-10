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
  
  // Mock indexing button
  document.getElementById('indexButton').addEventListener('click', () => {
    alert('Indexing files... (mock logic)');
  });
  