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
  document.getElementById('indexButton').addEventListener('click', () => {
    const dialog = document.getElementById('successDialog');
    dialog.classList.add('show');
  });
  document.getElementById('closeDialog').addEventListener('click', () => {
    const dialog = document.getElementById('successDialog');
    dialog.classList.remove('show');
  });
  