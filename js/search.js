document.getElementById('searchForm').addEventListener('submit', function (e) {
    e.preventDefault();
    const query = document.getElementById('searchInput').value.trim();
    const method = document.getElementById('similaritySelect').value;
    const resultsList = document.getElementById('results');
  
    resultsList.innerHTML = '';
  
    if (!query) {
      resultsList.innerHTML = '<li>Please enter a query.</li>';
      return;
    }
  
    // Mock search results
    const results = [
      `Result for "${query}" using ${method}...`,
      'Matched in document_01.txt',
      'Matched in document_03.txt',
      'Matched in document_07.txt'
    ];
  
    results.forEach(res => {
      const li = document.createElement('li');
      li.textContent = res;
      resultsList.appendChild(li);
    });
  });
  