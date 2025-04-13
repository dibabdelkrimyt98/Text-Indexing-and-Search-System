// ========== Sidebar Toggle ==========
document.getElementById('toggleSidebar')?.addEventListener('click', () => {
    document.getElementById('sidebar').classList.toggle('collapsed');
  });
  
  // ========== Theme Toggle ==========
  document.getElementById('themeToggle')?.addEventListener('click', () => {
    const body = document.body;
    body.classList.toggle('dark-mode');
    body.classList.toggle('light-mode');
  });
  
  // ========== Mock TF-IDF Data ==========
  const tfidfData = [
    { term: 'freedom', values: [0.234, 0.101, 0.034] },
    { term: 'resistance', values: [0.123, 0.210, 0.200] },
    { term: 'truth', values: [0.054, 0.034, 0.301] },
    { term: 'occupation', values: [0.321, 0.155, 0.078] },
  ];
  
  // Example dynamic document headers
  const documentCount = tfidfData[0].values.length;
  const docNames = Array.from({ length: documentCount }, (_, i) => `Document ${i + 1}`);
  
  // Inject table headers
  const thead = document.querySelector("#tfidfTable thead");
  const headRow = document.createElement("tr");
  headRow.innerHTML = `<th>Term</th>` + docNames.map(doc => `<th>${doc}</th>`).join('');
  thead.innerHTML = '';
  thead.appendChild(headRow);
  
  // Inject table rows
  const tableBody = document.querySelector("#tfidfTable tbody");
  tableBody.innerHTML = '';
  
  tfidfData.forEach(row => {
    const tr = document.createElement("tr");
    const valueCells = row.values.map(val => `<td>${val.toFixed(3)}</td>`).join('');
    tr.innerHTML = `<td><strong>${row.term}</strong></td>${valueCells}`;
    tableBody.appendChild(tr);
  });
  