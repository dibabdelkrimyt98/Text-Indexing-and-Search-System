window.addEventListener("DOMContentLoaded", () => {
  const resultsBody = document.getElementById("results-body");

  // Simulated data – replace this with real fetch from backend if needed
  const documents = [
    { name: "document_01.txt", status: "Indexed Successfully" },
    { name: "document_02.txt", status: "Indexed Successfully" },
    { name: "document_03.txt", status: "Indexed Successfully" }
  ];

  documents.forEach(doc => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${doc.name}</td><td>${doc.status}</td>`;
    resultsBody.appendChild(row);
  });
});
