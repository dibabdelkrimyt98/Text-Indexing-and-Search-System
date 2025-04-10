document.getElementById('searchForm').addEventListener('submit', function (e) {
    e.preventDefault();
  
    const query = document.getElementById('searchInput').value.trim();
    const method = document.getElementById('similaritySelect').value;
    const resultsList = document.getElementById('results');
  
    resultsList.innerHTML = '';
  
    if (!query) {
      resultsList.innerHTML = '<li class="fade-in">⚠️ Please enter a query.</li>';
      return;
    }
  
    // Show loading spinner
    const loading = document.createElement('li');
    loading.innerHTML = '<span class="loader"></span> Searching...';
    resultsList.appendChild(loading);
  
    // Simulate delay
    setTimeout(() => {
      resultsList.innerHTML = ''; // Clear loader
  
      const mockResults = [
        `Result for "<b>${query}</b>" using <i>${method}</i>:`,
        `Matched in <strong>document_01.txt</strong>`,
        `Matched in <strong>document_03.txt</strong>`,
        `Matched in <strong>document_07.txt</strong>`
      ];
  
      mockResults.forEach((res, i) => {
        const li = document.createElement('li');
        li.innerHTML = res;
        li.classList.add('fade-in');
        li.style.animationDelay = `${i * 100}ms`; // stagger effect
        resultsList.appendChild(li);
      });
    }, 1000); // simulate 1s delay
  });
  const title = "Search Indexed Documents";
const headingEl = document.getElementById("searchHeading");

function animateHeading(text, element, delay = 20) {
  element.innerHTML = ""; // Clear first

  [...text].forEach((char, index) => {
    const span = document.createElement("span");
    span.textContent = char === " " ? "\u00A0" : char; // Add real space
    span.style.opacity = "0";
    span.style.display = "inline-block";
    span.style.transform = "translateY(-10px)";
    span.style.transition = "all 0.3s ease";

    setTimeout(() => {
      element.appendChild(span);
      setTimeout(() => {
        span.style.opacity = "1";
        span.style.transform = "translateY(0)";
      }, 10);
    }, index * delay);
  });
}

window.addEventListener("DOMContentLoaded", () => {
  animateHeading(title, headingEl, 30); // Faster animation
});
