// ========== Typing Heading Animation ==========
const title = "Search Indexed Documents";
const headingEl = document.getElementById("searchHeading");

function animateHeading(text, element, delay = 20) {
  element.innerHTML = "";
  [...text].forEach((char, index) => {
    const span = document.createElement("span");
    span.textContent = char === " " ? "\u00A0" : char;
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
  animateHeading(title, headingEl, 30);
});

// ========== Live Input Typing Indicator ==========
const searchInput = document.getElementById('searchInput');
const loader = document.getElementById('typingLoader');

searchInput.addEventListener('input', () => {
  loader.style.opacity = searchInput.value.trim() ? 1 : 0;
});

// ========== Highlight Query Matches ==========
function highlightText(text, query) {
  const regex = new RegExp(`(${query})`, 'gi');
  return text.replace(regex, '<mark>$1</mark>');
}

// ========== Display Results with Animation ==========
function displayResults(results, query) {
  const resultsList = document.getElementById("results");
  resultsList.innerHTML = '';

  if (results.length === 0) {
    resultsList.innerHTML = "<li>No results found.</li>";
    return;
  }

  results.forEach((res, idx) => {
    const li = document.createElement("li");
    li.innerHTML = highlightText(res, query);
    li.style.opacity = 0;
    li.style.transform = "translateY(10px)";
    li.style.transition = "all 0.4s ease";
    li.classList.add("fade-in");

    resultsList.appendChild(li);

    setTimeout(() => {
      li.style.opacity = 1;
      li.style.transform = "translateY(0)";
    }, idx * 100);
  });
}

// ========== Submit Search Form ==========
document.getElementById("searchForm").addEventListener("submit", function (e) {
  e.preventDefault();

  const query = searchInput.value.trim();
  const method = document.getElementById("similaritySelect").value;
  const resultsList = document.getElementById("results");

  resultsList.innerHTML = "";

  if (!query) {
    resultsList.innerHTML = '<li class="fade-in">⚠️ Please enter a query.</li>';
    return;
  }

  // Show searching animation
  const loading = document.createElement("li");
  loading.innerHTML = '<span class="loader"></span> Searching...';
  resultsList.appendChild(loading);

  // Optional: change button while searching
  const searchBtn = document.querySelector('#searchForm button');
  const originalBtnHTML = searchBtn.innerHTML;
  searchBtn.disabled = true;
  searchBtn.innerHTML = `<span class="loading-dots"><span>.</span><span>.</span><span>.</span></span>`;

  setTimeout(() => {
    resultsList.innerHTML = "";

    const mockResults = [
      `Result for "${query}" using ${method}:`,
      `Matched in document_01.txt`,
      `Matched in document_03.txt`,
      `Matched in document_07.txt`
    ];

    displayResults(mockResults, query);

    // Reset button
    searchBtn.innerHTML = originalBtnHTML;
    searchBtn.disabled = false;
  }, 1000);
});
