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
document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.getElementById('searchForm');
    const searchInput = document.getElementById('searchInput');
    const similaritySelect = document.getElementById('similaritySelect');
    const fileTypeSelect = document.getElementById('fileType');
    const dateRangeSelect = document.getElementById('dateRange');
    const exactMatchCheckbox = document.getElementById('exactMatch');
    const searchResults = document.getElementById('searchResults');
    const typingLoader = document.querySelector('.typing-loader');
    
    // Get CSRF token
    function getCSRFToken() {
        const cookies = document.cookie.split(';');
        for (let cookie of cookies) {
            const [name, value] = cookie.trim().split('=');
            if (name === 'csrftoken') {
                return value;
            }
        }
        return '';
    }

    // Format date
    function formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    // Create result item HTML
    function createResultItem(result) {
        const li = document.createElement('li');
        li.className = 'search-result-item';
        
        li.innerHTML = `
            <div class="result-header">
                <h3 class="result-title">${result.title}</h3>
                <span class="result-score">${result.score}% match</span>
            </div>
            <div class="result-meta">
                <span class="result-type"><i class="fas fa-file"></i> ${result.file_type}</span>
                <span class="result-date"><i class="fas fa-calendar"></i> ${result.uploaded_at}</span>
                <span class="result-size"><i class="fas fa-weight"></i> ${result.size}</span>
            </div>
            ${result.preview ? `<p class="result-preview">${result.preview}</p>` : ''}
        `;
        
        return li;
    }

    // Handle search form submission
    searchForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const query = searchInput.value.trim();
        if (!query) return;

        // Show loading state
        searchResults.innerHTML = '';
        typingLoader.style.display = 'flex';

        try {
            const response = await fetch('/indexer_app/api/search/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'X-CSRFToken': getCSRFToken()
                },
                body: new URLSearchParams({
                    query: query,
                    method: similaritySelect.value,
                    fileType: fileTypeSelect.value,
                    dateRange: dateRangeSelect.value,
                    exactMatch: exactMatchCheckbox.checked
                })
            });

            const data = await response.json();
            
            // Hide loading state
            typingLoader.style.display = 'none';

            // Clear previous results
            searchResults.innerHTML = '';

            if (data.error) {
                searchResults.innerHTML = `<div class="error-message">${data.error}</div>`;
                return;
            }

            // Add search summary
            const summary = document.createElement('div');
            summary.className = 'search-summary';
            summary.innerHTML = `Found ${data.total} results for "${data.query}"`;
            searchResults.appendChild(summary);

            if (data.results.length === 0) {
                searchResults.innerHTML += '<div class="no-results">No matching documents found</div>';
                return;
            }

            // Create results list
            const resultsList = document.createElement('ul');
            resultsList.className = 'results-list';

            // Add each result
            data.results.forEach((result, index) => {
                const resultItem = createResultItem(result);
                resultItem.style.animationDelay = `${index * 0.1}s`;
                resultsList.appendChild(resultItem);
            });

            searchResults.appendChild(resultsList);

        } catch (error) {
            console.error('Search error:', error);
            typingLoader.style.display = 'none';
            searchResults.innerHTML = '<div class="error-message">An error occurred while searching</div>';
        }
    });
});
