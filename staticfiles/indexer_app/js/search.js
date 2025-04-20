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
  if (headingEl) {
    animateHeading(title, headingEl, 30);
  }
});

// ========== Live Input Typing Indicator ==========
const searchInput = document.getElementById('searchInput');
const loader = document.getElementById('typingLoader');

searchInput.addEventListener('input', () => {
  if (loader) {
    loader.style.opacity = searchInput.value.trim() ? 1 : 0;
  }
});

// ========== Highlight Query Matches ==========
function highlightText(text, query) {
  const regex = new RegExp(`(${query})`, 'gi');
  return text.replace(regex, '<mark>$1</mark>');
}

// ========== Submit Search Form ==========
document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.getElementById('searchForm');
    const searchInput = document.getElementById('searchInput');
    const searchResults = document.getElementById('searchResults');
    const typingLoader = document.querySelector('.typing-loader');

    if (!searchForm || !searchResults) {
        console.error('Required elements not found');
        return;
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
                },
                body: new URLSearchParams({
                    query: query,
                    method: document.getElementById('similaritySelect').value
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Hide loading state
            typingLoader.style.display = 'none';
            
            // Clear previous results
            searchResults.innerHTML = '';

            if (data.error) {
                searchResults.innerHTML = `<div class="error-message">${data.error}</div>`;
                return;
            }

            // Create results container
            const resultsTable = document.createElement('table');
            resultsTable.className = 'results-table';
            
            // Add table header
            resultsTable.innerHTML = `
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Document Title</th>
                        <th>Type</th>
                        <th>Relevance Score</th>
                    </tr>
                </thead>
                <tbody></tbody>
            `;

            // Add results to table
            const tbody = resultsTable.querySelector('tbody');
            data.results.forEach((result, index) => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${index + 1}</td>
                    <td>${result.title}</td>
                    <td>${result.file_type.toUpperCase()}</td>
                    <td>${result.score}%</td>
                `;
                tbody.appendChild(row);
            });

            // Add summary
            const summary = document.createElement('div');
            summary.className = 'search-summary';
            summary.textContent = `Found ${data.total} results for "${query}" using ${document.getElementById('similaritySelect').options[document.getElementById('similaritySelect').selectedIndex].text}`;
            
            // Add results to page
            searchResults.appendChild(summary);
            searchResults.appendChild(resultsTable);

        } catch (error) {
            console.error('Search error:', error);
            typingLoader.style.display = 'none';
            searchResults.innerHTML = '<div class="error-message">An error occurred while searching</div>';
        }
    });
});
