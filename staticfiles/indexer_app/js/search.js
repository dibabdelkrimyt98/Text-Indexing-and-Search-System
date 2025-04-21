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
function highlightText(text, query, exactMatch = false) {
  if (!query) return text;
  
  if (exactMatch) {
    const regex = new RegExp(`(${query})`, 'g');
  return text.replace(regex, '<mark>$1</mark>');
  } else {
    const words = query.split(/\s+/);
    let highlightedText = text;
    words.forEach(word => {
      const regex = new RegExp(`(${word})`, 'gi');
      highlightedText = highlightedText.replace(regex, '<mark>$1</mark>');
    });
    return highlightedText;
  }
}

// ========== Submit Search Form ==========
document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.getElementById('searchForm');
    const searchInput = document.getElementById('searchInput');
    const searchResults = document.getElementById('searchResults');
    const loadingSpinner = document.querySelector('.typing-loader');
    const exactMatchCheckbox = document.getElementById('exactMatch');
    const fileTypeSelect = document.getElementById('fileType');
    const dateRangeSelect = document.getElementById('dateRange');
    const similaritySelect = document.getElementById('similaritySelect');

    // Debug check for elements
    console.log('Elements found:', {
        searchForm: !!searchForm,
        searchInput: !!searchInput,
        searchResults: !!searchResults,
        loadingSpinner: !!loadingSpinner,
        exactMatchCheckbox: !!exactMatchCheckbox,
        fileTypeSelect: !!fileTypeSelect,
        dateRangeSelect: !!dateRangeSelect,
        similaritySelect: !!similaritySelect
    });

    if (!searchForm || !searchResults) {
        console.error('Required elements not found');
        return;
    }

    // Handle search form submission
    searchForm.addEventListener('submit', async function(e) {
  e.preventDefault();
        console.log('Form submitted'); // Debug log

  const query = searchInput.value.trim();
  if (!query) {
            console.log('Empty query, stopping'); // Debug log
    return;
  }

        // Show loading state
        searchResults.innerHTML = '';
        if (loadingSpinner) {
            loadingSpinner.style.display = 'flex';
            console.log('Loading spinner shown'); // Debug log
        } else {
            console.warn('Loading spinner not found'); // Debug log
        }

        try {
            const formData = new URLSearchParams({
                query: query,
                method: similaritySelect ? similaritySelect.value : 'cosine',
                exact_match: exactMatchCheckbox ? exactMatchCheckbox.checked : false,
                file_type: fileTypeSelect ? fileTypeSelect.value : 'all',
                date_range: dateRangeSelect ? dateRangeSelect.value : 'all'
            });

            console.log('Sending request with data:', Object.fromEntries(formData)); // Debug log

            const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]');
            console.log('CSRF token found:', !!csrfToken); // Debug log

            const response = await fetch('/indexer_app/api/search/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'X-CSRFToken': csrfToken ? csrfToken.value : ''
                },
                body: formData
            });

            console.log('Response status:', response.status); // Debug log

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server error:', errorText); // Debug log
                throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
            }

            const data = await response.json();
            console.log('Response data:', data); // Debug log
            
            // Hide loading state
            if (loadingSpinner) {
                loadingSpinner.style.display = 'none';
            }
            
            // Clear previous results
            searchResults.innerHTML = '';

            if (data.error) {
                console.error('Server returned error:', data.error); // Debug log
                searchResults.innerHTML = `
                    <div class="alert alert-danger" role="alert">
                        ${data.error}
                    </div>`;
                return;
            }

            // Add summary
            const summary = document.createElement('div');
            summary.className = 'search-summary mb-4';
            summary.innerHTML = `
                <div class="alert alert-info" role="alert">
                    Found ${data.total} results for "${query}"
                    ${exactMatchCheckbox && exactMatchCheckbox.checked ? ' (Exact Match)' : ''}
                    ${fileTypeSelect && fileTypeSelect.value !== 'all' ? ` | Type: ${fileTypeSelect.value.toUpperCase()}` : ''}
                    ${dateRangeSelect && dateRangeSelect.value !== 'all' ? ` | Period: ${dateRangeSelect.options[dateRangeSelect.selectedIndex].text}` : ''}
                </div>`;
            searchResults.appendChild(summary);

            // Create results container
            if (!data.results || data.results.length === 0) {
                searchResults.innerHTML += `
                    <div class="alert alert-warning" role="alert">
                        No documents found matching your search criteria.
                    </div>`;
                return;
            }

            const resultsContainer = document.createElement('div');
            resultsContainer.className = 'results-container';

            // Add results
            data.results.forEach((result, index) => {
                const resultCard = document.createElement('div');
                resultCard.className = 'result-card';
                resultCard.innerHTML = `
                    <div class="card mb-3">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-start">
                                <h5 class="card-title">${highlightText(result.title, query, exactMatchCheckbox && exactMatchCheckbox.checked)}</h5>
                                <span class="badge bg-primary">${result.score}%</span>
                            </div>
                            <div class="card-subtitle mb-2 text-muted">
                                <span class="badge bg-secondary">${result.file_type.toUpperCase()}</span>
                                <small class="ms-2">${result.uploaded_at}</small>
                            </div>
                            ${result.preview ? `
                                <p class="card-text">
                                    ${highlightText(result.preview, query, exactMatchCheckbox && exactMatchCheckbox.checked)}
                                </p>
                            ` : ''}
                            <a href="/media/documents/${result.title}" class="btn btn-sm btn-outline-primary" target="_blank">
                                View Document
                            </a>
                        </div>
                    </div>`;
                resultsContainer.appendChild(resultCard);
            });

            searchResults.appendChild(resultsContainer);

        } catch (error) {
            console.error('Search error:', error); // Debug log
            if (loadingSpinner) {
                loadingSpinner.style.display = 'none';
            }
            searchResults.innerHTML = `
                <div class="alert alert-danger" role="alert">
                    An error occurred while searching. Please try again. Error: ${error.message}
                </div>`;
        }
    });
});
