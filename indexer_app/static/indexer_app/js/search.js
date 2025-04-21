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

    if (!searchForm || !searchResults) {
        console.error('Required elements not found');
        return;
    }

    // Handle search form submission
    searchForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        console.log('Form submitted');

        const query = searchInput.value.trim();
        if (!query) {
            console.log('Empty query, stopping');
            return;
        }

        // Show loading state
        searchResults.innerHTML = '';
        if (loadingSpinner) {
            loadingSpinner.style.display = 'block';
        }

        try {
            const formData = new FormData(searchForm);
            console.log('Form data:', Object.fromEntries(formData));

            const response = await fetch('/indexer_app/api/search/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': formData.get('csrfmiddlewaretoken')
                },
                body: formData
            });

            console.log('Response status:', response.status);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server error:', errorText);
                throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
            }

            const data = await response.json();
            console.log('Response data:', data);
            
            // Hide loading state
            if (loadingSpinner) {
                loadingSpinner.style.display = 'none';
            }
            
            // Clear previous results
            searchResults.innerHTML = '';

            if (data.error) {
                console.error('Server returned error:', data.error);
                searchResults.innerHTML = `
                    <div class="alert alert-danger">
                        ${data.error}
                    </div>`;
                return;
            }

            // Add summary
            const summary = document.createElement('div');
            summary.className = 'results-summary';
            summary.innerHTML = `
                <h3>Found ${data.total} results for "${query}"</h3>
                <div class="filters-summary">
                    ${exactMatchCheckbox.checked ? '<span>Exact Match</span>' : ''}
                    ${fileTypeSelect.value !== 'all' ? `<span>Type: ${fileTypeSelect.value.toUpperCase()}</span>` : ''}
                    ${dateRangeSelect.value !== 'all' ? `<span>Period: ${dateRangeSelect.options[dateRangeSelect.selectedIndex].text}</span>` : ''}
                </div>`;
            searchResults.appendChild(summary);

            if (!data.results || data.results.length === 0) {
                searchResults.innerHTML += `
                    <div class="no-results">
                        No documents found matching your search criteria.
                    </div>`;
                return;
            }

            // Add results
            data.results.forEach(result => {
                const resultCard = document.createElement('div');
                resultCard.className = 'result-card';
                resultCard.innerHTML = `
                    <div class="result-content">
                        <div class="result-header">
                            <h3 class="result-title">
                                <a href="/media/documents/${result.title}" target="_blank">
                                    ${highlightText(result.title, query, exactMatchCheckbox.checked)}
                                </a>
                            </h3>
                            <span class="score-badge">${result.score}%</span>
                        </div>
                        <div class="result-meta">
                            <span class="file-type">
                                <i class="fas fa-file"></i> ${result.file_type.toUpperCase()}
                            </span>
                            <span class="upload-date">
                                <i class="fas fa-calendar"></i> ${result.uploaded_at}
                            </span>
                            <span class="file-size">
                                <i class="fas fa-weight"></i> ${result.size}
                            </span>
                        </div>
                        ${result.preview ? `
                            <p class="preview-text">
                                ${highlightText(result.preview, query, exactMatchCheckbox.checked)}
                            </p>
                        ` : ''}
                    </div>`;
                searchResults.appendChild(resultCard);
            });

        } catch (error) {
            console.error('Search error:', error);
            if (loadingSpinner) {
                loadingSpinner.style.display = 'none';
            }
            searchResults.innerHTML = `
                <div class="alert alert-danger">
                    An error occurred while searching. Please try again. Error: ${error.message}
                </div>`;
        }
    });
});
