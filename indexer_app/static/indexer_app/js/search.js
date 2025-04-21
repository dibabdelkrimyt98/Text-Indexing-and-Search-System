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
console.log('Search.js file loaded successfully');

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded');
    
    // Get DOM elements
    const searchForm = document.getElementById('searchForm');
    const searchInput = document.getElementById('searchInput');
    const searchResults = document.getElementById('searchResults');
    const loadingSpinner = document.querySelector('.typing-loader');
    const searchButton = document.querySelector('.search-btn');
    
    // Debug log for element existence
    console.log('Search Elements:', {
        searchForm: !!searchForm,
        searchInput: !!searchInput,
        searchResults: !!searchResults,
        loadingSpinner: !!loadingSpinner,
        searchButton: !!searchButton
    });

    if (!searchForm || !searchResults) {
        console.error('Required search elements not found');
        return;
    }

    // Modified button click handler
    if (searchButton) {
        searchButton.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('Search button clicked');
            
            // Trigger form submission
            const submitEvent = new Event('submit', {
                bubbles: true,
                cancelable: true
            });
            searchForm.dispatchEvent(submitEvent);
        });
    }

    // Handle search form submission
    searchForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        console.log('Form submit event triggered');

        // Disable the button to prevent double submission
        if (searchButton) {
            searchButton.disabled = true;
        }

        const query = searchInput.value.trim();
        if (!query) {
            console.log('Empty query, stopping');
            if (searchButton) {
                searchButton.disabled = false;
            }
            return;
        }

        try {
            // Show loading state
            searchResults.innerHTML = '';
            document.body.classList.add('loading');
            if (loadingSpinner) {
                loadingSpinner.style.display = 'block';
            }

            // Get form data
            const formData = new FormData(searchForm);
            console.log('Search query:', query);
            console.log('Form data:', Object.fromEntries(formData));

            // Get CSRF token
            const csrftoken = document.querySelector('[name=csrfmiddlewaretoken]').value;
            if (!csrftoken) {
                throw new Error('CSRF token not found');
            }

            // Make search request
            const response = await fetch('/api/search/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrftoken,
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: formData
            });

            console.log('Search response:', {
                status: response.status,
                ok: response.ok,
                statusText: response.statusText
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server error: ${response.status} - ${errorText}`);
            }

            const data = await response.json();
            console.log('Search results:', data);

            // Hide loading state
            document.body.classList.remove('loading');
            if (loadingSpinner) {
                loadingSpinner.style.display = 'none';
            }

            // Display results
            if (data.results && data.results.length > 0) {
                searchResults.innerHTML = data.results.map(doc => `
                    <div class="search-result">
                        <div class="result-header">
                            <a href="/indexer_app/download/${doc.id}/" class="result-title" download>
                                ${doc.title}
                            </a>
                            <span class="result-score">${(doc.score * 100).toFixed(1)}% match</span>
                        </div>
                        <div class="result-meta">
                            <span class="result-type">${doc.file_type}</span>
                            <span class="result-size">${formatFileSize(doc.size)}</span>
                        </div>
                        ${doc.context ? `<div class="result-context">${doc.context}</div>` : ''}
                    </div>
                `).join('');
                
                searchResults.style.display = 'block';
            } else {
                searchResults.innerHTML = '<div class="no-results">No documents found matching your search.</div>';
            }

        } catch (error) {
            console.error('Search error:', error);
            searchResults.innerHTML = `
                <div class="alert alert-danger">
                    <p>Error performing search: ${error.message}</p>
                    <p>Please try again or contact support if the problem persists.</p>
                </div>`;
        } finally {
            // Hide loading state
            document.body.classList.remove('loading');
            if (loadingSpinner) {
                loadingSpinner.style.display = 'none';
            }
            // Re-enable the button
            if (searchButton) {
                searchButton.disabled = false;
            }
        }
    });
});
