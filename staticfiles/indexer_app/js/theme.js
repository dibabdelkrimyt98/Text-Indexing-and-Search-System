const toggleBtn = document.getElementById('themeToggle');

toggleBtn.addEventListener('click', () => {
  document.body.classList.toggle('dark-mode');
  toggleBtn.textContent = document.body.classList.contains('dark-mode') ? '☀️' : '🌙';
  
  // Log the current theme for debugging
  console.log('Theme changed to:', document.body.classList.contains('dark-mode') ? 'dark' : 'light');
});
