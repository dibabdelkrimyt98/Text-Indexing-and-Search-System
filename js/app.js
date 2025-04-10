// Typing effect
const text = "Empowering Knowledge. Preserving Truth.";
let index = 0;
const speed = 60;

function typeWriter() {
  if (index < text.length) {
    document.getElementById("typingText").innerHTML += text.charAt(index);
    index++;
    setTimeout(typeWriter, speed);
  }
}
window.addEventListener("DOMContentLoaded", typeWriter);

// Theme toggle
document.getElementById('themeToggle').addEventListener('click', () => {
  const body = document.body;
  const isDark = body.classList.toggle('dark-mode');
  body.classList.toggle('light-mode', !isDark);
});

// Sidebar toggle
document.getElementById('toggleSidebar').addEventListener('click', () => {
  document.getElementById('sidebar').classList.toggle('collapsed');
});

// Language switcher (mock)
document.querySelectorAll('.lang-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    alert(`Language switched to ${btn.dataset.lang} (demo)`);
    // Here you can load translation files or update UI dynamically
  });
});

const toggleSearchBtn = document.getElementById('toggleSearch');
const floatingSearch = document.getElementById('floatingSearch');

toggleSearchBtn.addEventListener('click', (e) => {
  e.preventDefault();
  floatingSearch.classList.toggle('active');
});
