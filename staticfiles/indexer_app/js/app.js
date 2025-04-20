// ========== Typing Effect ==========

const text1 = "You are welcome in AOS System.";
const text2 = "Empowering Knowledge .. Preserving Truth.";

let index1 = 0;
let index2 = 0;
const speed = 55;

function typeWriterLine1() {
  const el1 = document.getElementById("typingText1");
  if (index1 < text1.length) {
    el1.innerHTML += text1.charAt(index1);
    index1++;
    setTimeout(typeWriterLine1, speed);
  } else {
    setTimeout(typeWriterLine2, 500); // wait before typing line 2
  }
}

function typeWriterLine2() {
  const el2 = document.getElementById("typingText2");
  if (index2 < text2.length) {
    el2.innerHTML += text2.charAt(index2);
    index2++;
    setTimeout(typeWriterLine2, speed);
  }
}
window.addEventListener("DOMContentLoaded", typeWriterLine1);

// ========== Theme Toggle ==========

document.getElementById('themeToggle').addEventListener('click', () => {
  const body = document.body;
  const isDark = body.classList.toggle('dark-mode');
  body.classList.toggle('light-mode', !isDark);
});

// ========== Sidebar Toggle ==========

document.getElementById('toggleSidebar').addEventListener('click', () => {
  document.getElementById('sidebar').classList.toggle('collapsed');
});

// ========== Language Switcher ==========

document.querySelectorAll('.lang-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    alert(`Language switched to ${btn.dataset.lang} (demo)`);
    // TODO: Integrate actual language switching
  });
});

// ========== Floating Search Toggle ==========

const toggleSearchBtn = document.getElementById('toggleSearch');
const floatingSearch = document.getElementById('floatingSearch');

if (toggleSearchBtn && floatingSearch) {
  toggleSearchBtn.addEventListener('click', (e) => {
    e.preventDefault();
    floatingSearch.classList.toggle('active');
  });
}
