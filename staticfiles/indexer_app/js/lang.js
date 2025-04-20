const languageSelect = document.getElementById('languageSelect');

languageSelect.addEventListener('change', async (e) => {
  const lang = e.target.value;
  const res = await fetch(`lang/${lang}.json`);
  const translations = await res.json();

  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (translations[key]) {
      el.innerHTML = translations[key];
    }
  });

  document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr';
});
