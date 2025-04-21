document.addEventListener('DOMContentLoaded', function() {
    const langButtons = document.querySelectorAll('.lang-btn');
    
    langButtons.forEach(button => {
        button.addEventListener('click', async function() {
            const lang = this.getAttribute('data-lang');
            console.log('Language selected:', lang);
            
            try {
                // Update active state
                langButtons.forEach(btn => btn.classList.remove('active'));
                this.classList.add('active');
                
                // Update document direction
                document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr';
                
                // Try to load translations if they exist
                try {
                    const res = await fetch(`/static/indexer_app/lang/${lang}.json`);
                    if (res.ok) {
                        const translations = await res.json();
                        document.querySelectorAll('[data-i18n]').forEach(el => {
                            const key = el.getAttribute('data-i18n');
                            if (translations[key]) {
                                el.innerHTML = translations[key];
                            }
                        });
                    }
                } catch (error) {
                    console.log('No translations found for:', lang);
                }
                
            } catch (error) {
                console.error('Error changing language:', error);
            }
        });
    });
});
