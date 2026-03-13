// Custom JavaScript for MER-Factory documentation
document.addEventListener('DOMContentLoaded', function () {

    // --- NEW: Mobile Navigation Toggle ---
    const menuToggle = document.querySelector('.menu-toggle');
    const sidebar = document.querySelector('header');
    const mainContent = document.querySelector('section.main-content');

    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', function (event) {
            sidebar.classList.toggle('sidebar-visible');
            event.stopPropagation(); // Prevent click from bubbling to main content
        });

        if (mainContent) {
            mainContent.addEventListener('click', function () {
                if (sidebar.classList.contains('sidebar-visible')) {
                    sidebar.classList.remove('sidebar-visible');
                }
            });
        }

        sidebar.addEventListener('click', function (event) {
            if (event.target.tagName === 'A' && sidebar.classList.contains('sidebar-visible')) {
                sidebar.classList.remove('sidebar-visible');
            }
        });
    }
    // --- END: Mobile Navigation Toggle ---


    // Scroll to Top Button
    const scrollToTopBtn = document.createElement('button');
    scrollToTopBtn.className = 'scroll-to-top';
    scrollToTopBtn.innerHTML = '<i class="fas fa-chevron-up"></i>';
    scrollToTopBtn.setAttribute('aria-label', 'Scroll to top');
    document.body.appendChild(scrollToTopBtn);

    // Show/hide scroll to top button
    window.addEventListener('scroll', function () {
        if (window.pageYOffset > 300) {
            scrollToTopBtn.classList.add('visible');
        } else {
            scrollToTopBtn.classList.remove('visible');
        }
    });

    // Scroll to top functionality
    scrollToTopBtn.addEventListener('click', function () {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add copy button to code blocks
    document.querySelectorAll('pre code').forEach((codeBlock) => {
        const pre = codeBlock.parentNode;
        const button = document.createElement('button');
        button.className = 'copy-code-btn';
        button.innerHTML = '<i class="fas fa-copy"></i>';
        button.setAttribute('aria-label', 'Copy code');

        // Style the button
        button.style.cssText = `
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            background: rgba(0,0,0,0.1);
            border: none;
            border-radius: 4px;
            padding: 0.5rem;
            cursor: pointer;
            opacity: 0;
            transition: opacity 0.3s ease;
            color: #666;
        `;

        // Make pre relative for absolute positioning
        pre.style.position = 'relative';

        // Show button on hover
        pre.addEventListener('mouseenter', () => {
            button.style.opacity = '1';
        });

        pre.addEventListener('mouseleave', () => {
            button.style.opacity = '0';
        });

        // Copy functionality
        button.addEventListener('click', async () => {
            const text = codeBlock.textContent;
            try {
                await navigator.clipboard.writeText(text);
                button.innerHTML = '<i class="fas fa-check"></i>';
                button.style.color = '#27ae60';
                setTimeout(() => {
                    button.innerHTML = '<i class="fas fa-copy"></i>';
                    button.style.color = '#666';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy text: ', err);
            }
        });

        pre.appendChild(button);
    });

    // Add active state to navigation links
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.style.backgroundColor = 'var(--light-gray)';
            link.style.color = 'var(--secondary-color)';
        }
    });

    // Print styles
    const printStyle = document.createElement('style');
    printStyle.textContent = `
        @media print {
            .header-content, .site-footer, .scroll-to-top, .copy-code-btn, .menu-toggle {
                display: none !important;
            }
            
            .main-content {
                margin: 0;
                padding: 0;
                box-shadow: none;
                border-radius: 0;
            }
            
            .wrapper {
                display: block;
            }
            
            a {
                color: black !important;
                text-decoration: none !important;
            }
            
            a[href]:after {
                content: " (" attr(href) ")";
                font-size: 0.8rem;
                color: #666;
            }
            
            pre, code {
                background: #f5f5f5 !important;
                border: 1px solid #ddd !important;
            }
        }
    `;
    document.head.appendChild(printStyle);


    // Add loading states for external links
    document.querySelectorAll('a[href^="http"]').forEach(link => {
        link.addEventListener('click', function () {
            const icon = this.querySelector('i');
            if (icon) {
                const originalClass = icon.className;
                icon.className = 'fas fa-spinner fa-spin';
                setTimeout(() => {
                    icon.className = originalClass;
                }, 1000);
            }
        });
    });
});
