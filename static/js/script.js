/* ==========================================================================
   OptiCrop – Main JavaScript
   Features: Dark mode, scroll animations, typing effect, back-to-top,
             loading overlay, navbar scroll, prediction button animation
   ========================================================================== */
(function () {
    "use strict";

    /* ----- Dark Mode Toggle ----- */
    const themeToggle = document.getElementById("themeToggle");
    const html = document.documentElement;

    // Load saved theme
    const savedTheme = localStorage.getItem("opticrop-theme") || "light";
    html.setAttribute("data-theme", savedTheme);
    updateThemeIcon(savedTheme);

    if (themeToggle) {
        themeToggle.addEventListener("click", function () {
            const current = html.getAttribute("data-theme");
            const next = current === "dark" ? "light" : "dark";
            html.setAttribute("data-theme", next);
            localStorage.setItem("opticrop-theme", next);
            updateThemeIcon(next);
        });
    }

    function updateThemeIcon(theme) {
        if (!themeToggle) return;
        const icon = themeToggle.querySelector("i");
        if (icon) {
            icon.className = theme === "dark" ? "fas fa-sun" : "fas fa-moon";
        }
    }

    /* ----- Navbar Scroll Effect ----- */
    const navbar = document.querySelector(".navbar-opticrop");
    window.addEventListener("scroll", function () {
        if (navbar) {
            navbar.classList.toggle("scrolled", window.scrollY > 50);
        }
    });

    /* ----- Scroll Animations (Intersection Observer) ----- */
    const animatedEls = document.querySelectorAll("[data-animate]");
    if ("IntersectionObserver" in window) {
        const observer = new IntersectionObserver(
            function (entries) {
                entries.forEach(function (entry) {
                    if (entry.isIntersecting) {
                        const delay = entry.target.dataset.delay || 0;
                        setTimeout(function () {
                            entry.target.classList.add("in-view");
                        }, delay);
                        observer.unobserve(entry.target);
                    }
                });
            },
            { threshold: 0.1 }
        );
        animatedEls.forEach(function (el) { observer.observe(el); });
    } else {
        // Fallback: show all
        animatedEls.forEach(function (el) { el.classList.add("in-view"); });
    }

    /* ----- Back to Top Button ----- */
    const backToTop = document.getElementById("backToTop");
    if (backToTop) {
        window.addEventListener("scroll", function () {
            backToTop.classList.toggle("visible", window.scrollY > 400);
        });
        backToTop.addEventListener("click", function () {
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    }

    /* ----- Loading Overlay + Prediction Button ----- */
    const predictForm = document.getElementById("predictionForm");
    const suitabilityForm = document.getElementById("suitabilityForm");
    const loadingOverlay = document.getElementById("loadingOverlay");

    function showLoading() {
        if (loadingOverlay) loadingOverlay.classList.add("active");
    }
    function hideLoading() {
        if (loadingOverlay) loadingOverlay.classList.remove("active");
    }

    if (predictForm) {
        predictForm.addEventListener("submit", function () {
            const btn = document.getElementById("predictBtn");
            if (btn) {
                btn.querySelector(".btn-predict-text").classList.add("d-none");
                btn.querySelector(".btn-predict-loader").classList.remove("d-none");
            }
            showLoading();
        });
    }

    if (suitabilityForm) {
        suitabilityForm.addEventListener("submit", function () {
            showLoading();
        });
    }

    // Hide loading overlay when page is fully loaded (in case of back navigation)
    window.addEventListener("load", hideLoading);

    /* ----- Typing Effect on Hero Title ----- */
    const heroTitle = document.querySelector(".hero-title");
    if (heroTitle && !sessionStorage.getItem("opticrop-typed")) {
        sessionStorage.setItem("opticrop-typed", "1");
        const fullText = heroTitle.innerHTML;
        heroTitle.innerHTML = "";
        heroTitle.classList.add("typing-cursor");
        let i = 0;
        const tempDiv = document.createElement("div");
        tempDiv.innerHTML = fullText;
        const textNodes = [];
        function collectText(node) {
            node.childNodes.forEach(function (child) {
                if (child.nodeType === 3) {
                    textNodes.push(child);
                } else if (child.nodeType === 1) {
                    collectText(child);
                }
            });
        }
        // Simple: type the plain text version
        const plainText = tempDiv.textContent;
        heroTitle.textContent = "";

        function typeChar() {
            if (i < plainText.length) {
                heroTitle.textContent = plainText.slice(0, i + 1);
                i++;
                setTimeout(typeChar, 35);
            } else {
                heroTitle.classList.remove("typing-cursor");
                heroTitle.innerHTML = fullText;
            }
        }
        // Delay start so animation kicks in after page paint
        setTimeout(typeChar, 300);
    }

    /* ----- Active Nav Link Highlight ----- */
    const navLinks = document.querySelectorAll(".navbar-opticrop .nav-link");
    const currentPage = window.location.pathname;
    navLinks.forEach(function (link) {
        const href = link.getAttribute("href");
        if (href && href !== "#" && currentPage === href) {
            link.classList.add("active");
        }
    });

    /* ----- Smooth Anchor Scroll (for #predict link) ----- */
    document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
        anchor.addEventListener("click", function (e) {
            const targetId = this.getAttribute("href");
            if (targetId === "#" || targetId.length < 2) return;
            const target = document.querySelector(targetId);
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: "smooth" });
            }
        });
    });
})();
