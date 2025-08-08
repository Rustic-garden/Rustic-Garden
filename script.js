/* ================================
   Vivaldi Spring - script.js
   Smooth nav, accessible carousel, reveals, form handling
   ================================ */

document.addEventListener('DOMContentLoaded', () => {
  /* ---------- NAV (supports <a href="#id"> or .nav-link[data-target="#id"] ) ---------- */
  const header = document.querySelector('.site-header');
  const navLinks = Array.from(document.querySelectorAll('a[href^="#"], .nav-link[data-target]'));
  navLinks.forEach(el => {
    el.addEventListener('click', (ev) => {
      ev.preventDefault();
      const targetId = el.getAttribute('href') || el.dataset.target;
      if (!targetId) return;
      const target = document.querySelector(targetId);
      if (!target) return;
      // close mobile menu if open
      const mobile = document.querySelector('.nav-mobile.open');
      if (mobile) mobile.classList.remove('open');
      // smooth scroll with offset for fixed header
      const headerOffset = header ? header.offsetHeight + 8 : 0;
      const targetTop = target.getBoundingClientRect().top + window.pageYOffset - headerOffset;
      window.scrollTo({ top: targetTop, behavior: 'smooth' });
      // update focus for accessibility
      target.setAttribute('tabindex', '-1');
      target.focus({ preventScroll: true });
      target.removeAttribute('tabindex');
    });
  });

  // mobile nav toggle
  const navToggle = document.querySelector('.nav-toggle');
  if (navToggle) {
    const navMobile = document.querySelector('.nav-mobile');
    navToggle.addEventListener('click', () => {
      if (!navMobile) return;
      navMobile.classList.toggle('open');
      navToggle.setAttribute('aria-expanded', navMobile.classList.contains('open'));
    });
  }

  /* ---------- REVEAL ON SCROLL (IntersectionObserver) ---------- */
  const reveals = document.querySelectorAll('.reveal, section');
  const io = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('in-view');
        // optionally unobserve to reduce work
        io.unobserve(entry.target);
      }
    });
  }, {
    root: null,
    rootMargin: '0px 0px -8% 0px',
    threshold: 0.12
  });
  reveals.forEach(el => io.observe(el));

  /* ---------- CAROUSEL - accessible, autoplay, swipe, keyboard ---------- */
  const carouselRoot = document.querySelector('.carousel-container');
  if (carouselRoot) {
    const track = carouselRoot.querySelector('.carousel-track');
    const items = Array.from(track.children);
    const prevBtn = carouselRoot.querySelector('.carousel-btn.left');
    const nextBtn = carouselRoot.querySelector('.carousel-btn.right');
    const progressBar = carouselRoot.querySelector('.carousel-progress > i');
    let index = 0;
    let autoplay = true;
    const autoplayInterval = 5000;
    let autoplayTimer = null;
    let progressStart = null;

    // ensure items have role and aria
    items.forEach((it, i) => {
      it.setAttribute('role', 'group');
      it.setAttribute('aria-roledescription', 'carousel item');
      it.setAttribute('aria-label', `${i + 1} of ${items.length}`);
    });

    // responsive item width calculation
    function itemWidth() {
      return items[0].offsetWidth + parseFloat(getComputedStyle(track).gap || 0);
    }

    function setPosition() {
      const w = itemWidth();
      track.style.transform = `translateX(-${index * w}px)`;
    }

    // normalize index within bounds
    function normalizeIdx(i) {
      const n = items.length;
      if (i < 0) return n - 1;
      if (i >= n) return 0;
      return i;
    }

    function goTo(i, instant = false) {
      index = normalizeIdx(i);
      if (instant) {
        track.style.transition = 'none';
      } else {
        track.style.transition = '';
      }
      setPosition();
      // restart progress animation
      startProgress();
    }

    // prev / next handlers
    nextBtn && nextBtn.addEventListener('click', () => { goTo(index + 1); pauseAutoplayTemporarily(); });
    prevBtn && prevBtn.addEventListener('click', () => { goTo(index - 1); pauseAutoplayTemporarily(); });

    // keyboard support
    carouselRoot.addEventListener('keydown', (ev) => {
      if (ev.key === 'ArrowLeft') { ev.preventDefault(); goTo(index - 1); pauseAutoplayTemporarily(); }
      if (ev.key === 'ArrowRight') { ev.preventDefault(); goTo(index + 1); pauseAutoplayTemporarily(); }
    });
    carouselRoot.setAttribute('tabindex', '0');

    // touch/swipe support
    let startX = 0;
    let dx = 0;
    let isDown = false;
    track.addEventListener('touchstart', (e) => {
      isDown = true;
      startX = e.touches[0].clientX;
      track.style.transition = 'none';
      pauseAutoplayTemporarily();
    }, {passive: true});
    track.addEventListener('touchmove', (e) => {
      if (!isDown) return;
      dx = e.touches[0].clientX - startX;
      track.style.transform = `translateX(calc(${-index * itemWidth()}px + ${dx}px))`;
    }, {passive: true});
    track.addEventListener('touchend', (e) => {
      if (!isDown) return;
      isDown = false;
      if (Math.abs(dx) > 50) {
        if (dx > 0) { goTo(index - 1); } else { goTo(index + 1); }
      } else {
        goTo(index, false);
      }
      dx = 0;
    });

    // autoplay & progress
    function startAutoplay() {
      if (!autoplay) return;
      stopAutoplay();
      progressStart = performance.now();
      autoplayTimer = setInterval(() => { goTo(index + 1); }, autoplayInterval);
      startProgress();
    }
    function stopAutoplay() {
      if (autoplayTimer) { clearInterval(autoplayTimer); autoplayTimer = null; }
      if (progressBar) { progressBar.style.transition = 'none'; progressBar.style.width = '0%'; }
    }
    function pauseAutoplayTemporarily() {
      autoplay = false;
      stopAutoplay();
      // resume after a delay
      setTimeout(() => { autoplay = true; startAutoplay(); }, 7000);
    }
    function startProgress() {
      if (!progressBar) return;
      progressBar.style.transition = `width ${autoplayInterval}ms linear`;
      // reset then trigger
      progressBar.style.width = '0%';
      // force reflow
      void progressBar.offsetWidth;
      progressBar.style.width = '100%';
    }

    // pause on hover/focus
    carouselRoot.addEventListener('mouseenter', () => { stopAutoplay(); });
    carouselRoot.addEventListener('mouseleave', () => { if (autoplay) startAutoplay(); });
    carouselRoot.addEventListener('focusin', () => { stopAutoplay(); });
    carouselRoot.addEventListener('focusout', () => { if (autoplay) startAutoplay(); });

    // handle window resize
    let resizeTimeout = null;
    window.addEventListener('resize', () => {
      if (resizeTimeout) clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => {
        setPosition();
      }, 120);
    });

    // initialize
    setPosition();
    startAutoplay();
  }

  /* ---------- FORMS - basic client-side validation & UX ---------- */
  const forms = document.querySelectorAll('form');
  forms.forEach(form => {
    form.addEventListener('submit', (ev) => {
      ev.preventDefault();
      // basic required validation
      const required = Array.from(form.querySelectorAll('[required]'));
      let ok = true;
      required.forEach(field => {
        if (!field.value || field.value.trim() === '') {
          ok = false;
          field.focus();
          field.setAttribute('aria-invalid', 'true');
          field.classList.add('input-error');
        } else {
          field.removeAttribute('aria-invalid');
          field.classList.remove('input-error');
        }
      });
      if (!ok) {
        showFormMessage(form, 'Please fill the required fields.', 'error');
        return;
      }

      // create simple summary modal (no network)
      const data = new FormData(form);
      const obj = {};
      for (const [k, v] of data.entries()) obj[k] = v;
      showFormMessage(form, 'Thanks — your information has been recorded locally. We will contact you shortly.', 'success');

      // optional: clear form after brief delay
      setTimeout(() => {
        form.reset();
      }, 900);
    });
  });

  function showFormMessage(form, message, type = 'info') {
    let msg = form.querySelector('.form-status');
    if (!msg) {
      msg = document.createElement('div');
      msg.className = 'form-status';
      form.insertAdjacentElement('afterbegin', msg);
    }
    msg.textContent = message;
    msg.setAttribute('role', 'status');
    msg.classList.remove('info','error','success');
    msg.classList.add(type);
    setTimeout(() => { msg.classList.add('visible'); }, 20);
    setTimeout(() => { msg.classList.remove('visible'); }, 5500);
  }

  /* ---------- IMAGE LAZY LOADING (native if possible) ---------- */
  const imgs = document.querySelectorAll('img[data-src]');
  imgs.forEach(img => {
    if ('loading' in HTMLImageElement.prototype) {
      img.src = img.dataset.src;
      img.removeAttribute('data-src');
    } else {
      // fallback: IntersectionObserver
      const iob = new IntersectionObserver((entries, observer) => {
        entries.forEach(e => {
          if (e.isIntersecting) {
            e.target.src = e.target.dataset.src;
            e.target.removeAttribute('data-src');
            observer.unobserve(e.target);
          }
        });
      }, { rootMargin: '120px' });
      iob.observe(img);
    }
  });

  /* ---------- SMALL UX: back-to-top button ---------- */
  let topBtn = document.querySelector('.back-to-top');
  if (!topBtn) {
    topBtn = document.createElement('button');
    topBtn.className = 'back-to-top';
    topBtn.setAttribute('aria-label', 'Back to top');
    topBtn.innerHTML = '↑';
    document.body.appendChild(topBtn);
    Object.assign(topBtn.style, {
      position: 'fixed',
      right: '16px',
      bottom: '20px',
      width: '44px',
      height: '44px',
      borderRadius: '10px',
      border: 'none',
      background: '#6a994e',
      color: '#fff',
      cursor: 'pointer',
      zIndex: 1100,
      display: 'none',
      boxShadow: '0 8px 20px rgba(47,62,70,0.12)'
    });
  }
  window.addEventListener('scroll', () => {
    topBtn.style.display = (window.scrollY > 420) ? 'inline-flex' : 'none';
  });
  topBtn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  /* ---------- small enhancement: keyboard "skip to content" support ---------- */
  const skipLink = document.querySelector('.skip-to-content');
  if (skipLink) {
    skipLink.addEventListener('click', (e) => {
      e.preventDefault();
      const id = skipLink.getAttribute('href');
      const target = document.querySelector(id);
      if (!target) return;
      target.setAttribute('tabindex','-1');
      target.focus();
      target.removeAttribute('tabindex');
    });
  }

}); // DOMContentLoaded
