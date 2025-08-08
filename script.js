document.addEventListener('DOMContentLoaded', () => {
    // Smooth scrolling for navigation links
    document.querySelectorAll('nav a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Carousel functionality for the services section
    const carouselContainer = document.querySelector('.carousel-container');
    const carouselTrack = document.querySelector('.carousel-track');
    const prevBtn = document.querySelector('.carousel-btn.prev');
    const nextBtn = document.querySelector('.carousel-btn.next');
    let currentIndex = 0;

    // Helper function to update the carousel position
    function updateCarousel() {
        const itemWidth = carouselTrack.children[0].offsetWidth;
        carouselTrack.style.transform = `translateX(-${currentIndex * itemWidth}px)`;
    }

    nextBtn.addEventListener('click', () => {
        // Prevent clicking too fast
        carouselTrack.style.transition = 'transform 0.5s ease-in-out';
        currentIndex++;
        // If we reach the end, reset to the first item with a quick transition
        if (currentIndex > carouselTrack.children.length - 1) {
            currentIndex = 0;
            carouselTrack.style.transition = 'none';
        }
        updateCarousel();
    });

    prevBtn.addEventListener('click', () => {
        // Prevent clicking too fast
        carouselTrack.style.transition = 'transform 0.5s ease-in-out';
        currentIndex--;
        // If we reach the beginning, loop back to the end
        if (currentIndex < 0) {
            currentIndex = carouselTrack.children.length - 1;
            carouselTrack.style.transition = 'none';
        }
        updateCarousel();
    });

    // Recalculate carousel position on window resize
    window.addEventListener('resize', updateCarousel);
});
