const track = document.querySelector('.carousel-track');
const prevBtn = document.querySelector('.carousel-btn.prev');
const nextBtn = document.querySelector('.carousel-btn.next');

let position = 0;
const cardWidth = 270; // product card width + gap

nextBtn.addEventListener('click', () => {
    if (Math.abs(position) < (track.children.length - 1) * cardWidth) {
        position -= cardWidth;
        track.style.transform = `translateX(${position}px)`;
    }
});

prevBtn.addEventListener('click', () => {
    if (position < 0) {
        position += cardWidth;
        track.style.transform = `translateX(${position}px)`;
    }
});
