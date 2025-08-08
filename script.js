// CAROUSEL FUNCTIONALITY
const track = document.querySelector('.carousel-track');
const prevBtn = document.querySelector('.prev');
const nextBtn = document.querySelector('.next');
let index = 0;

nextBtn.addEventListener('click', () => {
    if (index < track.children.length - 1) index++;
    track.style.transform = `translateX(${-index * 270}px)`;
});

prevBtn.addEventListener('click', () => {
    if (index > 0) index--;
    track.style.transform = `translateX(${-index * 270}px)`;
});

// FORM SUBMISSION TO WHATSAPP
document.querySelectorAll('form').forEach(form => {
    form.addEventListener('submit', e => {
        e.preventDefault();
        let formData = new FormData(form);
        let message = 'New Request:%0A';
        formData.forEach((value, key) => {
            message += `${key}: ${value}%0A`;
        });
        let phoneNumber = '254722704997'; // Replace with your WhatsApp number
        window.open(`https://wa.me/${phoneNumber}?text=${message}`, '_blank');
    });
});
