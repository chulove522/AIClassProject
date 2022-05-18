const slider = document.querySelector('.scroll-wrap');
let isDown = false;
let startX;
let scrollLeft;

slider.addEventListener('mousedown', (e) => {
    isDown = true;
    slider.classList.add('active');
    startX = e.pageX - slider.offsetLeft;
    scrollLeft = slider.scrollLeft;
});
slider.addEventListener('mouseleave', () => {
    isDown = false;
    slider.classList.remove('active');
});
slider.addEventListener('mouseup', () => {
    isDown = false;
    slider.classList.remove('active');
});
slider.addEventListener('mousemove', (e) => {
    if (!isDown) return;
    e.preventDefault();
    const x = e.pageX - slider.offsetLeft;
    const walk = (x - startX) * 3; //scroll-fast
    slider.scrollLeft = scrollLeft - walk;
    console.log(walk);
});

const generateImage = () => {
    const direction = Math.round(Math.random() * 360); //To output a volue between 0 and 360 in degrees to be given to the linear-gradient.

    const r1 = Math.round(Math.random() * 255); // Math.random() outputs a numer between 0(inclusive) & 1(exclusive) with around 17 decimal digits.
    const g1 = Math.round(Math.random() * 255); // We take this number and multiply it by 255. This number can never go above 255.
    const b1 = Math.round(Math.random() * 255); // We have a decimal number with we make an integer with Math.round() which rounds off the previous output.
    // to add random transparency to the image;         // This output is a whole number between 0 & 255 and can be assigned as values for the rgba() property.
    const a1 = Math.round(Math.random() * 10) / 10; //  *** for alpha values we need between 0 & 1 so we multiply the random number with 10 so as to get a value X.xxxxx, round it off so as to X and then                                                                divide it by 10 to get a decimal number or 1. ***  //

    const r2 = Math.round(Math.random() * 255);
    const g2 = Math.round(Math.random() * 255);

    const b2 = Math.round(Math.random() * 255);
    // to add random transparency to the image;
    const a2 = Math.round(Math.random() * 10) / 10;

    //Giving values to the linear gradiant.
    const background = `linear-gradient(${direction}deg, rgba(${r1},${g1},${b1},${a1}), rgba(${r2},${g2},${b2},${a2}))`;

    return background;
};

const mycards = document.querySelectorAll('.scroll-card');

scroll-card .forEach((card) => {
    card.style.backgroundImage = generateImage();

    card.addEventListener('click', (e) => {
        const body = document.querySelector('body');
        body.style.backgroundImage =
            e.currentTarget.style.backgroundImage;
    });
});