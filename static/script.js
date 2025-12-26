const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const predictButton = document.getElementById('predictButton');
const clearButton = document.getElementById('clearButton');
const predictionResult = document.getElementById('predictionResult');
const pixelGridDiv = document.getElementById('pixelGrid');

const CANVAS_SIZE = 280;
const PIXEL_SIZE = 10; // Each drawn pixel will be 10x10 on a 280x280 canvas
const GRID_DIM = CANVAS_SIZE / PIXEL_SIZE; // 28x28 grid

let isDrawing = false;
let pixels = Array(GRID_DIM * GRID_DIM).fill(0); // 28*28 = 784 pixels, 0 for white, 1 for black/drawn

// Initialize canvas
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
ctx.lineWidth = PIXEL_SIZE; // The "brush" size will be one grid cell

// Event Listeners for drawing
canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    drawPixel(e);
});

canvas.addEventListener('mousemove', (e) => {
    if (isDrawing) {
        drawPixel(e);
    }
});

canvas.addEventListener('mouseup', () => {
    isDrawing = false;
    ctx.beginPath();
});

canvas.addEventListener('mouseout', () => {
    isDrawing = false;
    ctx.beginPath();
});

function drawPixel(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Get grid coordinates
    const gridX = Math.floor(x / PIXEL_SIZE);
    const gridY = Math.floor(y / PIXEL_SIZE);

    if (gridX >= 0 && gridX < GRID_DIM && gridY >= 0 && gridY < GRID_DIM) {
        const index = gridY * GRID_DIM + gridX;
        pixels[index] = 1; // Mark as drawn

        // Draw on canvas
        ctx.fillStyle = 'black';
        ctx.fillRect(gridX * PIXEL_SIZE, gridY * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);

        // Update visual pixel grid
        updatePixelGrid();
    }
}

function updatePixelGrid() {
    pixelGridDiv.innerHTML = ''; // Clear previous grid
    pixelGridDiv.style.gridTemplateColumns = `repeat(${GRID_DIM}, ${PIXEL_SIZE}px)`;
    pixelGridDiv.style.gridTemplateRows = `repeat(${GRID_DIM}, ${PIXEL_SIZE}px)`;

    for (let i = 0; i < pixels.length; i++) {
        const pixelDiv = document.createElement('div');
        pixelDiv.classList.add('pixel');
        if (pixels[i] === 1) {
            pixelDiv.classList.add('active');
        }
        pixelGridDiv.appendChild(pixelDiv);
    }
}

// Initial pixel grid display
updatePixelGrid();

// Clear Button
clearButton.addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    pixels.fill(0);
    predictionResult.textContent = '_';
    updatePixelGrid();
});

// Predict Button
predictButton.addEventListener('click', async () => {
    // Convert pixels array to comma-separated string
    const imageDataString = pixels.join(',');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageDataString }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        const data = await response.json();
        predictionResult.textContent = data.prediction;
    } catch (error) {
        console.error('Error during prediction:', error);
        predictionResult.textContent = 'Error';
    }
});
