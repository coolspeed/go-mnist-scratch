const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const predictButton = document.getElementById('predictButton');
const clearButton = document.getElementById('clearButton');
const predictionResult = document.getElementById('predictionResult');
const inferenceTime = document.getElementById('inferenceTime');
const pixelGridDiv = document.getElementById('pixelGrid'); // We might not update this in real-time anymore

const CANVAS_SIZE = 280;

// Initialize canvas with white background
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

// Drawing state
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Event Listeners for smooth drawing
canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
});

canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseout', () => isDrawing = false);

function draw(e) {
    if (!isDrawing) return;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 20; // Thicker brush for better downsampling visibility
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();

    [lastX, lastY] = [e.offsetX, e.offsetY];
}

// Clear Button
clearButton.addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    predictionResult.textContent = '_';
    inferenceTime.textContent = '_';
    pixelGridDiv.innerHTML = ''; // Clear debug grid
});

// Predict Button
predictButton.addEventListener('click', async () => {
    // 1. Downsample to 28x28
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');

    // Smooth scaling
    tempCtx.imageSmoothingEnabled = true;
    tempCtx.imageSmoothingQuality = 'high';
    
    // Draw the main canvas onto the small canvas
    tempCtx.drawImage(canvas, 0, 0, 28, 28);

    // 2. Extract pixel data
    const imgData = tempCtx.getImageData(0, 0, 28, 28);
    const pixels = [];

    // Visualize the downsampled grid for debugging (optional)
    pixelGridDiv.innerHTML = '';
    pixelGridDiv.style.gridTemplateColumns = `repeat(28, 10px)`;
    pixelGridDiv.style.gridTemplateRows = `repeat(28, 10px)`;

    for (let i = 0; i < imgData.data.length; i += 4) {
        // Canvas is White background, Black drawing.
        // MNIST is Black background(0), White drawing(1).
        // So we need to invert and normalize.
        // R, G, B are same in grayscale.
        const colorVal = imgData.data[i]; // 0(black) to 255(white)
        
        // Invert: 0(black) -> 1.0, 255(white) -> 0.0
        let normalized = (255 - colorVal) / 255.0;
        
        // Improve contrast/thresholding to reduce noise
        // if (normalized < 0.1) normalized = 0; 
        
        pixels.push(normalized);

        // Update debug grid visualization
        const pixelDiv = document.createElement('div');
        pixelDiv.classList.add('pixel');
        // Visualize intensity
        const grayVal = Math.floor((1 - normalized) * 255);
        pixelDiv.style.backgroundColor = `rgb(${grayVal}, ${grayVal}, ${grayVal})`;
        if (normalized > 0.1) { 
             // pixelDiv.classList.add('active'); // Simply use color for better feedback
        }
        pixelGridDiv.appendChild(pixelDiv);
    }

    // 3. Send to server
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: pixels.join(',') }),
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        predictionResult.textContent = data.prediction;
        inferenceTime.textContent = data.duration_us;
    } catch (error) {
        console.error('Error:', error);
        predictionResult.textContent = 'Err';
        inferenceTime.textContent = '_';
    }
});