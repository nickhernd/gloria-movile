const BACKEND_URL = 'http://localhost:5000/predict';
const LABELS = ['Pez 1', 'Pez 2', 'Pez 3']; // ¡IMPORTANTE! Reemplaza con los nombres reales de tus clases de peces

let videoStream;

const imageUpload = document.getElementById('imageUpload');
const cameraButton = document.getElementById('cameraButton');
const classifyButton = document.getElementById('classifyButton');
const previewImage = document.getElementById('previewImage');
const videoFeed = document.getElementById('videoFeed');
const canvas = document.getElementById('canvas');
const predictionsDiv = document.getElementById('predictions');
const loadingDiv = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const resultsSection = document.querySelector('.results-section');

// Función para mostrar mensajes de error
function showError(message) {
    errorDiv.textContent = `Error: ${message}`;
    errorDiv.style.display = 'block';
    loadingDiv.style.display = 'none';
}

// Realizar la clasificación enviando la imagen al backend
async function classifyImage(imageSource) {
    predictionsDiv.innerHTML = '';
    resultsSection.style.display = 'none';
    loadingDiv.textContent = 'Clasificando...';
    loadingDiv.style.display = 'block';
    errorDiv.style.display = 'none';

    try {
        let blob;
        if (imageSource instanceof HTMLImageElement) {
            // Si es una imagen cargada desde archivo
            const response = await fetch(imageSource.src);
            blob = await response.blob();
        } else if (imageSource instanceof HTMLCanvasElement) {
            // Si es una imagen capturada de la cámara
            blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg'));
        } else {
            throw new Error("Fuente de imagen no soportada.");
        }

        const formData = new FormData();
        formData.append('image', blob, 'image.jpg');

        const response = await fetch(BACKEND_URL, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Error del servidor: ${response.status}`);
        }

        const data = await response.json();
        displayPredictions(data.predictions);
        resultsSection.style.display = 'block';

    } catch (err) {
        showError(`Error durante la clasificación: ${err.message}`);
        console.error('Error durante la clasificación:', err);
    } finally {
        loadingDiv.style.display = 'none';
    }
}

// Mostrar las predicciones de forma intuitiva
function displayPredictions(probabilities) {
    predictionsDiv.innerHTML = '';

    // Ordenar las predicciones de mayor a menor
    const sortedPredictions = probabilities.map((prob, index) => ({
        label: LABELS[index],
        probability: prob
    })).sort((a, b) => b.probability - a.probability);

    sortedPredictions.forEach(prediction => {
        const percentage = (prediction.probability * 100).toFixed(2);
        const predictionItem = document.createElement('div');
        predictionItem.className = 'prediction-item';
        predictionItem.innerHTML = `
            <span class="prediction-label">${prediction.label}</span>
            <div class="prediction-bar-container">
                <div class="prediction-bar" style="width: ${percentage}%;"></div>
            </div>
            <span class="prediction-percentage">${percentage}%</span>
        `;
        predictionsDiv.appendChild(predictionItem);
    });
}

// Evento para cargar imagen desde archivo
imageUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
            videoFeed.style.display = 'none';
            canvas.style.display = 'none';
            classifyButton.style.display = 'block';
            classifyButton.onclick = () => classifyImage(previewImage);
            resultsSection.style.display = 'none';
            errorDiv.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
});

// Evento para activar la cámara
cameraButton.addEventListener('click', async () => {
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        videoFeed.srcObject = videoStream;
        videoFeed.style.display = 'block';
        previewImage.style.display = 'none';
        canvas.style.display = 'none';
        classifyButton.style.display = 'block';
        classifyButton.onclick = () => {
            const context = canvas.getContext('2d');
            canvas.width = videoFeed.videoWidth;
            canvas.height = videoFeed.videoHeight;
            context.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
            videoFeed.style.display = 'none';
            canvas.style.display = 'block';
            if (videoStream) {
                videoStream.getTracks().forEach(track => track.stop());
            }
            classifyImage(canvas);
        };
        resultsSection.style.display = 'none';
        errorDiv.style.display = 'none';
    } catch (err) {
        showError(`No se pudo acceder a la cámara. Asegúrate de dar permisos. Detalles: ${err.message}`);
        console.error('Error al acceder a la cámara:', err);
    }
});

// No necesitamos cargar el modelo en el frontend, solo mostrar el botón de clasificar
window.onload = () => {
    loadingDiv.style.display = 'none'; // Ocultar el mensaje de carga del modelo
    classifyButton.style.display = 'none'; // Ocultar el botón de clasificar hasta que haya una imagen
};
