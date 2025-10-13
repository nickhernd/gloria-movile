const BACKEND_URL = 'http://localhost:5000/predict';

let videoStream;

// Elementos del DOM
const imageUpload = document.getElementById('imageUpload');
const cameraButton = document.getElementById('cameraButton');
const classifyButton = document.getElementById('classifyButton');
const previewImage = document.getElementById('previewImage');
const videoFeed = document.getElementById('videoFeed');
const canvas = document.getElementById('canvas');
const loadingDiv = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const errorMessage = document.getElementById('errorMessage');
const resultsDiv = document.getElementById('results');
const mainResult = document.getElementById('mainResult');
const detailedInfo = document.getElementById('detailedInfo');
const probabilities = document.getElementById('probabilities');

// Ocultar elementos al cargar
window.addEventListener('DOMContentLoaded', () => {
    hideElement(classifyButton);
    hideElement(loadingDiv);
    hideElement(errorDiv);
    hideElement(resultsDiv);
    hideElement(previewImage);
    hideElement(videoFeed);
    hideElement(canvas);
});

// Utilidades
function hideElement(element) {
    if (element) element.style.display = 'none';
}

function showElement(element, displayType = 'block') {
    if (element) element.style.display = displayType;
}

function showError(message) {
    errorMessage.textContent = message;
    showElement(errorDiv);
    hideElement(loadingDiv);
    hideElement(resultsDiv);
}

// Clasificar imagen
async function classifyImage(imageSource) {
    hideElement(resultsDiv);
    hideElement(errorDiv);
    showElement(loadingDiv);

    try {
        let blob;
        if (imageSource instanceof HTMLImageElement) {
            const response = await fetch(imageSource.src);
            blob = await response.blob();
        } else if (imageSource instanceof HTMLCanvasElement) {
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

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || data.message || `Error del servidor: ${response.status}`);
        }

        displayResults(data);
        showElement(resultsDiv);

    } catch (err) {
        showError(err.message);
        console.error('Error durante la clasificación:', err);
    } finally {
        hideElement(loadingDiv);
    }
}

// Mostrar resultados
function displayResults(data) {
    // Resultado principal
    const speciesIcon = data.species === 'Dorada' ? '🐟' : '🐠';
    const classificationColor = data.classification === 'Salvaje' ? 'success' : 'warning';

    mainResult.innerHTML = `
        <div class="text-center">
            <div class="display-3 mb-3">${speciesIcon}</div>
            <h2 class="mb-3">${data.summary}</h2>
            <div class="d-flex justify-content-center gap-3 flex-wrap">
                <span class="badge bg-primary fs-5 px-4 py-2">
                    ${data.species}
                </span>
                <span class="badge bg-${classificationColor} fs-5 px-4 py-2">
                    ${data.classification}
                </span>
            </div>
        </div>
    `;

    // Información detallada
    detailedInfo.innerHTML = `
        <div class="col-md-4">
            <div class="info-card rounded text-center">
                <h6 class="text-muted small mb-2">ESPECIE</h6>
                <p class="fw-bold fs-4 mb-0">${(data.species_confidence * 100).toFixed(0)}%</p>
            </div>
        </div>
        <div class="col-md-4">
            <div class="info-card rounded text-center">
                <h6 class="text-muted small mb-2">CLASIFICACIÓN</h6>
                <p class="fw-bold fs-4 mb-0">${(data.classification_confidence * 100).toFixed(0)}%</p>
            </div>
        </div>
        <div class="col-md-4">
            <div class="info-card rounded text-center">
                <h6 class="text-muted small mb-2">DETECCIÓN</h6>
                <p class="fw-bold fs-4 mb-0">${(data.fish_confidence * 100).toFixed(0)}%</p>
            </div>
        </div>
    `;

    // Mostrar probabilidades de especies si están disponibles
    let speciesDebug = '';
    if (data.species_probabilities) {
        const probDorada = data.species_probabilities.dorada * 100;
        const probLubina = data.species_probabilities.lubina * 100;
        speciesDebug = `
            <div class="mb-4">
                <div class="mb-2">
                    <div class="d-flex justify-content-between mb-1">
                        <span class="small">Dorada</span>
                        <span class="small fw-bold">${probDorada.toFixed(0)}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar bg-info" style="width: ${probDorada}%;"></div>
                    </div>
                </div>
                <div>
                    <div class="d-flex justify-content-between mb-1">
                        <span class="small">Lubina</span>
                        <span class="small fw-bold">${probLubina.toFixed(0)}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar bg-secondary" style="width: ${probLubina}%;"></div>
                    </div>
                </div>
            </div>
        `;
    }

    // Probabilidades
    const probCultivada = data.probabilities.cultivada * 100;
    const probSalvaje = data.probabilities.salvaje * 100;

    probabilities.innerHTML = `
        ${speciesDebug}
        <div class="mb-3">
            <div class="d-flex justify-content-between mb-1">
                <span class="small">Cultivada</span>
                <span class="small fw-bold">${probCultivada.toFixed(0)}%</span>
            </div>
            <div class="progress">
                <div class="progress-bar bg-warning" style="width: ${probCultivada}%;"></div>
            </div>
        </div>
        <div class="mb-3">
            <div class="d-flex justify-content-between mb-1">
                <span class="small">Salvaje</span>
                <span class="small fw-bold">${probSalvaje.toFixed(0)}%</span>
            </div>
            <div class="progress">
                <div class="progress-bar bg-success" style="width: ${probSalvaje}%;"></div>
            </div>
        </div>
    `;

    // Scroll a resultados
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Cargar imagen desde archivo
imageUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            showElement(previewImage);
            hideElement(videoFeed);
            hideElement(canvas);
            showElement(classifyButton);
            classifyButton.onclick = () => classifyImage(previewImage);
            hideElement(resultsDiv);
            hideElement(errorDiv);
        };
        reader.readAsDataURL(file);
    }
});

// Activar cámara
cameraButton.addEventListener('click', async () => {
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
        });
        videoFeed.srcObject = videoStream;
        showElement(videoFeed);
        hideElement(previewImage);
        hideElement(canvas);
        showElement(classifyButton);
        classifyButton.onclick = () => {
            const context = canvas.getContext('2d');
            canvas.width = videoFeed.videoWidth;
            canvas.height = videoFeed.videoHeight;
            context.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
            hideElement(videoFeed);
            showElement(canvas);
            if (videoStream) {
                videoStream.getTracks().forEach(track => track.stop());
            }
            classifyImage(canvas);
        };
        hideElement(resultsDiv);
        hideElement(errorDiv);
    } catch (err) {
        showError(`No se pudo acceder a la cámara: ${err.message}. Asegúrate de dar permisos.`);
        console.error('Error al acceder a la cámara:', err);
    }
});
