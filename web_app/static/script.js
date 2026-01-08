const BACKEND_URL = '/predict';
const REALTIME_URL = '/predict_realtime';
const QUALITY_URL = '/analyze_quality';

let videoStream;
let realtimeInterval = null;
let isRealtimeActive = false;
let qualityCheckInterval = null;
let isQualityCheckActive = false;
let autoCaptureEnabled = false;
let predictionHistory = [];

// Elementos del DOM
const imageUpload = document.getElementById('imageUpload');
const cameraButton = document.getElementById('cameraButton');
const classifyButton = document.getElementById('classifyButton');
const realtimeButton = document.getElementById('realtimeButton');
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
const realtimeOverlay = document.getElementById('realtimeOverlay');
const closeModalButton = document.getElementById('closeModal');

// Ocultar elementos al cargar
window.addEventListener('DOMContentLoaded', () => {
    hideElement(classifyButton);
    hideElement(realtimeButton);
    hideElement(loadingDiv);
    hideElement(errorDiv);
    hideElement(resultsDiv);
    hideElement(previewImage);
    hideElement(videoFeed);
    hideElement(canvas);
    hideElement(realtimeOverlay);

    // Cargar historial
    loadHistory();
});

// Utilidades
function hideElement(element) {
    if (element) {
        if (element.id === 'results') {
            element.classList.remove('active');
        } else {
            element.style.display = 'none';
        }
    }
}

function showElement(element, displayType = 'block') {
    if (element) {
        if (element.id === 'results') {
            element.classList.add('active');
        } else {
            element.style.display = displayType;
        }
    }
}

// Funciones para modal
function openModal() {
    resultsDiv.classList.add('active');
    document.body.style.overflow = 'hidden'; // Evitar scroll del body
}

function closeModal() {
    resultsDiv.classList.remove('active');
    document.body.style.overflow = ''; // Restaurar scroll del body
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

        // Añadir parámetro de Grad-CAM si está habilitado
        const gradcamToggle = document.getElementById('gradcamToggle');
        if (gradcamToggle && gradcamToggle.checked) {
            formData.append('enable_gradcam', 'true');
        }

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

// Agregar predicción al historial
function addToHistory(data, imageData) {
    const historyItem = {
        timestamp: new Date().toISOString(),
        species: data.species,
        classification: data.classification,
        confidence: ((data.species_confidence + data.classification_confidence) / 2 * 100).toFixed(0),
        imageData: imageData,
        fullData: data
    };

    predictionHistory.unshift(historyItem);

    // Limitar a las últimas 10 predicciones
    if (predictionHistory.length > 10) {
        predictionHistory.pop();
    }

    // Guardar en localStorage
    try {
        localStorage.setItem('predictionHistory', JSON.stringify(predictionHistory));
    } catch (e) {
        console.warn('No se pudo guardar el historial en localStorage:', e);
    }

    updateHistoryDisplay();
}

// Cargar historial desde localStorage
function loadHistory() {
    try {
        const stored = localStorage.getItem('predictionHistory');
        if (stored) {
            predictionHistory = JSON.parse(stored);
            updateHistoryDisplay();
        }
    } catch (e) {
        console.warn('No se pudo cargar el historial:', e);
    }
}

// Actualizar visualización del historial
function updateHistoryDisplay() {
    const historyContainer = document.getElementById('historyContainer');
    if (!historyContainer) return;

    if (predictionHistory.length === 0) {
        historyContainer.innerHTML = '<p class="text-muted text-center">No hay predicciones recientes</p>';
        return;
    }

    historyContainer.innerHTML = predictionHistory.map((item, index) => `
        <div class="history-item" onclick="showHistoryItem(${index})">
            <img src="${item.imageData}" alt="Predicción ${index + 1}" class="history-thumbnail">
            <div class="history-info">
                <div class="history-species">${item.species}</div>
                <div class="history-classification">${item.classification}</div>
                <div class="history-confidence">${item.confidence}%</div>
                <div class="history-time">${formatTimestamp(item.timestamp)}</div>
            </div>
        </div>
    `).join('');
}

// Mostrar item del historial
function showHistoryItem(index) {
    const item = predictionHistory[index];
    if (item) {
        displayResults(item.fullData);
        const modalImage = document.getElementById('modalImage');
        modalImage.src = item.imageData;
    }
}

// Formatear timestamp
function formatTimestamp(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'Ahora';
    if (diffMins < 60) return `Hace ${diffMins} min`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `Hace ${diffHours}h`;
    return date.toLocaleDateString('es-ES');
}

// Mostrar resultados
function displayResults(data) {
    // Mostrar la imagen en el modal
    const modalImage = document.getElementById('modalImage');
    let imageData;
    if (previewImage.style.display !== 'none' && previewImage.src) {
        modalImage.src = previewImage.src;
        imageData = previewImage.src;
    } else if (canvas.style.display !== 'none') {
        imageData = canvas.toDataURL('image/jpeg');
        modalImage.src = imageData;
    }

    // Agregar al historial
    if (imageData) {
        addToHistory(data, imageData);
    }

    // Resultado principal compacto con advertencias si las hay
    const speciesIcon = data.species === 'Dorada' ? '🐟' : (data.species === 'Lubina' ? '🐠' : '🐡');
    const classificationColor = data.classification === 'Salvaje' ? 'success' : 'warning';

    let warningsHTML = '';
    if (data.warnings && data.warnings.length > 0) {
        warningsHTML = `
            <div class="alert alert-warning mt-3 mb-3" role="alert">
                <strong><i class="bi bi-exclamation-triangle-fill"></i> Advertencias:</strong>
                ${data.warnings.map(w => `
                    <div class="mt-2">
                        <strong>${w.message}</strong>
                        <div class="small">${w.detail}</div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    mainResult.innerHTML = `
        ${warningsHTML}
        <h4>${speciesIcon} ${data.species} - ${data.classification}</h4>
    `;

    // Grid 2x2 de información
    detailedInfo.innerHTML = `
        <div class="info-card">
            <i class="bi bi-fish text-primary"></i>
            <h5>ESPECIE</h5>
            <div class="h2 text-primary">${(data.species_confidence * 100).toFixed(0)}%</div>
        </div>
        <div class="info-card">
            <i class="bi bi-award text-${classificationColor}"></i>
            <h5>CLASIFICACIÓN</h5>
            <div class="h2 text-${classificationColor}">${(data.classification_confidence * 100).toFixed(0)}%</div>
        </div>
        <div class="info-card">
            <i class="bi bi-eye text-info"></i>
            <h5>DETECCIÓN</h5>
            <div class="h2 text-info">${(data.fish_confidence * 100).toFixed(0)}%</div>
        </div>
        <div class="info-card">
            <i class="bi bi-check-circle text-success"></i>
            <h5>PRECISIÓN</h5>
            <div class="h2 text-success">${(((data.species_confidence + data.classification_confidence + data.fish_confidence) / 3) * 100).toFixed(0)}%</div>
        </div>
    `;

    // Mostrar probabilidades de especies si están disponibles
    let speciesDebug = '';
    if (data.species_probabilities) {
        const probDorada = data.species_probabilities.dorada * 100;
        const probLubina = data.species_probabilities.lubina * 100;
        const probOtro = data.species_probabilities.otro * 100;
        speciesDebug = `
            <div class="mb-4">
                <div class="mb-2">
                    <div class="d-flex justify-content-between mb-1">
                        <span class="small">🐟 Dorada</span>
                        <span class="small fw-bold">${probDorada.toFixed(0)}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar bg-info" style="width: ${probDorada}%;"></div>
                    </div>
                </div>
                <div class="mb-2">
                    <div class="d-flex justify-content-between mb-1">
                        <span class="small">🐠 Lubina</span>
                        <span class="small fw-bold">${probLubina.toFixed(0)}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar bg-secondary" style="width: ${probLubina}%;"></div>
                    </div>
                </div>
                <div>
                    <div class="d-flex justify-content-between mb-1">
                        <span class="small">🐡 Otro</span>
                        <span class="small fw-bold">${probOtro.toFixed(0)}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar bg-dark" style="width: ${probOtro}%;"></div>
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

    // Mostrar Grad-CAM si está disponible
    if (data.gradcam && !data.gradcam.error) {
        const gradcamSection = document.getElementById('gradcamSection');
        if (gradcamSection) {
            gradcamSection.style.display = 'block';
            gradcamSection.innerHTML = `
                <h5 class="mb-3 mt-4">
                    <i class="bi bi-binoculars"></i> Explicabilidad del Modelo (Grad-CAM)
                </h5>
                <p class="text-muted small">Las áreas rojas indican las zonas que más influyeron en la decisión del modelo.</p>
                <div class="row">
                    <div class="col-md-6 mb-3">
                        <div class="gradcam-container">
                            <h6 class="text-center mb-2">Especie: ${data.species}</h6>
                            <img src="${data.gradcam.species}" alt="Grad-CAM Especie" class="img-fluid rounded">
                        </div>
                    </div>
                    <div class="col-md-6 mb-3">
                        <div class="gradcam-container">
                            <h6 class="text-center mb-2">Clasificación: ${data.classification}</h6>
                            <img src="${data.gradcam.classification}" alt="Grad-CAM Clasificación" class="img-fluid rounded">
                        </div>
                    </div>
                </div>
                <p class="text-muted small text-center">Capa convolucional: ${data.gradcam.conv_layer}</p>
            `;
        }
    } else if (data.gradcam && data.gradcam.error) {
        console.warn('Error en Grad-CAM:', data.gradcam.error);
    }

    // Scroll a resultados
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Cargar imagen desde archivo
imageUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        stopRealtimeDetection(); // Detener si estaba activo
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            showElement(previewImage);
            hideElement(videoFeed);
            hideElement(canvas);
            showElement(classifyButton);
            hideElement(realtimeButton); // No mostrar botón en tiempo real para imágenes estáticas
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
        showElement(realtimeButton);

        // Mostrar botón de auto-captura
        const autoCaptureBtn = document.getElementById('autoCaptureButton');
        if (autoCaptureBtn) {
            showElement(autoCaptureBtn);
        }

        // Iniciar análisis de calidad automáticamente
        startQualityCheck();

        classifyButton.onclick = () => {
            stopRealtimeDetection(); // Detener detección en tiempo real si está activa
            stopQualityCheck(); // Detener análisis de calidad
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

// Iniciar análisis de calidad
function startQualityCheck() {
    if (isQualityCheckActive) return;

    console.log('Iniciando análisis de calidad');
    isQualityCheckActive = true;

    const startCheck = () => {
        if (videoFeed.videoWidth && videoFeed.videoHeight) {
            console.log('Video listo, iniciando análisis de calidad...');
            showElement(realtimeOverlay);
            analyzeQualityFrame();
            qualityCheckInterval = setInterval(analyzeQualityFrame, 1000);
        } else {
            setTimeout(startCheck, 200);
        }
    };

    startCheck();
}

// Detener análisis de calidad
function stopQualityCheck() {
    if (!isQualityCheckActive) return;

    isQualityCheckActive = false;

    if (qualityCheckInterval) {
        clearInterval(qualityCheckInterval);
        qualityCheckInterval = null;
    }

    realtimeOverlay.classList.remove('active');
    clearRealtimeOverlay();
}

// ======= DETECCIÓN EN TIEMPO REAL =======

// Analizar calidad de imagen en tiempo real
async function analyzeQualityFrame() {
    if (!videoFeed.videoWidth || !videoFeed.videoHeight) {
        return;
    }

    try {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = videoFeed.videoWidth;
        tempCanvas.height = videoFeed.videoHeight;
        const ctx = tempCanvas.getContext('2d');
        ctx.drawImage(videoFeed, 0, 0);

        const blob = await new Promise(resolve => tempCanvas.toBlob(resolve, 'image/jpeg', 0.8));

        const formData = new FormData();
        formData.append('image', blob, 'frame.jpg');

        const response = await fetch(QUALITY_URL, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (data.success) {
            updateQualityFeedback(data);

            // Auto-captura si está habilitada y la calidad es buena
            if (autoCaptureEnabled && data.is_good_quality && data.quality_score >= 80) {
                console.log('Auto-captura activada - Buena calidad detectada');
                autoCaptureEnabled = false; // Desactivar para evitar capturas múltiples
                setTimeout(() => {
                    classifyButton.click();
                }, 500);
            }
        }
    } catch (err) {
        console.error('Error en análisis de calidad:', err);
    }
}

// Actualizar feedback visual de calidad
function updateQualityFeedback(qualityData) {
    const overlayContent = realtimeOverlay.querySelector('.overlay-content');

    if (!overlayContent) return;

    const qualityColor = qualityData.quality_score >= 80 ? 'success' :
                        qualityData.quality_score >= 60 ? 'warning' : 'danger';

    const roiStatus = qualityData.roi_detected ?
        '<i class="bi bi-check-circle-fill text-success"></i> Objeto detectado' :
        '<i class="bi bi-x-circle-fill text-danger"></i> No se detecta objeto';

    let issuesHTML = '';
    if (qualityData.suggestions && qualityData.suggestions.length > 0) {
        issuesHTML = `
            <div class="quality-suggestions">
                <div class="suggestions-title">Sugerencias:</div>
                ${qualityData.suggestions.map(s => `<div class="suggestion-item"><i class="bi bi-lightbulb"></i> ${s}</div>`).join('')}
            </div>
        `;
    }

    overlayContent.innerHTML = `
        <div class="quality-feedback">
            <div class="quality-score-container">
                <div class="quality-score ${qualityColor}">
                    <div class="score-value">${qualityData.quality_score.toFixed(0)}</div>
                    <div class="score-label">Calidad</div>
                </div>
            </div>
            <div class="quality-status">
                <div class="status-item">${roiStatus}</div>
                ${qualityData.is_good_quality ?
                    '<div class="status-item text-success"><i class="bi bi-camera-fill"></i> Lista para capturar</div>' :
                    '<div class="status-item text-warning"><i class="bi bi-exclamation-triangle-fill"></i> Ajusta la imagen</div>'
                }
            </div>
            ${issuesHTML}
            ${qualityData.roi_bbox ? drawROIBox(qualityData.roi_bbox) : ''}
        </div>
    `;

    realtimeOverlay.classList.add('active');
}

// Dibujar ROI box overlay
function drawROIBox(bbox) {
    return `
        <div class="roi-indicator" style="
            position: absolute;
            left: ${(bbox.x / videoFeed.videoWidth) * 100}%;
            top: ${(bbox.y / videoFeed.videoHeight) * 100}%;
            width: ${(bbox.width / videoFeed.videoWidth) * 100}%;
            height: ${(bbox.height / videoFeed.videoHeight) * 100}%;
            border: 3px solid #00ff00;
            box-shadow: 0 0 10px rgba(0,255,0,0.5);
        "></div>
    `;
}

// Capturar frame del video y clasificar
async function captureAndClassifyFrame() {
    if (!videoFeed.videoWidth || !videoFeed.videoHeight) {
        console.log('Video no está listo:', videoFeed.videoWidth, videoFeed.videoHeight);
        return; // Video aún no está listo
    }

    try {
        // Crear canvas temporal para capturar frame
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = videoFeed.videoWidth;
        tempCanvas.height = videoFeed.videoHeight;
        const ctx = tempCanvas.getContext('2d');
        ctx.drawImage(videoFeed, 0, 0);

        // Convertir a blob con mayor calidad
        const blob = await new Promise(resolve => tempCanvas.toBlob(resolve, 'image/jpeg', 0.95));

        // Enviar al backend usando endpoint optimizado para tiempo real
        const formData = new FormData();
        formData.append('image', blob, 'frame.jpg');

        const response = await fetch(REALTIME_URL, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        console.log('Respuesta del backend:', data);

        if (data.success && data.is_fish) {
            // Pez detectado con éxito
            updateRealtimeOverlay(data, true);
        } else if (data.is_fish === false) {
            // No se detectó pez
            updateRealtimeOverlay({
                status: 'searching',
                species: 'Buscando pez...',
                classification: '',
                fish_confidence: data.fish_confidence || 0,
                message: data.message || 'Enfoca un pez en la cámara'
            }, false);
        } else {
            // Error en la detección
            updateRealtimeOverlay({
                status: 'error',
                species: 'Error',
                classification: data.error || 'No detectado',
                fish_confidence: 0
            }, false);
        }
    } catch (err) {
        console.error('Error en detección en tiempo real:', err);
        updateRealtimeOverlay({
            status: 'error',
            species: 'Error de conexión',
            classification: '',
            fish_confidence: 0,
            message: 'Verifica la conexión con el servidor'
        }, false);
    }
}

// Actualizar overlay con resultados mejorado con más información visual
function updateRealtimeOverlay(data, fishDetected) {
    console.log('Actualizando overlay:', data, 'Fish detected:', fishDetected);
    const overlayContent = realtimeOverlay.querySelector('.overlay-content');

    if (!overlayContent) {
        console.error('No se encontró overlay-content');
        return;
    }

    if (!fishDetected) {
        // Modo búsqueda o error
        const statusClass = data.status === 'searching' ? 'searching' : 'error';
        overlayContent.innerHTML = `
            <div class="overlay-status ${statusClass}">
                <i class="bi ${data.status === 'searching' ? 'bi-search' : 'bi-exclamation-triangle'}"></i>
                <div class="status-text">${data.species}</div>
                ${data.message ? `<div class="status-message">${data.message}</div>` : ''}
            </div>
        `;
    } else {
        // Pez detectado - mostrar información completa
        const speciesIcon = data.species === 'Dorada' ? '🐟' : (data.species === 'Lubina' ? '🐠' : '🐡');
        const classificationColor = data.classification === 'Salvaje' ? 'success' : 'warning';

        // Calcular porcentajes
        const fishConf = (data.fish_confidence * 100).toFixed(0);
        const speciesConf = (data.species_confidence * 100).toFixed(0);
        const classConf = (data.classification_confidence * 100).toFixed(0);

        const doradaProb = (data.species_probabilities.dorada * 100).toFixed(0);
        const lubinaProb = (data.species_probabilities.lubina * 100).toFixed(0);
        const otroProb = (data.species_probabilities.otro * 100).toFixed(0);
        const cultivadaProb = (data.probabilities.cultivada * 100).toFixed(0);
        const salvajeProb = (data.probabilities.salvaje * 100).toFixed(0);

        // Advertencia de confianza baja
        const lowConfidenceWarning = speciesConf < 85 ? `
            <div class="overlay-warning">
                <i class="bi bi-exclamation-triangle-fill"></i>
                Confianza baja (${speciesConf}%) - Repite la foto
            </div>
        ` : '';

        overlayContent.innerHTML = `
            <div class="overlay-detection-info">
                ${lowConfidenceWarning}
                <!-- Badges principales -->
                <div class="overlay-main-badges">
                    <div class="overlay-badge species-badge">
                        <span class="badge-icon">${speciesIcon}</span>
                        <span>${data.species}</span>
                        <span class="badge-confidence">${speciesConf}%</span>
                    </div>
                    <div class="overlay-badge classification-badge ${classificationColor}">
                        <span>${data.classification}</span>
                        <span class="badge-confidence">${classConf}%</span>
                    </div>
                </div>

                <!-- Barras de progreso de especies -->
                <div class="overlay-probabilities">
                    <div class="prob-section">
                        <div class="prob-header">
                            <span class="prob-label">Especies</span>
                        </div>
                        <div class="prob-bar-container">
                            <div class="prob-bar-wrapper">
                                <div class="prob-bar-label">
                                    <span>🐟 Dorada</span>
                                    <span class="prob-value">${doradaProb}%</span>
                                </div>
                                <div class="prob-bar">
                                    <div class="prob-bar-fill species-dorada" style="width: ${doradaProb}%"></div>
                                </div>
                            </div>
                            <div class="prob-bar-wrapper">
                                <div class="prob-bar-label">
                                    <span>🐠 Lubina</span>
                                    <span class="prob-value">${lubinaProb}%</span>
                                </div>
                                <div class="prob-bar">
                                    <div class="prob-bar-fill species-lubina" style="width: ${lubinaProb}%"></div>
                                </div>
                            </div>
                            <div class="prob-bar-wrapper">
                                <div class="prob-bar-label">
                                    <span>🐡 Otro</span>
                                    <span class="prob-value">${otroProb}%</span>
                                </div>
                                <div class="prob-bar">
                                    <div class="prob-bar-fill species-otro" style="width: ${otroProb}%"></div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Barras de clasificación -->
                    <div class="prob-section">
                        <div class="prob-header">
                            <span class="prob-label">Clasificación</span>
                        </div>
                        <div class="prob-bar-container">
                            <div class="prob-bar-wrapper">
                                <div class="prob-bar-label">
                                    <span>Cultivada</span>
                                    <span class="prob-value">${cultivadaProb}%</span>
                                </div>
                                <div class="prob-bar">
                                    <div class="prob-bar-fill classification-cultivada" style="width: ${cultivadaProb}%"></div>
                                </div>
                            </div>
                            <div class="prob-bar-wrapper">
                                <div class="prob-bar-label">
                                    <span>Salvaje</span>
                                    <span class="prob-value">${salvajeProb}%</span>
                                </div>
                                <div class="prob-bar">
                                    <div class="prob-bar-fill classification-salvaje" style="width: ${salvajeProb}%"></div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Confianza de detección -->
                    <div class="overlay-detection-confidence">
                        <i class="bi bi-check-circle-fill"></i>
                        <span>Pez detectado: ${fishConf}%</span>
                    </div>
                </div>
            </div>
        `;
    }

    // Mostrar overlay
    realtimeOverlay.classList.add('active');
    console.log('Overlay activado, clases:', realtimeOverlay.className);
    console.log('Overlay display:', window.getComputedStyle(realtimeOverlay).display);
}

// Limpiar overlay
function clearRealtimeOverlay() {
    const overlayContent = realtimeOverlay.querySelector('.overlay-content');
    if (overlayContent) {
        overlayContent.innerHTML = '';
    }
}

// Iniciar detección en tiempo real
function startRealtimeDetection() {
    if (isRealtimeActive) return;

    // Verificar que el video está activo y listo
    if (!videoFeed || videoFeed.style.display === 'none' || !videoStream) {
        showError('Por favor, activa la cámara primero antes de iniciar la detección en vivo.');
        return;
    }

    console.log('Iniciando detección en tiempo real');
    isRealtimeActive = true;
    realtimeButton.classList.add('active');
    realtimeButton.innerHTML = '<i class="bi bi-stop-circle"></i> Detener';

    hideElement(resultsDiv);
    hideElement(errorDiv);

    // Esperar a que el video esté completamente cargado antes de comenzar
    const startDetection = () => {
        if (videoFeed.videoWidth && videoFeed.videoHeight) {
            console.log('Video listo, iniciando detección...');
            showElement(realtimeOverlay);
            captureAndClassifyFrame(); // Primera ejecución inmediata
            realtimeInterval = setInterval(captureAndClassifyFrame, 1500);
        } else {
            console.log('Esperando a que el video esté listo...');
            setTimeout(startDetection, 200);
        }
    };

    startDetection();
}

// Detener detección en tiempo real
function stopRealtimeDetection() {
    if (!isRealtimeActive) return;

    isRealtimeActive = false;
    realtimeButton.classList.remove('active');
    realtimeButton.innerHTML = '<i class="bi bi-camera-video"></i> Detección en Vivo';

    if (realtimeInterval) {
        clearInterval(realtimeInterval);
        realtimeInterval = null;
    }

    realtimeOverlay.classList.remove('active');
    clearRealtimeOverlay();
}

// Toggle detección en tiempo real
realtimeButton.addEventListener('click', () => {
    if (isRealtimeActive) {
        stopRealtimeDetection();
    } else {
        startRealtimeDetection();
    }
});

// Event listeners para el modal
closeModalButton.addEventListener('click', closeModal);

// Cerrar modal al hacer click en el overlay
document.querySelector('.modal-overlay').addEventListener('click', closeModal);

// Cerrar modal con la tecla Escape
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && resultsDiv.classList.contains('active')) {
        closeModal();
    }
});

// Event listener para auto-captura
const autoCaptureButton = document.getElementById('autoCaptureButton');
if (autoCaptureButton) {
    autoCaptureButton.addEventListener('click', () => {
        if (autoCaptureEnabled) {
            autoCaptureEnabled = false;
            autoCaptureButton.classList.remove('active');
            autoCaptureButton.innerHTML = '<i class="bi bi-magic"></i> Auto-Captura';
        } else {
            autoCaptureEnabled = true;
            autoCaptureButton.classList.add('active');
            autoCaptureButton.innerHTML = '<i class="bi bi-magic"></i> Auto-Captura (ON)';
        }
    });
}
