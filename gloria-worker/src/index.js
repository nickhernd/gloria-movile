import { Router } from 'itty-router';
import * as tf from '@tensorflow/tfjs-core';
import { loadGraphModel } from '@tensorflow/tfjs-converter';
import '@tensorflow/tfjs-backend-wasm'; // Importa el backend WASM

// --- CONFIGURACIÓN DEL MODELO ---
// Estos valores son suposiciones y probablemente necesiten ajustarse
// según el modelo real que se entrenó.
const IMAGE_SIZE = 224; // El tamaño al que se redimensionará la imagen
// Ajusta estas clases según tu modelo
const SPECIES_CLASSES = ['Dorada', 'Lubina'];
const CLASSIFICATION_CLASSES = ['Salvaje', 'Cultivada'];

// --- Inicialización y Carga del Modelo (Singleton) ---
let modelPromise;

async function loadModel() {
    if (modelPromise) {
        return modelPromise;
    }

    console.log('Iniciando carga del modelo...');
    try {
        // Establecer el backend de TFJS a WASM, que es compatible con Workers
        await tf.setBackend('wasm');
        // Asegúrate de que el backend está listo
        await tf.ready();
        console.log('Backend de TFJS establecido y listo: ' + tf.getBackend());

        // Cargar el modelo desde la URL pública.
        // Cloudflare Pages servirá este archivo desde /tfjs-model/model.json
        const model = await loadGraphModel('/tfjs-model/model.json');

        console.log('Modelo cargado exitosamente desde URL.');
        return model;
    } catch (error) {
        console.error('Error al cargar el modelo:', error);
        throw error;
    }
}

// Iniciar la carga del modelo en un arranque en frío.
// Esto asegura que el modelo se cargue una sola vez.
modelPromise = loadModel();


// --- Lógica del Router y Predicción ---
const router = Router();

// Ruta de predicción principal
router.post('/predict', async (request) => {
    try {
        const model = await modelPromise; // Espera a que el modelo esté cargado
        if (!model) {
            return new Response(JSON.stringify({ error: 'El modelo no está disponible.' }), { status: 500, headers: { 'Content-Type': 'application/json' } });
        }

        const formData = await request.formData();
        const imageFile = formData.get('image');

        if (!imageFile) {
            return new Response(JSON.stringify({ error: 'No se proporcionó ninguna imagen.' }), { status: 400, headers: { 'Content-Type': 'application/json' } });
        }

        const imageBuffer = await imageFile.arrayBuffer();

        // Decodificar, redimensionar y normalizar la imagen
        const imageTensor = tf.tidy(() => {
            // tf.decodeImage funciona con un ArrayBuffer
            const decoded = tf.decodeImage(new Uint8Array(imageBuffer), 3); // 3 canales para RGB
            // Redimensionar a lo que el modelo espera
            const resized = tf.image.resizeBilinear(decoded, [IMAGE_SIZE, IMAGE_SIZE]);
            // Normalizar los valores de píxeles (ej: a [0, 1])
            const normalized = resized.div(tf.scalar(255.0));
            // Añadir una dimensión de batch [1, IMAGE_SIZE, IMAGE_SIZE, 3]
            return normalized.expandDims(0);
        });

        // Realizar la predicción
        const prediction = model.predict(imageTensor);

        // Limpiar el tensor de la imagen de la memoria de la GPU
        tf.dispose(imageTensor);

        // --- Post-procesamiento ---
        // Asumimos que el modelo tiene dos salidas: una para especie, una para clasificación (salvaje/cultivada)
        // La salida 'prediction' podría ser un solo tensor o un array de tensores.
        // Necesitamos adaptar esto a la salida REAL de tu modelo.
        let speciesOutput, classificationOutput;

        if (Array.isArray(prediction)) {
            // Si el modelo tiene múltiples salidas (ej: una para especie, otra para clasificación)
            speciesOutput = prediction[0];
            classificationOutput = prediction[1];
        } else {
            // Si el modelo tiene una única salida con todas las probabilidades
            // Aquí se necesitaría saber el orden de las probabilidades en el tensor de salida.
            // Por ejemplo, las primeras N para especie, las siguientes M para clasificación.
            // ESTO ES UNA SUPOSICIÓN: Dividimos el tensor de salida en dos si es una única salida.
            // Si tu modelo tiene una salida compleja, necesitarás ajustar esto.
            const totalClasses = SPECIES_CLASSES.length + CLASSIFICATION_CLASSES.length;
            if (prediction.shape[1] === totalClasses) {
                 speciesOutput = prediction.slice([0, 0], [1, SPECIES_CLASSES.length]);
                 classificationOutput = prediction.slice([0, SPECIES_CLASSES.length], [1, CLASSIFICATION_CLASSES.length]);
            } else {
                 // Fallback si la forma no coincide con las expectativas de salida combinada
                 console.warn("La forma de salida del modelo no coincide con el número esperado de clases combinadas. Usando la salida completa para ambas.");
                 speciesOutput = prediction;
                 classificationOutput = prediction;
            }
        }


        const speciesData = await speciesOutput.data();
        const classData = await classificationOutput.data();

        tf.dispose(prediction); // Limpiar memoria

        // Interpretar los resultados (esto depende fuertemente de cómo tu modelo fue entrenado)
        // Probabilidades de especie
        const species_probabilities = {};
        let maxSpeciesConfidence = 0;
        let predictedSpecies = '';
        SPECIES_CLASSES.forEach((name, i) => {
            species_probabilities[name.toLowerCase()] = speciesData[i] || 0;
            if (speciesData[i] > maxSpeciesConfidence) {
                maxSpeciesConfidence = speciesData[i];
                predictedSpecies = name;
            }
        });

        // Probabilidades de clasificación (Salvaje/Cultivada)
        const classification_probabilities = {};
        let maxClassificationConfidence = 0;
        let predictedClassification = '';
        CLASSIFICATION_CLASSES.forEach((name, i) => {
            classification_probabilities[name.toLowerCase()] = classData[i] || 0;
            if (classData[i] > maxClassificationConfidence) {
                maxClassificationConfidence = classData[i];
                predictedClassification = name;
            }
        });

        const result = {
            success: true,
            is_fish: true, // Asumimos que la imagen siempre contiene un pez
            fish_confidence: 0.99, // Valor de ejemplo, si tu modelo predice esto, inclúyelo
            species: predictedSpecies,
            species_confidence: maxSpeciesConfidence,
            classification: predictedClassification,
            classification_confidence: maxClassificationConfidence,
            species_probabilities: species_probabilities,
            probabilities: classification_probabilities // Renombrado para coincidir con frontend
        };

        return new Response(JSON.stringify(result), {
            headers: { 'Content-Type': 'application/json' },
        });

    } catch (error) {
        console.error('Error en la predicción:', error);
        return new Response(JSON.stringify({ error: `Error en el servidor: ${error.message}` }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    } finally {
        tf.disposeVariables(); // Limpiar tensores para evitar fugas de memoria
    }
});

// La ruta /predict_realtime usará la misma lógica de predicción
router.post('/predict_realtime', async (request) => {
    return router.handle(request.clone());
});


// --- Handler Principal del Worker ---
export default {
    async fetch(request, env, ctx) {
        const url = new URL(request.url);
        // Las peticiones de API las maneja nuestro router
        if (url.pathname.startsWith('/predict')) {
            return router.handle(request);
        }
        // Para todas las demás peticiones (archivos estáticos), usa el Pages asset server
        return env.ASSETS.fetch(request);
    },
};