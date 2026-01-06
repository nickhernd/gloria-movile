from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import cv2
from skimage import filters, exposure

script_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(script_dir, 'templates')
static_dir = os.path.join(script_dir, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# Carpeta para guardar las imágenes subidas
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Modelo global
model = None

# Nombres de clases
SPECIES_NAMES = {
    0: "Dorada",
    1: "Lubina"
}

CLASS_NAMES = {
    0: "Cultivada",
    1: "Salvaje"
}

def load_model():
    """Carga el modelo MobileNet"""
    global model
    try:
        print("Cargando modelo MobileNet...")
        # Construir la ruta absoluta al modelo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'student_mobilenet_little.h5')
        print(f"Buscando modelo en: {model_path}")
        model = keras.models.load_model(model_path)
        print("Modelo cargado exitosamente")
        print(f"Entrada del modelo: {model.input_shape}")
        print(f"Salidas del modelo: {len(model.outputs)} salidas")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        model = None

def preprocess_image(image_path):
    """
    Preprocesa la imagen para el modelo MobileNet

    Args:
        image_path: Ruta a la imagen o objeto BytesIO

    Returns:
        numpy array con la imagen preprocesada
    """
    # Cargar imagen
    if isinstance(image_path, (str, bytes, os.PathLike)):
        img = Image.open(image_path)
    else:
        # Es un BytesIO
        img = Image.open(image_path)

    # Convertir a RGB si es necesario
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Redimensionar a 224x224
    img = img.resize((224, 224), Image.BILINEAR)

    # Convertir a array numpy
    img_array = np.array(img, dtype=np.float32)

    # Normalizar a [0, 1]
    img_array = img_array / 255.0

    # Expandir dimensiones para batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def detect_blur(image_array):
    """
    Detecta si la imagen está borrosa usando la varianza del Laplaciano

    Args:
        image_array: numpy array de la imagen en escala de grises

    Returns:
        dict con blur_score y is_blurry
    """
    # Convertir a escala de grises si es necesario
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array

    # Calcular varianza del Laplaciano
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Umbral: valores < 100 indican imagen borrosa
    is_blurry = laplacian_var < 100

    return {
        "blur_score": float(laplacian_var),
        "is_blurry": bool(is_blurry),
        "quality": "baja" if is_blurry else "buena"
    }

def detect_brightness_quality(image_array):
    """
    Analiza la calidad de iluminación de la imagen

    Args:
        image_array: numpy array de la imagen (RGB)

    Returns:
        dict con métricas de brillo
    """
    # Convertir a escala de grises
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array

    # Calcular brillo medio
    mean_brightness = np.mean(gray)

    # Calcular contraste (desviación estándar)
    contrast = np.std(gray)

    # Evaluar calidad
    is_too_dark = mean_brightness < 50
    is_too_bright = mean_brightness > 200
    is_low_contrast = contrast < 30

    quality = "buena"
    if is_too_dark:
        quality = "muy_oscura"
    elif is_too_bright:
        quality = "muy_clara"
    elif is_low_contrast:
        quality = "bajo_contraste"

    return {
        "mean_brightness": float(mean_brightness),
        "contrast": float(contrast),
        "is_too_dark": bool(is_too_dark),
        "is_too_bright": bool(is_too_bright),
        "is_low_contrast": bool(is_low_contrast),
        "quality": quality
    }

def detect_roi(image_array):
    """
    Detecta la región de interés (ROI) donde está el pez usando detección de bordes y contornos

    Args:
        image_array: numpy array de la imagen (RGB)

    Returns:
        dict con bbox y ROI cropped, o None si no se detecta
    """
    # Convertir a escala de grises
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array

    # Aplicar filtro de suavizado
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detección de bordes con Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Dilatar para conectar bordes
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Encontrar contornos
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Obtener el contorno más grande (asumimos que es el pez)
    largest_contour = max(contours, key=cv2.contourArea)

    # Calcular área del contorno
    contour_area = cv2.contourArea(largest_contour)
    image_area = gray.shape[0] * gray.shape[1]
    area_ratio = contour_area / image_area

    # Si el contorno es muy pequeño o muy grande, probablemente no es un pez
    if area_ratio < 0.05 or area_ratio > 0.95:
        return None

    # Obtener bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Añadir margen del 10%
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)

    x = max(0, x - margin_x)
    y = max(0, y - margin_y)
    w = min(image_array.shape[1] - x, w + 2 * margin_x)
    h = min(image_array.shape[0] - y, h + 2 * margin_y)

    # Extraer ROI
    roi = image_array[y:y+h, x:x+w]

    return {
        "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
        "roi": roi,
        "area_ratio": float(area_ratio),
        "detected": True
    }

def analyze_image_quality(image_path):
    """
    Analiza la calidad completa de la imagen y detecta ROI

    Args:
        image_path: Ruta a la imagen o objeto BytesIO

    Returns:
        dict con todas las métricas de calidad
    """
    # Cargar imagen
    if isinstance(image_path, (str, bytes, os.PathLike)):
        img = Image.open(image_path)
    else:
        img = Image.open(image_path)

    # Convertir a RGB si es necesario
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Convertir a numpy array
    img_array = np.array(img, dtype=np.uint8)

    # Analizar blur
    blur_metrics = detect_blur(img_array)

    # Analizar brillo
    brightness_metrics = detect_brightness_quality(img_array)

    # Detectar ROI
    roi_result = detect_roi(img_array)

    # Determinar calidad general
    quality_issues = []

    if blur_metrics["is_blurry"]:
        quality_issues.append("imagen_borrosa")

    if brightness_metrics["is_too_dark"]:
        quality_issues.append("muy_oscura")
    elif brightness_metrics["is_too_bright"]:
        quality_issues.append("muy_clara")

    if brightness_metrics["is_low_contrast"]:
        quality_issues.append("bajo_contraste")

    if roi_result is None:
        quality_issues.append("no_se_detecta_objeto")

    # Calcular score de calidad (0-100)
    quality_score = 100.0

    if blur_metrics["is_blurry"]:
        quality_score -= 30
    if brightness_metrics["is_too_dark"] or brightness_metrics["is_too_bright"]:
        quality_score -= 25
    if brightness_metrics["is_low_contrast"]:
        quality_score -= 15
    if roi_result is None:
        quality_score -= 30

    quality_score = max(0, quality_score)

    is_good_quality = quality_score >= 60 and len(quality_issues) == 0

    return {
        "quality_score": quality_score,
        "is_good_quality": is_good_quality,
        "quality_issues": quality_issues,
        "blur_metrics": blur_metrics,
        "brightness_metrics": brightness_metrics,
        "roi_detected": roi_result is not None,
        "roi_result": roi_result,
        "suggestions": get_quality_suggestions(quality_issues)
    }

def get_quality_suggestions(quality_issues):
    """
    Genera sugerencias basadas en los problemas de calidad detectados

    Args:
        quality_issues: lista de problemas detectados

    Returns:
        lista de sugerencias en español
    """
    suggestions = []

    if "imagen_borrosa" in quality_issues:
        suggestions.append("Mantén el teléfono estable o usa el temporizador")

    if "muy_oscura" in quality_issues:
        suggestions.append("Aumenta la iluminación o usa el flash")

    if "muy_clara" in quality_issues:
        suggestions.append("Reduce la luz directa o cambia el ángulo")

    if "bajo_contraste" in quality_issues:
        suggestions.append("Usa un fondo que contraste con el pez")

    if "no_se_detecta_objeto" in quality_issues:
        suggestions.append("Acerca el pez a la cámara y centra la imagen")

    return suggestions

def predict_fish(image_path, use_roi=True):
    """
    Realiza la predicción completa del pez

    Args:
        image_path: Ruta a la imagen o BytesIO
        use_roi: Si True, usa ROI detection para mejorar la predicción

    Returns:
        dict con las predicciones y métricas de calidad
    """
    if model is None:
        raise Exception("Modelo no cargado")

    # Analizar calidad de imagen
    quality_analysis = analyze_image_quality(image_path)

    # Preparar imagen para predicción
    image_to_process = image_path

    # Si ROI fue detectado y use_roi=True, usar ROI en lugar de imagen completa
    if use_roi and quality_analysis["roi_detected"]:
        roi_array = quality_analysis["roi_result"]["roi"]
        # Convertir ROI numpy array a PIL Image y luego a BytesIO
        roi_pil = Image.fromarray(roi_array.astype('uint8'), 'RGB')
        roi_bytes = io.BytesIO()
        roi_pil.save(roi_bytes, format='JPEG')
        roi_bytes.seek(0)
        image_to_process = roi_bytes

    # Preprocesar imagen
    img_array = preprocess_image(image_to_process)

    # Hacer predicción
    predictions = model.predict(img_array, verbose=0)

    # Interpretar resultados
    # predictions[0] = especie (logits)
    # predictions[1] = clasificación (probabilidades)

    species_logits = predictions[0][0]  # Shape: (2,)
    classification_probs = predictions[1][0]  # Shape: (2,)

    # Aplicar softmax a los logits de especie para obtener probabilidades
    species_probs = tf.nn.softmax(species_logits).numpy()

    # Obtener predicciones
    species_id = int(np.argmax(species_probs))
    species_confidence = float(species_probs[species_id])

    classification_id = int(np.argmax(classification_probs))
    classification_confidence = float(classification_probs[classification_id])

    # Estimación de confianza de detección de pez
    # Usamos la confianza promedio como proxy
    fish_confidence = float((species_confidence + classification_confidence) / 2)

    # Construir resultado
    result = {
        "success": True,
        "is_fish": fish_confidence > 0.5,  # Umbral para detectar pez
        "fish_confidence": fish_confidence,
        "species": SPECIES_NAMES[species_id],
        "species_id": species_id,
        "species_confidence": species_confidence,
        "species_probabilities": {
            "dorada": float(species_probs[0]),
            "lubina": float(species_probs[1])
        },
        "classification": CLASS_NAMES[classification_id],
        "classification_id": classification_id,
        "classification_confidence": classification_confidence,
        "probabilities": {
            "cultivada": float(classification_probs[0]),
            "salvaje": float(classification_probs[1])
        },
        "summary": f"{SPECIES_NAMES[species_id]} {CLASS_NAMES[classification_id]}",
        "quality_analysis": {
            "quality_score": quality_analysis["quality_score"],
            "is_good_quality": quality_analysis["is_good_quality"],
            "quality_issues": quality_analysis["quality_issues"],
            "suggestions": quality_analysis["suggestions"],
            "roi_detected": quality_analysis["roi_detected"],
            "roi_bbox": quality_analysis["roi_result"]["bbox"] if quality_analysis["roi_detected"] else None
        }
    }

    return result

# Cargar el modelo al iniciar
with app.app_context():
    load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No se encontró la imagen en la solicitud."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No se seleccionó ningún archivo."}), 400

    try:
        # Leer los datos de la imagen en memoria
        image_data = file.read()

        # Guardar la imagen en el servidor con un nombre único
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_filename = f"{timestamp}-{filename}"

        # Asegurarse de que la carpeta de subida exista
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        with open(image_path, 'wb') as f:
            f.write(image_data)
        print(f"Imagen guardada como: {unique_filename}")

        # Hacer predicción
        result = predict_fish(image_path)

        # Verificar si se detectó un pez
        if not result["is_fish"]:
            return jsonify({
                "error": "No se detectó un pez en la imagen",
                "is_fish": False,
                "fish_confidence": result["fish_confidence"],
                "message": "Por favor, toma una foto donde aparezca claramente un pez"
            }), 400

        # Agregar filename al resultado
        result["filename"] = unique_filename

        # Log para debug
        print(f"Predicción - Especie: {result['species']} ({result['species_confidence']:.2%}), "
              f"Clasificación: {result['classification']} ({result['classification_confidence']:.2%})")

        return jsonify(result)

    except Exception as e:
        print(f"Error durante la predicción: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error durante la predicción: {str(e)}"}), 500

@app.route('/predict_realtime', methods=['POST'])
def predict_realtime():
    """
    Endpoint optimizado para detección en tiempo real.
    No guarda imágenes en disco para mayor velocidad.
    """
    if model is None:
        return jsonify({"error": "Modelo no cargado."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No se encontró la imagen en la solicitud."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No se seleccionó ningún archivo."}), 400

    try:
        # Leer los datos de la imagen en memoria (sin guardar en disco)
        image_data = file.read()
        image_path = io.BytesIO(image_data)

        # Hacer predicción
        result = predict_fish(image_path)

        # Para tiempo real, siempre retornar success=True
        # El frontend maneja el caso de no detectar pez
        if not result["is_fish"]:
            return jsonify({
                "success": False,
                "is_fish": False,
                "fish_confidence": result["fish_confidence"],
                "message": "Buscando pez..."
            }), 200

        return jsonify(result)

    except Exception as e:
        print(f"Error en detección en tiempo real: {e}")
        return jsonify({"success": False, "error": f"Error: {str(e)}"}), 500

@app.route('/analyze_quality', methods=['POST'])
def analyze_quality():
    """
    Endpoint para análisis rápido de calidad de imagen sin hacer predicción.
    Útil para feedback en tiempo real.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No se encontró la imagen en la solicitud."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No se seleccionó ningún archivo."}), 400

    try:
        # Leer los datos de la imagen en memoria
        image_data = file.read()
        image_path = io.BytesIO(image_data)

        # Analizar calidad
        quality_analysis = analyze_image_quality(image_path)

        return jsonify({
            "success": True,
            "quality_score": quality_analysis["quality_score"],
            "is_good_quality": quality_analysis["is_good_quality"],
            "quality_issues": quality_analysis["quality_issues"],
            "suggestions": quality_analysis["suggestions"],
            "roi_detected": quality_analysis["roi_detected"],
            "roi_bbox": quality_analysis["roi_result"]["bbox"] if quality_analysis["roi_detected"] else None
        })

    except Exception as e:
        print(f"Error en análisis de calidad: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    """Endpoint para verificar el estado del servicio"""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": "MobileNetV2 Student"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
