from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
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

def predict_fish(image_path):
    """
    Realiza la predicción completa del pez

    Args:
        image_path: Ruta a la imagen o BytesIO

    Returns:
        dict con las predicciones
    """
    if model is None:
        raise Exception("Modelo no cargado")

    # Preprocesar imagen
    img_array = preprocess_image(image_path)

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
        "summary": f"{SPECIES_NAMES[species_id]} {CLASS_NAMES[classification_id]}"
    }

    return result

# Cargar el modelo al iniciar
with app.app_context():
    load_model()

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
