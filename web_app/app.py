from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import clip
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

# Configuración de CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
preprocess = None

# Definición de clases de texto para cada especie
# DORADA (Sparus aurata / S_AURATA)
TEXT_LABELS_DORADA = [
    "a close-up of a fish with a uniform gray color that transitions to a white ventral side, an oval body and the mouth is slightly tilted upwards, the lateral fins are close to the body and small,",  # Cultivada
    "a close-up of a fish with a straight body, the color transitions from a darker upper side to a white ventral side and in which the mouth follows direction of the main axis of the body, the dorsal fin is symmetrical"  # Salvaje
]

# LUBINA (Dicentrarchus labrax / D_LABRAX)
TEXT_LABELS_LUBINA = [
    "a close-up of a dark grey fish with a curved bottom, the lateral fins are close to the body and small",  # Cultivada
    "a close-up of a fish with a flat bottom and in mouth follows the direction of the main axis of the body, the lateral fins separated from the body and oriented slightly backward and the dorsal fin is symmetrical"  # Salvaje
]

# Labels para validar que hay un pez
VALIDATION_LABELS = [
    "a close-up photo of a fish",
    "a photo without any fish, an empty photo, no fish present"
]

# Labels para detectar tipo de pez con características distintivas
SPECIES_LABELS = [
    "a photo of a dorada fish with oval rounded body, silver grey color with golden spots near eyes, steep forehead profile, sparus aurata gilthead seabream",
    "a photo of a sea bass fish with elongated streamlined body, dark silver grey color, straight head profile, prominent jaw, dicentrarchus labrax lubina"
]

CLASS_NAMES = {
    0: "Cultivada",
    1: "Salvaje"
}

SPECIES_NAMES = {
    0: "Dorada",
    1: "Lubina"
}

def load_model():
    """Carga el modelo CLIP"""
    global model, preprocess
    try:
        print(f"Cargando modelo CLIP en dispositivo: {device}")
        model, preprocess = clip.load('ViT-B/32', device)
        print("Modelo CLIP cargado exitosamente")
    except Exception as e:
        print(f"Error al cargar el modelo CLIP: {e}")
        model = None

def validate_fish_presence(image_path):
    """
    Valida que haya un pez en la imagen

    Returns:
        tuple: (is_fish_present, confidence)
    """
    if model is None:
        raise Exception("Modelo no cargado")

    # Cargar y preprocesar imagen
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Tokenizar textos de validación
    text_tokens = clip.tokenize(VALIDATION_LABELS).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

        # Normalizar
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calcular similaridad
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    probs = similarity[0].cpu().numpy()
    fish_confidence = float(probs[0])

    # Si la confianza de que hay un pez es mayor al 50%, es válido
    return fish_confidence > 0.5, fish_confidence

def detect_species(image_path):
    """
    Detecta automáticamente si es dorada o lubina usando ensemble de prompts

    Returns:
        tuple: (species_name, species_id, confidence)
    """
    if model is None:
        raise Exception("Modelo no cargado")

    # Cargar y preprocesar imagen
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Múltiples prompts por especie para mejor detección (ensemble)
    dorada_prompts = [
        "a photo of a dorada fish with oval rounded body, silver grey color with golden spots near eyes, steep forehead profile, sparus aurata gilthead seabream",
        "a gilthead sea bream fish with rounded oval shape and golden markings",
        "sparus aurata dorada with compact oval body shape"
    ]

    lubina_prompts = [
        "a photo of a sea bass fish with elongated streamlined body, dark silver grey color, straight head profile, prominent jaw, dicentrarchus labrax lubina",
        "a european sea bass with long streamlined body and prominent lower jaw",
        "dicentrarchus labrax lubina with elongated torpedo-shaped body"
    ]

    # Combinar todos los prompts
    all_prompts = dorada_prompts + lubina_prompts
    text_tokens = clip.tokenize(all_prompts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

        # Normalizar
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calcular similaridad
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    probs = similarity[0].cpu().numpy()

    # Promediar las probabilidades de cada especie (ensemble)
    num_dorada_prompts = len(dorada_prompts)
    num_lubina_prompts = len(lubina_prompts)

    dorada_prob = np.mean(probs[:num_dorada_prompts])
    lubina_prob = np.mean(probs[num_dorada_prompts:])

    # Normalizar probabilidades
    total = dorada_prob + lubina_prob
    dorada_prob_norm = dorada_prob / total
    lubina_prob_norm = lubina_prob / total

    # Seleccionar especie con mayor probabilidad
    if dorada_prob_norm > lubina_prob_norm:
        species_id = 0
        confidence = float(dorada_prob_norm)
    else:
        species_id = 1
        confidence = float(lubina_prob_norm)

    print(f"Detección de especie - Dorada: {dorada_prob_norm:.3f}, Lubina: {lubina_prob_norm:.3f}")

    return SPECIES_NAMES[species_id], species_id, confidence

def classify_fish(image_path, species_id):
    """
    Clasifica si el pez es cultivado o salvaje

    Args:
        image_path: Ruta a la imagen
        species_id: 0 para Dorada, 1 para Lubina

    Returns:
        dict con clase predicha, confianza y probabilidades
    """
    if model is None:
        raise Exception("Modelo no cargado")

    # Seleccionar text labels según la especie
    if species_id == 0:  # Dorada
        text_labels = TEXT_LABELS_DORADA
    else:  # Lubina
        text_labels = TEXT_LABELS_LUBINA

    # Cargar y preprocesar imagen
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Tokenizar textos
    text_tokens = clip.tokenize(text_labels).to(device)

    # Obtener embeddings
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

        # Normalizar
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calcular similaridad y aplicar softmax
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Obtener predicción
    probs = similarity[0].cpu().numpy()
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])

    return {
        "predicted_class": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": confidence,
        "probabilities": {
            "cultivada": float(probs[0]),
            "salvaje": float(probs[1])
        }
    }

# Cargar el modelo al iniciar la aplicación Flask
with app.app_context():
    load_model()

@app.route('/')
def index():
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

        # 1. Validar que hay un pez en la imagen
        is_fish, fish_confidence = validate_fish_presence(image_path)

        if not is_fish:
            return jsonify({
                "error": "No se detectó un pez en la imagen",
                "is_fish": False,
                "fish_confidence": fish_confidence,
                "message": "Por favor, toma una foto donde aparezca claramente un pez"
            }), 400

        # 2. Detectar tipo de pez (Dorada o Lubina)
        species_name, species_id, species_confidence = detect_species(image_path)

        # Calcular probabilidades de ambas especies para debug
        species_probs = {
            "dorada": species_confidence if species_id == 0 else (1 - species_confidence),
            "lubina": species_confidence if species_id == 1 else (1 - species_confidence)
        }

        # 3. Clasificar si es salvaje o cultivado
        classification = classify_fish(image_path, species_id)

        # Construir respuesta completa
        result = {
            "success": True,
            "is_fish": True,
            "fish_confidence": fish_confidence,
            "species": species_name,
            "species_id": species_id,
            "species_confidence": species_confidence,
            "species_probabilities": species_probs,
            "classification": classification["class_name"],
            "classification_id": classification["predicted_class"],
            "classification_confidence": classification["confidence"],
            "probabilities": classification["probabilities"],
            "filename": unique_filename,
            "summary": f"{species_name} {classification['class_name']}"
        }

        # Log para debug
        print(f"Predicción final - Especie: {species_name} ({species_confidence:.2%}), Clasificación: {classification['class_name']} ({classification['confidence']:.2%})")

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
    Retorna respuestas más rápidas con menor logging.
    """
    if model is None:
        return jsonify({"error": "Modelo no cargado."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No se encontró la imagen en la solicitud."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No se seleccionó ningún archivo."}), 400

    try:
        # Leer los datos de la imagen en memoria (sin guardar en disco para mayor velocidad)
        image_data = file.read()

        # Crear una ruta temporal en memoria
        from io import BytesIO
        image_path = BytesIO(image_data)

        # 1. Validar que hay un pez en la imagen
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Validación de pez
        text_tokens = clip.tokenize(VALIDATION_LABELS).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        probs = similarity[0].cpu().numpy()
        fish_confidence = float(probs[0])

        if fish_confidence <= 0.5:
            return jsonify({
                "success": False,
                "is_fish": False,
                "fish_confidence": fish_confidence,
                "message": "Buscando pez..."
            }), 200

        # 2. Detectar especie con prompts ensemble
        dorada_prompts = [
            "a photo of a dorada fish with oval rounded body, silver grey color with golden spots near eyes, steep forehead profile, sparus aurata gilthead seabream",
            "a gilthead sea bream fish with rounded oval shape and golden markings",
            "sparus aurata dorada with compact oval body shape"
        ]

        lubina_prompts = [
            "a photo of a sea bass fish with elongated streamlined body, dark silver grey color, straight head profile, prominent jaw, dicentrarchus labrax lubina",
            "a european sea bass with long streamlined body and prominent lower jaw",
            "dicentrarchus labrax lubina with elongated torpedo-shaped body"
        ]

        all_prompts = dorada_prompts + lubina_prompts
        text_tokens = clip.tokenize(all_prompts).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        probs = similarity[0].cpu().numpy()
        num_dorada_prompts = len(dorada_prompts)

        dorada_prob = np.mean(probs[:num_dorada_prompts])
        lubina_prob = np.mean(probs[num_dorada_prompts:])

        total = dorada_prob + lubina_prob
        dorada_prob_norm = dorada_prob / total
        lubina_prob_norm = lubina_prob / total

        if dorada_prob_norm > lubina_prob_norm:
            species_id = 0
            species_confidence = float(dorada_prob_norm)
        else:
            species_id = 1
            species_confidence = float(lubina_prob_norm)

        species_name = SPECIES_NAMES[species_id]

        # 3. Clasificar cultivado vs salvaje
        text_labels = TEXT_LABELS_DORADA if species_id == 0 else TEXT_LABELS_LUBINA
        text_tokens = clip.tokenize(text_labels).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        probs = similarity[0].cpu().numpy()
        predicted_class = int(np.argmax(probs))
        classification_confidence = float(probs[predicted_class])

        # Construir respuesta compacta para tiempo real
        result = {
            "success": True,
            "is_fish": True,
            "fish_confidence": fish_confidence,
            "species": species_name,
            "species_id": species_id,
            "species_confidence": species_confidence,
            "species_probabilities": {
                "dorada": float(dorada_prob_norm),
                "lubina": float(lubina_prob_norm)
            },
            "classification": CLASS_NAMES[predicted_class],
            "classification_id": predicted_class,
            "classification_confidence": classification_confidence,
            "probabilities": {
                "cultivada": float(probs[0]),
                "salvaje": float(probs[1])
            },
            "summary": f"{species_name} {CLASS_NAMES[predicted_class]}"
        }

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
        "device": device
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
