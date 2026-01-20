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
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(script_dir, 'templates')
static_dir = os.path.join(script_dir, 'static')

# ==================== ARQUITECTURA CONVNEXT ====================
class ConvNeXtBlock(nn.Module):
    """Bloque ConvNeXt con LayerScale"""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # MLP con nombres específicos para coincidir con state_dict
        class MLP(nn.Module):
            def __init__(self, in_features, hidden_features):
                super().__init__()
                self.fc1 = nn.Linear(in_features, hidden_features)
                self.act = nn.GELU()
                self.fc2 = nn.Linear(hidden_features, in_features)

            def forward(self, x):
                x = self.fc1(x)
                x = self.act(x)
                x = self.fc2(x)
                return x

        self.mlp = MLP(dim, 4 * dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.conv_dw(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + x
        return x


class ConvNeXt(nn.Module):
    """Arquitectura ConvNeXt"""
    def __init__(self, in_chans=3, num_classes=2, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()

        # Stem con nomenclatura específica
        class Stem(nn.Module):
            def __init__(self, in_chans, out_chans):
                super().__init__()
                self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=4, stride=4, bias=True)
                self.norm = nn.LayerNorm(out_chans, eps=1e-6)

            def forward(self, x):
                x = self.conv(x)
                x = x.permute(0, 2, 3, 1)
                x = self.norm(x)
                x = x.permute(0, 3, 1, 2)
                return x

        # Usar Sequential para poder acceder con índices numéricos
        self.stem = nn.Sequential()
        self.stem.add_module('0', nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, bias=True))
        self.stem.add_module('1', nn.LayerNorm(dims[0], eps=1e-6))

        # Stages
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.ModuleDict()

            # Downsample (excepto el primer stage)
            if i > 0:
                downsample = nn.Sequential()
                downsample.add_module('0', nn.LayerNorm(dims[i-1], eps=1e-6))
                downsample.add_module('1', nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2, bias=True))
                stage['downsample'] = downsample

            # Blocks
            blocks = nn.ModuleList([ConvNeXtBlock(dims[i]) for _ in range(depths[i])])
            stage['blocks'] = blocks

            self.stages.append(stage)

        # Head con nomenclatura específica
        class Head(nn.Module):
            def __init__(self, in_features, num_classes):
                super().__init__()
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                self.norm = nn.LayerNorm(in_features, eps=1e-6)
                self.fc = nn.Linear(in_features, num_classes)

            def forward(self, x):
                x = self.global_pool(x)
                x = x.flatten(1)
                x = self.norm(x)
                x = self.fc(x)
                return x

        self.head = Head(dims[-1], num_classes)

    def forward(self, x):
        # Stem
        x = self.stem[0](x)
        x = x.permute(0, 2, 3, 1)
        x = self.stem[1](x)
        x = x.permute(0, 3, 1, 2)

        # Stages
        for i, stage in enumerate(self.stages):
            # Downsample si existe
            if 'downsample' in stage and i > 0:
                x = x.permute(0, 2, 3, 1)
                x = stage['downsample'][0](x)
                x = x.permute(0, 3, 1, 2)
                x = stage['downsample'][1](x)

            # Apply blocks
            for block in stage['blocks']:
                x = block(x)

        # Head
        x = self.head(x)

        return x

# ==================== FIN ARQUITECTURA CONVNEXT ====================

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# Carpeta para guardar las imágenes subidas
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Modelos globales
model = None  # Modelo TensorFlow antiguo (opcional, por compatibilidad)
model_fish_detector = None  # Modelo PyTorch: detecta si hay pez
model_species_classifier = None  # Modelo PyTorch: clasifica entre dorada y lubina
model_wild_cultivated = None  # Modelo PyTorch: clasifica cultivada vs salvaje

# Nombres de clases para modelos PyTorch
FISH_DETECTOR_NAMES = {
    0: "No Pez",
    1: "Pez"
}

SPECIES_NAMES_PYTORCH = {
    0: "Dorada",
    1: "Lubina"
}

# Nombres de clases para modelo TensorFlow antiguo (por compatibilidad)
SPECIES_NAMES = {
    0: "Dorada",
    1: "Lubina",
    2: "Otro"
}

CLASS_NAMES = {
    0: "Cultivada",
    1: "Salvaje"
}

# Threshold de confianza mínima
CONFIDENCE_THRESHOLD = 0.85

# Device para PyTorch (CPU o CUDA)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {DEVICE}")

def load_pytorch_models():
    """Carga los tres modelos PyTorch de detección y clasificación"""
    global model_fish_detector, model_species_classifier, model_wild_cultivated

    try:
        # Construir rutas a los modelos
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        fish_detector_path = os.path.join(parent_dir, 'model', 'best_nopez_pez.pth')
        species_classifier_path = os.path.join(parent_dir, 'model', 'best_aurata_labrax.pth')
        wild_cultivated_path = os.path.join(parent_dir, 'model', 'best_cautiva_salvaje.pth')

        # Cargar modelo detector de pez
        print("Cargando modelo detector de pez (best_nopez_pez)...")
        print(f"Ruta: {fish_detector_path}")
        model_fish_detector = ConvNeXt(in_chans=3, num_classes=2)
        state_dict = torch.load(fish_detector_path, map_location=DEVICE)
        model_fish_detector.load_state_dict(state_dict)
        model_fish_detector.to(DEVICE)
        model_fish_detector.eval()
        print("✓ Modelo detector de pez cargado exitosamente")

        # Cargar modelo clasificador de especies
        print("Cargando modelo clasificador de especies (best_aurata_labrax)...")
        print(f"Ruta: {species_classifier_path}")
        model_species_classifier = ConvNeXt(in_chans=3, num_classes=2)
        state_dict = torch.load(species_classifier_path, map_location=DEVICE)
        model_species_classifier.load_state_dict(state_dict)
        model_species_classifier.to(DEVICE)
        model_species_classifier.eval()
        print("✓ Modelo clasificador de especies cargado exitosamente")

        # Cargar modelo clasificador cultivada/salvaje (PyTorch)
        print("Cargando modelo clasificador cultivada/salvaje (best_cautiva_salvaje)...")
        print(f"Ruta: {wild_cultivated_path}")
        model_wild_cultivated = ConvNeXt(in_chans=3, num_classes=2)
        state_dict = torch.load(wild_cultivated_path, map_location=DEVICE)
        model_wild_cultivated.load_state_dict(state_dict)
        model_wild_cultivated.to(DEVICE)
        model_wild_cultivated.eval()
        print("✓ Modelo clasificador cultivada/salvaje cargado exitosamente")

        return True

    except Exception as e:
        print(f"Error al cargar los modelos: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_model():
    """DEPRECATED: Mantiene la función antigua por compatibilidad"""
    return load_pytorch_models()

def preprocess_image_pytorch(image_path):
    """
    Preprocesa la imagen para los modelos PyTorch (ConvNeXt)

    Args:
        image_path: Ruta a la imagen o objeto BytesIO

    Returns:
        torch.Tensor con la imagen preprocesada (1, 3, 224, 224)
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

    # Normalizar con estadísticas de ImageNet
    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    img_array = img_array / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    # Convertir a tensor PyTorch: (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

    # Añadir dimensión de batch: (C, H, W) -> (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


def preprocess_image(image_path):
    """
    DEPRECATED: Preprocesa la imagen para el modelo MobileNet (TensorFlow)
    Mantenida por compatibilidad, pero ya no se usa.

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

def generate_gradcam(model, img_array, layer_name, class_index=None):
    """
    Genera un mapa de calor Grad-CAM para visualizar qué partes de la imagen influyeron en la decisión

    Args:
        model: Modelo Keras
        img_array: Imagen preprocesada (1, 224, 224, 3)
        layer_name: Nombre de la capa convolucional objetivo
        class_index: Índice de la clase a visualizar (None = clase predicha)

    Returns:
        numpy array con el mapa de calor superpuesto sobre la imagen original
    """
    # Crear modelo que devuelve las activaciones de la capa conv y las predicciones
    grad_model = keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # Calcular gradientes
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Si es un modelo con múltiples salidas, usar la primera
        if isinstance(predictions, list):
            predictions = predictions[0]

        # Si no se especifica clase, usar la predicha
        if class_index is None:
            class_index = tf.argmax(predictions[0])

        # Extraer probabilidad de la clase objetivo
        class_channel = predictions[:, class_index]

    # Calcular gradientes de la clase respecto a la salida de la capa conv
    grads = tape.gradient(class_channel, conv_outputs)

    # Pooling de gradientes (promedio global)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiplicar cada canal por su importancia y sumar
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()
    conv_outputs = conv_outputs.numpy()

    for i in range(len(pooled_grads)):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Crear el mapa de calor
    heatmap = np.mean(conv_outputs, axis=-1)

    # Normalizar el mapa de calor entre 0 y 1
    heatmap = np.maximum(heatmap, 0)  # ReLU
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap

def superimpose_gradcam(img_array, heatmap, alpha=0.4):
    """
    Superpone el mapa de calor Grad-CAM sobre la imagen original

    Args:
        img_array: Imagen original (224, 224, 3) normalizada [0, 1]
        heatmap: Mapa de calor Grad-CAM (7, 7) o tamaño de la capa conv
        alpha: Transparencia del mapa de calor (0-1)

    Returns:
        Imagen PIL con el mapa de calor superpuesto
    """
    # Redimensionar heatmap al tamaño de la imagen
    heatmap_resized = cv2.resize(heatmap, (224, 224))

    # Convertir heatmap a colormap
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # RGB, sin alpha

    # Desnormalizar imagen
    img = img_array[0] * 255.0
    img = img.astype(np.uint8)

    # Convertir a float para mezclar
    img_float = img.astype(np.float32) / 255.0

    # Superponer
    superimposed = heatmap_colored * alpha + img_float * (1 - alpha)
    superimposed = np.clip(superimposed * 255, 0, 255).astype(np.uint8)

    # Convertir a PIL Image
    return Image.fromarray(superimposed)

def get_last_conv_layer_name(model):
    """
    Encuentra la última capa convolucional del modelo

    Args:
        model: Modelo Keras

    Returns:
        Nombre de la última capa convolucional
    """
    for layer in reversed(model.layers):
        # Buscar capas Conv2D
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
        # También buscar en modelos anidados (como MobileNet)
        if hasattr(layer, 'layers'):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, keras.layers.Conv2D):
                    return sublayer.name

    # Si no se encuentra, devolver None
    return None

def predict_fish_pytorch(image_path, use_roi=False, generate_gradcam_images=False):
    """
    Realiza la predicción completa del pez usando tres modelos PyTorch en cascada

    Args:
        image_path: Ruta a la imagen o BytesIO
        use_roi: Si True, usa ROI detection para mejorar la predicción (DESACTIVADO por defecto)
        generate_gradcam_images: Si True, genera mapas de calor Grad-CAM (no implementado para PyTorch aún)

    Returns:
        dict con las predicciones y métricas de calidad
    """
    if model_fish_detector is None or model_species_classifier is None:
        raise Exception("Modelos PyTorch no cargados")

    if model_wild_cultivated is None:
        print("Advertencia: Modelo de clasificación cultivada/salvaje no cargado")

    # Analizar calidad de imagen
    quality_analysis = analyze_image_quality(image_path)

    # Preparar imagen para predicción
    image_to_process = image_path

    # Si ROI fue detectado y use_roi=True, usar ROI en lugar de imagen completa
    if use_roi and quality_analysis["roi_detected"]:
        roi_array = quality_analysis["roi_result"]["roi"]
        roi_pil = Image.fromarray(roi_array.astype('uint8'), 'RGB')
        roi_bytes = io.BytesIO()
        roi_pil.save(roi_bytes, format='JPEG')
        roi_bytes.seek(0)
        image_to_process = roi_bytes

    # Preprocesar imagen para PyTorch
    img_tensor = preprocess_image_pytorch(image_to_process).to(DEVICE)

    # === PASO 1: Detectar si hay pez ===
    with torch.no_grad():
        fish_logits = model_fish_detector(img_tensor)
        fish_probs = F.softmax(fish_logits, dim=1)[0]  # [prob_no_pez, prob_pez]
        fish_pred = torch.argmax(fish_probs).item()
        fish_confidence = float(fish_probs[fish_pred])

    is_fish = fish_pred == 1  # 1 = Pez, 0 = No Pez

    # Si no se detectó pez, retornar resultado negativo
    if not is_fish:
        return {
            "success": True,
            "is_fish": False,
            "fish_confidence": float(fish_probs[0]),  # Confianza de "No Pez"
            "species": "No detectado",
            "species_id": -1,
            "species_confidence": 0.0,
            "species_probabilities": {
                "dorada": 0.0,
                "lubina": 0.0,
                "otro": 0.0
            },
            "classification": "No aplica",
            "classification_id": -1,
            "classification_confidence": 0.0,
            "probabilities": {
                "cultivada": 0.0,
                "salvaje": 0.0
            },
            "summary": "No se detectó un pez",
            "low_confidence": True,
            "warnings": [{
                "type": "no_fish_detected",
                "message": "No se detectó un pez en la imagen",
                "detail": f"Confianza de detección: {float(fish_probs[1])*100:.1f}%"
            }],
            "quality_analysis": {
                "quality_score": quality_analysis["quality_score"],
                "is_good_quality": quality_analysis["is_good_quality"],
                "quality_issues": quality_analysis["quality_issues"],
                "suggestions": quality_analysis["suggestions"],
                "roi_detected": quality_analysis["roi_detected"],
                "roi_bbox": quality_analysis["roi_result"]["bbox"] if quality_analysis["roi_detected"] else None
            }
        }

    # === PASO 2: Clasificar especie (Dorada vs Lubina) ===
    with torch.no_grad():
        species_logits = model_species_classifier(img_tensor)
        species_probs = F.softmax(species_logits, dim=1)[0]  # [prob_dorada, prob_lubina]
        species_id = torch.argmax(species_probs).item()
        species_confidence = float(species_probs[species_id])

    # === PASO 3: Clasificar cultivada vs salvaje (PyTorch) ===
    classification_confidence = 0.0
    classification_id = 0
    classification_name = "No disponible"
    cultivada_prob = 0.0
    salvaje_prob = 0.0

    if model_wild_cultivated is not None:
        # Usar la misma imagen preprocesada para PyTorch
        with torch.no_grad():
            classification_logits = model_wild_cultivated(img_tensor)
            classification_probs = F.softmax(classification_logits, dim=1)[0]  # [prob_cultivada, prob_salvaje]
            classification_id = torch.argmax(classification_probs).item()
            classification_confidence = float(classification_probs[classification_id])
            classification_name = CLASS_NAMES[classification_id]

            cultivada_prob = float(classification_probs[0])
            salvaje_prob = float(classification_probs[1])

    # Verificar si la confianza es baja
    low_confidence = (species_confidence < CONFIDENCE_THRESHOLD or
                     fish_confidence < CONFIDENCE_THRESHOLD or
                     (model_wild_cultivated is not None and classification_confidence < CONFIDENCE_THRESHOLD))

    # Generar advertencias
    warnings = []
    if low_confidence:
        detail_parts = [f"Confianza de especie: {species_confidence*100:.1f}%",
                       f"Confianza de detección: {fish_confidence*100:.1f}%"]
        if model_wild_cultivated is not None:
            detail_parts.append(f"Confianza de clasificación: {classification_confidence*100:.1f}%")

        warnings.append({
            "type": "low_confidence",
            "message": "Confianza baja en la predicción. Por favor, repite la foto.",
            "detail": ", ".join(detail_parts)
        })

    # Construir resultado
    result = {
        "success": True,
        "is_fish": True,
        "fish_confidence": fish_confidence,
        "species": SPECIES_NAMES_PYTORCH[species_id],
        "species_id": species_id,
        "species_confidence": species_confidence,
        "species_probabilities": {
            "dorada": float(species_probs[0]),
            "lubina": float(species_probs[1]),
            "otro": 0.0  # No disponible en modelo PyTorch
        },
        "classification": classification_name,
        "classification_id": classification_id,
        "classification_confidence": classification_confidence,
        "probabilities": {
            "cultivada": cultivada_prob,
            "salvaje": salvaje_prob
        },
        "summary": f"{SPECIES_NAMES_PYTORCH[species_id]} - {classification_name}",
        "low_confidence": low_confidence,
        "warnings": warnings,
        "quality_analysis": {
            "quality_score": quality_analysis["quality_score"],
            "is_good_quality": quality_analysis["is_good_quality"],
            "quality_issues": quality_analysis["quality_issues"],
            "suggestions": quality_analysis["suggestions"],
            "roi_detected": quality_analysis["roi_detected"],
            "roi_bbox": quality_analysis["roi_result"]["bbox"] if quality_analysis["roi_detected"] else None
        }
    }

    # TODO: Implementar Grad-CAM para PyTorch si se solicita
    if generate_gradcam_images:
        result["gradcam"] = {
            "error": "Grad-CAM para PyTorch no implementado aún"
        }

    return result


def predict_fish(image_path, use_roi=False, generate_gradcam_images=False):
    """
    Función wrapper que usa los modelos PyTorch por defecto

    Args:
        image_path: Ruta a la imagen o BytesIO
        use_roi: Si True, usa ROI detection para mejorar la predicción
        generate_gradcam_images: Si True, genera mapas de calor Grad-CAM

    Returns:
        dict con las predicciones y métricas de calidad
    """
    return predict_fish_pytorch(image_path, use_roi, generate_gradcam_images)

# Cargar el modelo al iniciar
with app.app_context():
    load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model_fish_detector is None or model_species_classifier is None or model_wild_cultivated is None:
        return jsonify({"error": "Modelos no cargados correctamente."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No se encontró la imagen en la solicitud."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No se seleccionó ningún archivo."}), 400

    try:
        # Verificar si se solicita Grad-CAM
        enable_gradcam = request.form.get('enable_gradcam', 'false').lower() == 'true'

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
        result = predict_fish(image_path, generate_gradcam_images=enable_gradcam)

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
    if model_fish_detector is None or model_species_classifier is None or model_wild_cultivated is None:
        return jsonify({"error": "Modelos no cargados correctamente."}), 500

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
    all_models_ready = (model_fish_detector is not None and
                       model_species_classifier is not None and
                       model_wild_cultivated is not None)

    return jsonify({
        "status": "ok" if all_models_ready else "partial",
        "fish_detector_loaded": model_fish_detector is not None,
        "species_classifier_loaded": model_species_classifier is not None,
        "wild_cultivated_classifier_loaded": model_wild_cultivated is not None,
        "models_ready": all_models_ready,
        "model_types": {
            "fish_detector": "ConvNeXt-Tiny (PyTorch)",
            "species_classifier": "ConvNeXt-Tiny (PyTorch)",
            "wild_cultivated_classifier": "ConvNeXt-Tiny (PyTorch)"
        },
        "pipeline": "Cascada: Detector -> Clasificador Especies -> Clasificador Cultivada/Salvaje"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
