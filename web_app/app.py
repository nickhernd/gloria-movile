from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app) # Habilitar CORS para permitir solicitudes desde el frontend

# Ruta al modelo TFLite
MODEL_PATH = '2-modelo_pez_vgg.tflite'
IMAGE_SIZE = 224

interpreter = None
input_details = None
output_details = None

def load_model():
    global interpreter, input_details, output_details
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"Modelo TFLite cargado exitosamente desde {MODEL_PATH}")
        print(f"Detalles de entrada: {input_details}")
        print(f"Detalles de salida: {output_details}")
    except Exception as e:
        print(f"Error al cargar el modelo TFLite: {e}")
        interpreter = None

# Cargar el modelo al iniciar la aplicación Flask
with app.app_context():
    load_model()

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return jsonify({"error": "Modelo no cargado."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No se encontró la imagen en la solicitud."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No se seleccionó ningún archivo."}), 400

    try:
        # Leer la imagen
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB') # Asegurarse de que sea RGB
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE)) # Redimensionar

        # Convertir a array de numpy y normalizar
        input_data = np.array(img, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0) # Añadir dimensión de batch

        # Establecer el tensor de entrada y realizar la inferencia
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Obtener los resultados
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions = output_data[0].tolist() # Convertir a lista de Python

        return jsonify({"predictions": predictions})

    except Exception as e:
        print(f"Error durante la predicción: {e}")
        return jsonify({"error": f"Error durante la predicción: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
