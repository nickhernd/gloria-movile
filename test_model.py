import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Cargar modelo
model = keras.models.load_model('web_app/student_mobilenet_little.h5')

# Crear una imagen de prueba (224x224x3)
dummy_image = np.random.rand(1, 224, 224, 3).astype(np.float32)

print("Forma de entrada:", dummy_image.shape)
print("\nRealizando predicción de prueba...")

# Hacer predicción
predictions = model.predict(dummy_image, verbose=0)

print("\n" + "="*60)
print("RESULTADOS DE LA PREDICCIÓN")
print("="*60)

if isinstance(predictions, list):
    print(f"El modelo devuelve {len(predictions)} salidas:")
    for i, pred in enumerate(predictions):
        print(f"\nSalida {i+1}:")
        print(f"  Forma: {pred.shape}")
        print(f"  Valores: {pred}")
        print(f"  Suma (debería ser ~1.0 si es softmax): {pred.sum()}")
        print(f"  Clase predicha: {np.argmax(pred)}")
        print(f"  Confianza: {pred.max():.4f}")
else:
    print("El modelo devuelve una única salida:")
    print(f"  Forma: {predictions.shape}")
    print(f"  Valores: {predictions}")

print("\n" + "="*60)
print("INTERPRETACIÓN")
print("="*60)
print("Salida 1 probablemente: Especie (índice 0 = Dorada, índice 1 = Lubina)")
print("Salida 2 probablemente: Clasificación (índice 0 = Cultivada, índice 1 = Salvaje)")
