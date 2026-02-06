import tensorflow as tf
from tensorflow import keras
import numpy as np

# Cargar el modelo
model_path = 'web_app/student_mobilenet_little.h5'
print(f"Cargando modelo desde: {model_path}")

model = keras.models.load_model(model_path)

print("\n" + "="*60)
print("RESUMEN DEL MODELO")
print("="*60)
model.summary()

print("\n" + "="*60)
print("INFORMACIÓN DE ENTRADA")
print("="*60)
print(f"Shape de entrada: {model.input_shape}")
print(f"Dtype de entrada: {model.input.dtype}")

print("\n" + "="*60)
print("INFORMACIÓN DE SALIDA")
print("="*60)
print(f"Shape de salida: {model.output_shape}")
print(f"Número de clases: {model.output_shape[-1]}")

print("\n" + "="*60)
print("CAPAS DEL MODELO")
print("="*60)
for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name} - {layer.__class__.__name__}")
    if hasattr(layer, 'output_shape'):
        print(f"   Output shape: {layer.output_shape}")

print("\n" + "="*60)
print("ÚLTIMA CAPA (Clasificación)")
print("="*60)
last_layer = model.layers[-1]
print(f"Tipo: {last_layer.__class__.__name__}")
print(f"Activación: {last_layer.activation.__name__ if hasattr(last_layer, 'activation') else 'N/A'}")
print(f"Unidades: {last_layer.units if hasattr(last_layer, 'units') else 'N/A'}")
