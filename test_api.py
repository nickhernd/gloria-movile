#!/usr/bin/env python3
"""
Script para probar el modelo .h5 a través de la API Flask
"""
import requests
import numpy as np
from PIL import Image
import io
import sys

print("="*70)
print("PRUEBA DE LA API FLASK CON EL MODELO .h5")
print("="*70)

# URL del servidor
BASE_URL = "http://localhost:5000"

# 1. Verificar que el servidor está corriendo
print("\n[1/4] Verificando servidor...")
try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    health = response.json()
    print(f"✅ Servidor activo")
    print(f"   - Estado: {health['status']}")
    print(f"   - Modelo cargado: {health['model_loaded']}")
    print(f"   - Tipo de modelo: {health['model_type']}")
except requests.exceptions.ConnectionError:
    print("❌ Error: El servidor no está corriendo")
    print("   Ejecuta: cd web_app && python3 app.py")
    sys.exit(1)

# 2. Crear imagen de prueba
print("\n[2/4] Creando imagen de prueba...")
# Crear una imagen con colores que simulen un pez
img = Image.new('RGB', (224, 224))
pixels = img.load()
for i in range(224):
    for j in range(224):
        # Gradiente de azul a gris (simula agua y pez)
        r = int(100 + (i/224) * 100)
        g = int(120 + (j/224) * 80)
        b = int(150 + ((i+j)/(224*2)) * 50)
        pixels[i, j] = (r, g, b)

# Guardar en buffer
buffer = io.BytesIO()
img.save(buffer, format='PNG')
buffer.seek(0)
print(f"✅ Imagen creada: 224x224 RGB")

# 3. Enviar al endpoint /predict
print("\n[3/4] Enviando imagen al endpoint /predict...")
files = {'image': ('test_fish.png', buffer, 'image/png')}
try:
    response = requests.post(f"{BASE_URL}/predict", files=files, timeout=10)

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Predicción exitosa")
        print(f"\n   📊 RESULTADOS:")
        print(f"      🐟 Especie: {result['species']}")
        print(f"         Confianza: {result['species_confidence']*100:.2f}%")
        print(f"         Probabilidades:")
        print(f"            - Dorada: {result['species_probabilities']['dorada']*100:.2f}%")
        print(f"            - Lubina: {result['species_probabilities']['lubina']*100:.2f}%")
        print(f"\n      🌊 Clasificación: {result['classification']}")
        print(f"         Confianza: {result['classification_confidence']*100:.2f}%")
        print(f"         Probabilidades:")
        print(f"            - Cultivada: {result['probabilities']['cultivada']*100:.2f}%")
        print(f"            - Salvaje: {result['probabilities']['salvaje']*100:.2f}%")
        print(f"\n      🎯 Detección de pez: {result['fish_confidence']*100:.2f}%")
        print(f"      📝 Resumen: {result['summary']}")
    else:
        print(f"⚠️  Respuesta con código: {response.status_code}")
        print(f"   Mensaje: {response.json()}")

except requests.exceptions.Timeout:
    print("❌ Timeout: El servidor tardó demasiado en responder")
except Exception as e:
    print(f"❌ Error: {e}")

# 4. Probar endpoint de tiempo real
print("\n[4/4] Probando endpoint /predict_realtime...")
buffer.seek(0)
files = {'image': ('test_fish.png', buffer, 'image/png')}
try:
    response = requests.post(f"{BASE_URL}/predict_realtime", files=files, timeout=10)

    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            print(f"✅ Endpoint de tiempo real funciona correctamente")
            print(f"   Especie: {result['species']} ({result['species_confidence']*100:.1f}%)")
            print(f"   Clasificación: {result['classification']} ({result['classification_confidence']*100:.1f}%)")
        else:
            print(f"⚠️  No se detectó pez (esperado para imagen de prueba)")
    else:
        print(f"⚠️  Código de respuesta: {response.status_code}")

except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*70)
print("✅ PRUEBAS COMPLETADAS")
print("="*70)
