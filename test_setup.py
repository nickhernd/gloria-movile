#!/usr/bin/env python3
"""
Script de prueba para verificar la configuración del sistema
"""

import sys

def test_imports():
    """Verifica que todas las dependencias estén instaladas"""
    print("=== Verificando importaciones ===")

    required_modules = [
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS'),
        ('torch', 'PyTorch'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
    ]

    all_ok = True
    for module, name in required_modules:
        try:
            __import__(module)
            print(f"✓ {name} instalado correctamente")
        except ImportError:
            print(f"✗ {name} NO está instalado")
            all_ok = False

    # Verificar CLIP por separado (instalación especial)
    try:
        import clip
        print(f"✓ CLIP instalado correctamente")
        print(f"  Modelos disponibles: {clip.available_models()}")
    except ImportError:
        print(f"✗ CLIP NO está instalado")
        print(f"  Instalar con: pip install git+https://github.com/openai/CLIP.git")
        all_ok = False

    return all_ok

def test_cuda():
    """Verifica si CUDA está disponible"""
    print("\n=== Verificando dispositivos ===")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA disponible")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Dispositivo: cuda")
        else:
            print(f"⚠ CUDA no disponible, usando CPU")
            print(f"  Dispositivo: cpu")
            print(f"  (La inferencia será más lenta pero funcional)")
    except Exception as e:
        print(f"✗ Error al verificar CUDA: {e}")

def test_clip_model():
    """Intenta cargar el modelo CLIP"""
    print("\n=== Probando carga del modelo CLIP ===")
    try:
        import torch
        import clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Cargando modelo CLIP ViT-B/32 en {device}...")

        model, preprocess = clip.load('ViT-B/32', device)
        print(f"✓ Modelo CLIP cargado exitosamente")

        # Verificar tokenización
        text = clip.tokenize(["a test phrase"])
        print(f"✓ Tokenización funcionando correctamente")

        return True
    except Exception as e:
        print(f"✗ Error al cargar CLIP: {e}")
        return False

def test_file_structure():
    """Verifica que la estructura de archivos sea correcta"""
    print("\n=== Verificando estructura de archivos ===")
    import os

    required_files = [
        'web_app/app.py',
        'web_app/templates/index.html',
        'web_app/static/script.js',
        'web_app/static/style.css',
        'requirements.txt',
    ]

    all_ok = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} existe")
        else:
            print(f"✗ {file_path} NO existe")
            all_ok = False

    # Verificar carpeta de uploads
    if os.path.exists('web_app/uploads'):
        print(f"✓ web_app/uploads existe")
    else:
        print(f"⚠ web_app/uploads no existe (se creará automáticamente)")

    return all_ok

def main():
    print("=" * 60)
    print("VERIFICACIÓN DE CONFIGURACIÓN - Gloria Mobile")
    print("=" * 60)

    results = []

    # Test 1: Importaciones
    results.append(("Dependencias", test_imports()))

    # Test 2: CUDA
    test_cuda()

    # Test 3: Modelo CLIP
    results.append(("Modelo CLIP", test_clip_model()))

    # Test 4: Estructura de archivos
    results.append(("Estructura de archivos", test_file_structure()))

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)

    all_passed = all(result for _, result in results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    print("=" * 60)

    if all_passed:
        print("\n✓ ¡Todo listo! Puedes ejecutar la aplicación con:")
        print("  cd web_app")
        print("  python app.py")
        return 0
    else:
        print("\n✗ Hay problemas que deben resolverse antes de ejecutar la aplicación.")
        print("  Revisa los mensajes de error arriba e instala las dependencias faltantes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
