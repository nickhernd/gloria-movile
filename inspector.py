import tensorflow as tf
import sys

if len(sys.argv) < 2:
    print("Uso: python inspector.py <ruta_al_modelo.tflite>")
    sys.exit(1)

model_path = sys.argv[1]

try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    print(f"\n--- Detalles del Modelo TFLite: {model_path} ---\n")

    # Detalles de las entradas
    input_details = interpreter.get_input_details()
    print("Entradas:")
    for i, detail in enumerate(input_details):
        print(f"  Entrada {i}:")
        print(f"    Nombre: {detail['name']}")
        print(f"    Forma (Shape): {detail['shape']}")
        print(f"    Tipo de Dato (Dtype): {detail['dtype']}")
        print(f"    Índice de Tensor: {detail['index']}")
        print(f"    Cuantización: {detail.get('quantization', 'N/A')}")

    # Detalles de las salidas
    output_details = interpreter.get_output_details()
    print("\nSalidas:")
    for i, detail in enumerate(output_details):
        print(f"  Salida {i}:")
        print(f"    Nombre: {detail['name']}")
        print(f"    Forma (Shape): {detail['shape']}")
        print(f"    Tipo de Dato (Dtype): {detail['dtype']}")
        print(f"    Índice de Tensor: {detail['index']}")
        print(f"    Cuantización: {detail.get('quantization', 'N/A')}")

    print("\n--- Fin de los Detalles ---\n")

except Exception as e:
    print(f"Error al cargar o inspeccionar el modelo: {e}")
    sys.exit(1)
