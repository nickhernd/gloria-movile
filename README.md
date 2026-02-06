# GLORIA - Sistema de Identificación de Especies y Origen de Peces

Este proyecto implementa una solución tecnológica avanzada basada en Inteligencia Artificial para la identificación automática de especies de peces y la determinación de su origen (Salvaje o de Piscifactoría). Desarrollado en el marco del proyecto GLORIA, su objetivo es mejorar la resiliencia y sostenibilidad del sector acuícola mediante el uso de herramientas de visión artificial.

## Descripción del Proyecto

El sistema está diseñado para operar en entornos reales como pescaderías, lonjas y mercados, superando las limitaciones de los entornos controlados de laboratorio. Utiliza algoritmos de Aprendizaje Profundo (Deep Learning) para analizar imágenes de peces y proporcionar información crítica sobre su taxonomía y procedencia comercial.

### Funcionalidades Principales
- **Identificación de Especies:** Clasificación precisa entre Dorada (Sparus aurata) y Lubina (Dicentrarchus labrax).
- **Discriminación de Origen:** Determinación del método de obtención del ejemplar (captura salvaje o producción en piscifactoría).
- **Análisis de Calidad de Imagen:** Evaluación en tiempo real de parámetros de iluminación, enfoque y encuadre para garantizar la fiabilidad de la predicción.
- **Adaptación de Dominio:** Modelos optimizados para el reconocimiento de especímenes en fondos complejos y bajo condiciones lumínicas variables.

## Requisitos Técnicos

- **Lenguaje de Programación:** Python 3.9 o superior.
- **Framework Web:** Flask.
- **Bibliotecas de IA/ML:** PyTorch (arquitecturas ViT y ConvNeXt) y TensorFlow (arquitectura MobileNet).
- **Procesamiento de Imágenes:** OpenCV y scikit-image.

## Instalación y Configuración Local

### 1. Preparación del Entorno
Se recomienda el uso de un entorno virtual para asegurar el aislamiento de las dependencias.

```bash
# Creación del entorno virtual
python -m venv .venv

# Activación del entorno
# En sistemas Linux/macOS:
source .venv/bin/activate
# En sistemas Windows:
.venv\Scripts\activate
```

### 2. Instalación de Dependencias
Ejecute el siguiente comando para instalar los paquetes necesarios:

```bash
pip install -r requirements.txt
```

### 3. Ejecución de la Aplicación
Para iniciar el servidor de aplicaciones en el entorno local:

```bash
python web_app/app.py
```

Una vez iniciado, la interfaz será accesible a través de la dirección: http://localhost:5000

## Arquitectura del Sistema

El proceso de análisis se divide en las siguientes etapas secuenciales:
1. **Preprocesamiento:** Normalización de la imagen y validación de métricas de calidad.
2. **Detección de Región de Interés (ROI):** Localización del espécimen dentro del cuadro para optimizar el análisis.
3. **Clasificación Taxonómica:** Un modelo basado en Vision Transformer (ViT) identifica la especie.
4. **Clasificación de Origen:** Selección dinámica de un modelo especializado según la especie detectada para determinar si es Salvaje o de Cultivo.
5. **Generación de Resultados:** Presentación de la identificación junto con sus correspondientes índices de confianza estadística.

## Estructura del Repositorio

- `web_app/`: Contiene el código fuente de la aplicación y la lógica de inferencia.
- `model/`: Almacena los pesos y las configuraciones de los modelos entrenados.
- `uploads/`: Directorio destinado al almacenamiento temporal de imágenes procesadas.

---
**Proyecto GLORIA** - Desarrollo de herramientas tecnológicas para la sostenibilidad marina.