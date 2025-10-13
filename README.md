# Clasificador de Peces - Gloria Mobile

Aplicación web inteligente que detecta automáticamente el tipo de pez (Dorada o Lubina) y clasifica si es **cultivado** o **salvaje** usando el modelo CLIP de OpenAI con visión por computadora.

## ✨ Características

- 🤖 **Detección automática de especie**: Identifica si es Dorada o Lubina sin necesidad de especificar
- 🔍 **Validación inteligente**: Verifica que haya un pez en la imagen antes de clasificar
- 📊 **Clasificación dual**: Determina si el pez es cultivado o salvaje
- 📱 **Interfaz moderna**: Diseño limpio y responsivo con gradientes y animaciones
- 📸 **Captura desde cámara**: Toma fotos directamente desde el navegador
- 📁 **Carga de archivos**: Sube imágenes desde tu dispositivo
- 🎯 **Alta precisión**: Niveles de confianza para cada predicción

## Especies Soportadas

- **Dorada** (Sparus aurata / S_AURATA)
- **Lubina** (Dicentrarchus labrax / D_LABRAX)

## Clases de Clasificación

- **Cultivada (Farmed)**: Peces de acuicultura
- **Salvaje (Wild)**: Peces capturados en estado natural

## Tecnologías

- **Backend**: Flask + PyTorch + CLIP (ViT-B/32)
- **Frontend**: HTML5 + Bootstrap 5 + JavaScript
- **Modelo**: OpenAI CLIP para clasificación de imagen-texto

## Instalación

### Opción 1: Docker (Recomendado) 🐳

La forma más rápida y sencilla de ejecutar la aplicación:

#### Método rápido (con script):

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd gloria-movile

# 2. Ejecutar script de inicio
./start.sh        # Linux/Mac
start.bat         # Windows
```

#### Método manual:

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd gloria-movile

# 2. Construir y ejecutar con docker-compose
docker-compose up --build
```

La aplicación estará disponible en `http://localhost:5000`

**Comandos útiles de Docker:**

```bash
# Ejecutar en segundo plano (detached)
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener la aplicación
docker-compose down

# Reconstruir imagen después de cambios
docker-compose up --build

# Limpiar todo (incluye volúmenes)
docker-compose down -v
```

**Ventajas de Docker:**
- ✅ No requiere instalar dependencias manualmente
- ✅ Ambiente aislado y reproducible
- ✅ Funciona igual en cualquier sistema operativo
- ✅ Fácil de desplegar en servidores

---

### Opción 2: Instalación Manual

#### 1. Clonar el repositorio

```bash
git clone <url-del-repo>
cd gloria-movile
```

#### 2. Crear entorno virtual (recomendado)

```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

#### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

**Nota**: La instalación de CLIP puede tardar varios minutos.

#### 4. Verificar instalación

```bash
cd web_app
python app.py
```

El servidor debería iniciar en `http://localhost:5000`

## Uso

### Iniciar la aplicación

**Con Docker:**
```bash
docker-compose up
```

**Sin Docker:**
```bash
cd web_app
python app.py
```

### Interfaz Web

1. Abre tu navegador en `http://localhost:5000`
2. **Carga una imagen** o **toma una foto** de un pez
3. Haz clic en **"Analizar Pez"**
4. El sistema automáticamente:
   - ✅ Valida que hay un pez en la imagen
   - 🐟 Detecta si es Dorada o Lubina
   - 🏷️ Clasifica si es cultivado o salvaje
5. Observa los resultados con:
   - Especie detectada
   - Clasificación (Cultivado/Salvaje)
   - Niveles de confianza
   - Probabilidades detalladas

### API REST

**Endpoint**: `POST /predict`

**Parámetros**:
- `image`: Archivo de imagen (form-data) - ¡Solo envía la imagen!

**Ejemplo con curl**:

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/fish.jpg"
```

**Respuesta exitosa**:

```json
{
  "success": true,
  "is_fish": true,
  "fish_confidence": 0.95,
  "species": "Dorada",
  "species_id": 0,
  "species_confidence": 0.87,
  "classification": "Salvaje",
  "classification_id": 1,
  "classification_confidence": 0.82,
  "probabilities": {
    "cultivada": 0.18,
    "salvaje": 0.82
  },
  "filename": "20251012-153045-fish.jpg",
  "summary": "Dorada Salvaje"
}
```

**Respuesta con error (sin pez detectado)**:

```json
{
  "error": "No se detectó un pez en la imagen",
  "is_fish": false,
  "fish_confidence": 0.35,
  "message": "Por favor, toma una foto donde aparezca claramente un pez"
}
```

**Health Check**: `GET /health`

```bash
curl http://localhost:5000/health
```

## Estructura del Proyecto

```
gloria-movile/
├── model/
│   └── CLIP_Image&Text.ipynb    # Notebook de investigación con CLIP
├── web_app/
│   ├── static/
│   │   ├── script.js            # Lógica del frontend
│   │   └── style.css            # Estilos
│   ├── templates/
│   │   └── index.html           # Interfaz web
│   ├── uploads/                 # Imágenes subidas
│   └── app.py                   # Backend Flask con CLIP
├── Dockerfile                   # Configuración de imagen Docker
├── docker-compose.yml           # Orquestación de contenedores
├── .dockerignore                # Archivos ignorados por Docker
├── start.sh                     # Script de inicio (Linux/Mac)
├── start.bat                    # Script de inicio (Windows)
├── inspector.py                 # Utilidad para inspeccionar modelos TFLite
├── test_setup.py                # Script de verificación del sistema
├── requirements.txt             # Dependencias Python
└── README.md                    # Este archivo
```

## Cómo Funciona

### Proceso de Análisis en 3 Pasos

La aplicación usa **CLIP (Contrastive Language-Image Pre-training)** de OpenAI con un pipeline de clasificación en tres etapas:

#### 1️⃣ Validación de Imagen
- Verifica que la imagen contenga un pez
- Calcula confianza de detección
- Rechaza imágenes sin peces (>50% confianza requerida)

#### 2️⃣ Detección de Especie
- Identifica automáticamente si es **Dorada** o **Lubina**
- Usa prompts específicos para cada especie
- Selecciona la especie con mayor confianza

#### 3️⃣ Clasificación Cultivado/Salvaje
- Aplica los text prompts específicos de la especie detectada
- Clasifica como **Cultivada** o **Salvaje**
- Genera probabilidades para cada clase

### Tecnología CLIP

CLIP usa embeddings multimodales para relacionar imágenes con texto:

1. **Carga el modelo**: CLIP ViT-B/32 preentrenado
2. **Embeddings de texto**: Convierte descripciones en vectores
3. **Embeddings de imagen**: Convierte la imagen en un vector
4. **Similaridad coseno**: Calcula similitud imagen-texto
5. **Softmax**: Genera probabilidades normalizadas

### Text Prompts

**Dorada (Cultivada)**:
> "a close-up of a fish with a uniform gray color that transitions to a white ventral side, an oval body and the mouth is slightly tilted upwards, the lateral fins are close to the body and small"

**Dorada (Salvaje)**:
> "a close-up of a fish with a straight body, the color transitions from a darker upper side to a white ventral side and in which the mouth follows direction of the main axis of the body, the dorsal fin is symmetrical"

**Lubina (Cultivada)**:
> "a close-up of a dark grey fish with a curved bottom, the lateral fins are close to the body and small"

**Lubina (Salvaje)**:
> "a close-up of a fish with a flat bottom and in mouth follows the direction of the main axis of the body, the lateral fins separated from the body and oriented slightly backward and the dorsal fin is symmetrical"

## Requisitos del Sistema

### Con Docker:
- Docker 20.10+
- Docker Compose 1.29+
- 4GB RAM mínimo (8GB recomendado)
- 5GB de espacio en disco

### Sin Docker (instalación manual):
- Python 3.8+
- 2GB RAM mínimo (4GB recomendado)
- GPU opcional (CUDA) para inferencia más rápida
- 3GB de espacio en disco

## Solución de Problemas

### Docker

**Error: "Cannot connect to Docker daemon"**

Asegúrate de que Docker esté ejecutándose:

```bash
# Linux/Mac
sudo systemctl start docker

# Windows: Inicia Docker Desktop
```

**Build muy lento**

La primera vez puede tardar 10-15 minutos descargando PyTorch y CLIP. Posteriores builds serán mucho más rápidos gracias a la caché de Docker.

**Puerto 5000 ocupado**

Cambia el puerto en `docker-compose.yml`:

```yaml
ports:
  - "8080:5000"  # Usar puerto 8080 en lugar de 5000
```

### Instalación Manual

**Error: "Modelo no cargado"**

Verifica que CLIP se haya instalado correctamente:

```bash
python -c "import clip; print(clip.available_models())"
```

**Instalación lenta de dependencias**

PyTorch puede tardar en instalarse. Considera usar versiones precompiladas:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### General

**La cámara no funciona**

- Asegúrate de dar permisos de cámara en tu navegador
- Usa HTTPS o localhost (las APIs de cámara requieren contexto seguro)

## Licencia

Este proyecto es parte de la investigación de clasificación de peces.

## Contacto

Para preguntas o problemas, abre un issue en el repositorio.
