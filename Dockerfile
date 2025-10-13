# Usar imagen base de Python
FROM python:3.10-slim

# Instalar dependencias del sistema necesarias para PyTorch y Pillow
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements primero para aprovechar caché de Docker
COPY requirements.txt .

# Instalar dependencias de Python
# Instalamos PyTorch CPU-only para reducir tamaño de imagen
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir flask flask-cors pillow numpy werkzeug && \
    pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# Copiar el código de la aplicación
COPY web_app/ /app/web_app/
COPY model/ /app/model/

# Crear directorio para uploads
RUN mkdir -p /app/web_app/uploads

# Exponer puerto
EXPOSE 5000

# Cambiar al directorio de la aplicación
WORKDIR /app/web_app

# Variables de entorno
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
