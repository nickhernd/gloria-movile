# Usar imagen base de Python
FROM python:3.10-slim

# Instalar dependencias del sistema necesarias para PyTorch, Pillow y OpenCV
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements primero para aprovechar caché de Docker
COPY requirements.txt .

# Instalar dependencias de Python desde requirements.txt
RUN pip install --no-cache-dir --timeout=600 -r requirements.txt

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
