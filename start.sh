#!/bin/bash

# Script de inicio rápido para Gloria Mobile

echo "================================================"
echo "  Gloria Mobile - Clasificador de Peces 🐟"
echo "================================================"
echo ""

# Verificar si Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker no está instalado"
    echo "   Por favor instala Docker desde: https://docs.docker.com/get-docker/"
    exit 1
fi

# Verificar si Docker Compose está instalado
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: Docker Compose no está instalado"
    echo "   Por favor instala Docker Compose desde: https://docs.docker.com/compose/install/"
    exit 1
fi

# Verificar si Docker daemon está ejecutándose
if ! docker info &> /dev/null; then
    echo "❌ Error: Docker daemon no está ejecutándose"
    echo "   Por favor inicia Docker y vuelve a intentar"
    exit 1
fi

echo "✅ Docker está instalado y ejecutándose"
echo ""

# Preguntar si quiere construir o solo iniciar
echo "¿Qué deseas hacer?"
echo "1) Construir e iniciar (primera vez o después de cambios)"
echo "2) Iniciar (si ya construiste antes)"
echo "3) Detener la aplicación"
echo "4) Ver logs"
echo "5) Limpiar todo (eliminar contenedores e imágenes)"
echo ""
read -p "Selecciona una opción (1-5): " option

case $option in
    1)
        echo ""
        echo "🔨 Construyendo imagen Docker..."
        echo "   (Esto puede tardar 10-15 minutos la primera vez)"
        echo ""
        docker-compose up --build
        ;;
    2)
        echo ""
        echo "🚀 Iniciando aplicación..."
        echo ""
        docker-compose up
        ;;
    3)
        echo ""
        echo "⏹️  Deteniendo aplicación..."
        docker-compose down
        echo "✅ Aplicación detenida"
        ;;
    4)
        echo ""
        echo "📋 Mostrando logs (Ctrl+C para salir)..."
        echo ""
        docker-compose logs -f
        ;;
    5)
        echo ""
        echo "🗑️  Limpiando todo..."
        docker-compose down -v --rmi all
        echo "✅ Limpieza completada"
        ;;
    *)
        echo "❌ Opción inválida"
        exit 1
        ;;
esac

if [ $option -eq 1 ] || [ $option -eq 2 ]; then
    echo ""
    echo "================================================"
    echo "  ✅ Aplicación ejecutándose en:"
    echo "     http://localhost:5000"
    echo "================================================"
    echo ""
    echo "Presiona Ctrl+C para detener"
fi
