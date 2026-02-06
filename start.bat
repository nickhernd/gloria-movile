@echo off
REM Script de inicio rápido para Gloria Mobile (Windows)

echo ================================================
echo   Gloria Mobile - Clasificador de Peces
echo ================================================
echo.

REM Verificar si Docker está instalado
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker no esta instalado
    echo Por favor instala Docker Desktop desde: https://docs.docker.com/desktop/windows/install/
    pause
    exit /b 1
)

REM Verificar si Docker Compose está instalado
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker Compose no esta instalado
    echo Por favor instala Docker Desktop que incluye Docker Compose
    pause
    exit /b 1
)

REM Verificar si Docker daemon está ejecutándose
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker daemon no esta ejecutandose
    echo Por favor inicia Docker Desktop y vuelve a intentar
    pause
    exit /b 1
)

echo Docker esta instalado y ejecutandose
echo.

REM Menu de opciones
echo Que deseas hacer?
echo 1) Construir e iniciar (primera vez o despues de cambios)
echo 2) Iniciar (si ya construiste antes)
echo 3) Detener la aplicacion
echo 4) Ver logs
echo 5) Limpiar todo (eliminar contenedores e imagenes)
echo.

set /p option="Selecciona una opcion (1-5): "

if "%option%"=="1" (
    echo.
    echo Construyendo imagen Docker...
    echo (Esto puede tardar 10-15 minutos la primera vez)
    echo.
    docker-compose up --build
    goto :end
)

if "%option%"=="2" (
    echo.
    echo Iniciando aplicacion...
    echo.
    docker-compose up
    goto :end
)

if "%option%"=="3" (
    echo.
    echo Deteniendo aplicacion...
    docker-compose down
    echo Aplicacion detenida
    pause
    exit /b 0
)

if "%option%"=="4" (
    echo.
    echo Mostrando logs (Ctrl+C para salir)...
    echo.
    docker-compose logs -f
    goto :end
)

if "%option%"=="5" (
    echo.
    echo Limpiando todo...
    docker-compose down -v --rmi all
    echo Limpieza completada
    pause
    exit /b 0
)

echo ERROR: Opcion invalida
pause
exit /b 1

:end
echo.
echo ================================================
echo   Aplicacion ejecutandose en:
echo      http://localhost:5000
echo ================================================
echo.
echo Presiona Ctrl+C para detener
pause
