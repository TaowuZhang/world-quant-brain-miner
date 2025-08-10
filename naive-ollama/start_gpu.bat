@echo off
REM Naive-Ollma GPU Docker Startup Script for Windows

echo 🚀 Starting Naive-Ollma with GPU acceleration...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop and try again.
    pause
    exit /b 1
)

REM Check if NVIDIA Docker runtime is available
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ⚠️  NVIDIA Docker runtime not detected. GPU acceleration may not work.
    echo    Make sure you have NVIDIA Docker installed and configured.
    pause
)

REM Check if credential file exists
if not exist "credential.txt" (
    echo 📝 Setting up credentials...
    if exist "credential.example.txt" (
        copy credential.example.txt credential.txt >nul
        echo ✅ Created credential.txt from example
        echo ⚠️  Please edit credential.txt with your WorldQuant Brain credentials
        echo    Format: ["username", "password"]
        pause
    ) else (
        echo ❌ credential.example.txt not found. Please create credential.txt manually.
        pause
        exit /b 1
    )
)

REM Create necessary directories
echo 📁 Creating directories...
if not exist "results" mkdir results
if not exist "logs" mkdir logs

REM Check if credential file is properly formatted
python -c "import json; json.load(open('credential.txt'))" >nul 2>&1
if errorlevel 1 (
    echo ❌ credential.txt is not valid JSON. Please check the format.
    pause
    exit /b 1
)

echo 🔧 Building and starting GPU-optimized services...
echo    This may take several minutes on first run...

REM Stop any existing services
docker-compose down >nul 2>&1

REM Build and start GPU services
docker-compose -f docker-compose.gpu.yml up --build -d

echo ⏳ Waiting for services to start...
timeout /t 15 /nobreak >nul

REM Check if services are running
docker-compose -f docker-compose.gpu.yml ps | findstr "Up" >nul
if not errorlevel 1 (
    echo ✅ GPU services are running!
    echo.
    echo 📊 Monitoring:
    echo    - Application logs: docker-compose -f docker-compose.gpu.yml logs -f naive-ollma
    echo    - Web interface: http://localhost:3000
    echo    - Results: ./results/
    echo    - Logs: ./logs/
    echo.
    echo 🛑 To stop services: docker-compose -f docker-compose.gpu.yml down
    echo.
    
    REM Show logs
    echo 📋 Recent logs:
    docker-compose -f docker-compose.gpu.yml logs --tail=20 naive-ollma
    
    REM Show GPU status
    echo.
    echo 🔍 GPU Status:
    docker-compose -f docker-compose.gpu.yml exec naive-ollma nvidia-smi 2>nul || echo "GPU not detected in container"
) else (
    echo ❌ Services failed to start. Check logs:
    docker-compose -f docker-compose.gpu.yml logs
    pause
    exit /b 1
)

pause
