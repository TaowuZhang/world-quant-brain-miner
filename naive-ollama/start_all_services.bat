@echo off
echo Starting WorldQuant Alpha Mining Services with Docker Compose
echo ==============================================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running. Please start Docker first.
    pause
    exit /b 1
)

REM Check if docker-compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo Error: docker-compose is not installed. Please install it first.
    pause
    exit /b 1
)

REM Check if credential.txt exists
if not exist "credential.txt" (
    echo Error: credential.txt not found. Please create it with your WorldQuant credentials.
    pause
    exit /b 1
)

echo Building and starting services...
echo This will start:
echo   - Ollama service (AI model serving)
echo   - Machine Miner (traditional alpha mining)
echo   - Alpha Generator (AI-powered alpha generation)
echo   - Alpha Expression Miner (continuous expression mining)
echo   - Web UI (monitoring interface at http://localhost:3000)
echo.

REM Build and start all services
docker-compose up --build -d

echo.
echo Services started successfully!
echo.
echo Service Status:
docker-compose ps
echo.
echo Logs can be viewed with:
echo   docker-compose logs -f [service-name]
echo.
echo Available services:
echo   - ollama
echo   - machine-miner
echo   - alpha-generator
echo   - alpha-expression-miner
echo   - ollama-webui
echo.
echo Web UI available at: http://localhost:3000
echo Ollama API available at: http://localhost:11434
echo.
echo To stop all services:
echo   docker-compose down
echo.
echo To view logs for a specific service:
echo   docker-compose logs -f machine-miner
echo   docker-compose logs -f alpha-generator
echo   docker-compose logs -f alpha-expression-miner
echo.
pause
