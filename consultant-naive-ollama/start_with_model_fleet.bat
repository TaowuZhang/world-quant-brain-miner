@echo off
setlocal enabledelayedexpansion

REM WorldQuant Alpha Mining System with Model Fleet Management
REM This script starts the alpha mining system with automatic model downgrading on VRAM issues

echo Starting WorldQuant Alpha Mining System with Model Fleet Management...
echo ================================================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running. Please start Docker first.
    pause
    exit /b 1
)

REM Check if nvidia-smi is available
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo Warning: nvidia-smi not found. VRAM monitoring will be limited.
)

REM Create necessary directories
if not exist results mkdir results
if not exist logs mkdir logs

REM Start the model fleet manager in background
echo Starting Model Fleet Manager...
start /B python model_fleet_manager.py --monitor

REM Wait a moment for the fleet manager to initialize
timeout /t 5 /nobreak >nul

REM Check fleet status
echo Checking Model Fleet Status...
python model_fleet_manager.py --status

REM Start the main Docker services
echo Starting Docker services...
docker-compose -f docker-compose.gpu.yml up -d

REM Wait for services to be ready
echo Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Check service status
echo Service Status:
docker-compose -f docker-compose.gpu.yml ps

REM Show current model being used
echo.
echo Current Model Configuration:
python model_fleet_manager.py --status

echo.
echo Alpha Mining System is running with Model Fleet Management!
echo ================================================================
echo Web Dashboard: http://localhost:5000
echo Ollama WebUI: http://localhost:3000
echo Ollama API: http://localhost:11434
echo.
echo Model Fleet Manager is monitoring for VRAM issues and will automatically
echo downgrade to smaller models if needed.
echo.
echo Press Ctrl+C to stop all services

REM Keep the script running
pause
