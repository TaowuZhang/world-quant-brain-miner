@echo off
setlocal enabledelayedexpansion

REM WorldQuant Alpha Mining System with VRAM Monitoring
REM This script starts the alpha mining system with automatic VRAM management

echo Starting WorldQuant Alpha Mining System with VRAM Monitoring...

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

REM Start VRAM monitor in background
echo Starting VRAM monitor...
start /B python vram_monitor.py --threshold 0.85 --interval 30

REM Wait a moment for VRAM monitor to start
timeout /t 5 /nobreak >nul

REM Start the main alpha mining system
echo Starting alpha mining system...
docker-compose -f docker-compose.gpu.yml up -d

REM Wait for services to be ready
echo Waiting for services to be ready...
timeout /t 30 /nobreak >nul

REM Check if services are running
docker-compose -f docker-compose.gpu.yml ps | findstr "Up" >nul
if errorlevel 1 (
    echo Error: Services failed to start properly.
    goto cleanup
)

echo Alpha mining system started successfully!
echo Dashboard available at: http://localhost:5000
echo Ollama WebUI available at: http://localhost:3000
echo.
echo Monitoring system... (Press Ctrl+C to stop)
echo.

REM Monitor the system
:monitor_loop
timeout /t 60 /nobreak >nul

REM Check if main services are still running
docker-compose -f docker-compose.gpu.yml ps | findstr "Up" >nul
if errorlevel 1 (
    echo Error: Main services stopped unexpectedly
    goto cleanup
)

goto monitor_loop

:cleanup
echo.
echo Shutting down alpha mining system...
docker-compose -f docker-compose.gpu.yml down
taskkill /f /im python.exe /fi "WINDOWTITLE eq vram_monitor.py" >nul 2>&1
echo Cleanup complete.
pause
