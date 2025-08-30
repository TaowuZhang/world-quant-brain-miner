@echo off
echo ========================================
echo   Alpha Generator Dashboard
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "web_dashboard.py" (
    echo ERROR: web_dashboard.py not found!
    pause
    exit /b 1
)

if not exist "credential.txt" (
    echo ERROR: credential.txt not found!
    echo Please copy credential.example.txt to credential.txt and update with your credentials.
    pause
    exit /b 1
)

REM Create directories if they don't exist
if not exist "results" mkdir results
if not exist "logs" mkdir logs
if not exist "templates" mkdir templates

echo.
echo Starting Alpha Generator Dashboard...
echo.
echo Services:
echo - Dashboard: http://localhost:5000
echo - Ollama WebUI: http://localhost:3000
echo - Ollama API: http://localhost:11434
echo.
echo Press Ctrl+C to stop the dashboard
echo.

REM Start the dashboard
python web_dashboard.py

pause
