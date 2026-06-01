@echo off
setlocal

cd /d "%~dp0"

if "%~1"=="" (
    echo Usage: run_stress.bat COMx
    echo Example: run_stress.bat COM4
    exit /b 1
)

echo [1/2] Installing required packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [2/2] Starting stress serial bridge on %~1 ...
".venv\Scripts\python.exe" stress_serial_bridge.py --port %~1 --baud 9600

if errorlevel 1 (
    echo.
    echo Stress bridge exited with an error.
    pause
    exit /b 1
)

endlocal
