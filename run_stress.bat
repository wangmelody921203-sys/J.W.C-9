@echo off
setlocal

cd /d "%~dp0"

if "%~1"=="" (
    echo Usage: run_stress.bat COMx
    echo Example: run_stress.bat COM4
    echo Example (cloud relay): run_stress.bat COM4 --push-url https://your-render-service.onrender.com/stress/report --push-token your_token
    echo Example (per-user): run_stress.bat COM4 --push-url https://your-render-service.onrender.com/stress/report --push-token your_token --user-id 00000000-0000-0000-0000-000000000000
    exit /b 1
)

set "PORT=%~1"
shift

echo [1/2] Installing required packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [2/2] Starting stress serial bridge on %PORT% ...
".venv\Scripts\python.exe" stress_serial_bridge.py --port %PORT% --baud 9600 %*

if errorlevel 1 (
    echo.
    echo Stress bridge exited with an error.
    pause
    exit /b 1
)

endlocal
