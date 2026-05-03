@echo off
setlocal

cd /d "%~dp0"

echo [1/2] Installing required packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [2/2] Starting emotion recognition...
echo Press Q in the camera window to quit.
".venv\Scripts\python.exe" emotion_camera.py

if errorlevel 1 (
    echo.
    echo Emotion system exited with an error.
    pause
    exit /b 1
)

endlocal
