@echo off
setlocal
cd /d "%~dp0"

set "PHP_EXE=php"
if exist "C:\xampp\php\php.exe" set "PHP_EXE=C:\xampp\php\php.exe"
if exist "%~dp0xampp\php\php.exe" set "PHP_EXE=%~dp0xampp\php\php.exe"

%PHP_EXE% -v >nul 2>&1
if errorlevel 1 (
	echo Cannot find php.exe. Please install XAMPP or add PHP to PATH.
	pause
	exit /b 1
)

echo Starting PHP server at http://localhost:8080
start "" "http://localhost:8080"
"%PHP_EXE%" -S 127.0.0.1:8080
endlocal
