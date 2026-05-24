@echo off
setlocal

if "%AGENT_BEARER_TOKEN%"=="" (
  echo [ERROR] AGENT_BEARER_TOKEN is required.
  exit /b 2
)

if "%AGENT_BASE_URL%"=="" (
  set AGENT_BASE_URL=http://127.0.0.1:8000
)

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scheduler\run_daily_jobs.py
) else (
  python scheduler\run_daily_jobs.py
)

endlocal
