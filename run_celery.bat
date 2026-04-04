@echo off
echo =======================================================
echo Starting Celery Worker for Background Tasks...
echo Important: Ensure Redis is running on localhost:6379
echo =======================================================

if not exist "ai_env" (
    echo Error: Virtual environment 'ai_env' not found.
    echo Please run run_backend.bat first to set it up.
    pause
    exit /b 1
)

.\ai_env\Scripts\celery.exe -A celery_worker.app worker --pool=solo --loglevel=info

pause
