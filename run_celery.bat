@echo off
echo =======================================================
echo Starting Celery Worker for Background Tasks...
echo Important: Ensure Redis is running on localhost:6379
echo =======================================================

:: MODULE 4: Clinical Safety Check - Is Redis Running?
echo Checking for Redis Broker...
powershell -Command "if (!(Test-NetConnection -Port 6379 -ComputerName localhost -InformationLevel Quiet)) { Write-Host 'ERROR: Redis is NOT running!' -ForegroundColor Red; exit 1 } else { Write-Host 'Redis Broker found!' -ForegroundColor Green }"
if %errorlevel% neq 0 (
    echo [ERROR] The Redis message broker is not active.
    echo Please run 'run_redis.bat' in a separate window first.
    echo Celery needs Redis to handle patient notifications.
    pause
    exit /b 1
)

if not exist "ai_env" (
    echo Error: Virtual environment 'ai_env' not found.
    echo Please run run_backend.bat first to set it up.
    pause
    exit /b 1
)

.\ai_env\Scripts\celery.exe -A celery_worker.app worker --pool=solo --loglevel=info

pause
