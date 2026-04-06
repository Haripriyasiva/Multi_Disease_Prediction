@echo off
title Medical AI Master Starter
echo ==================================================
echo   MEDICAL AI DIAGNOSTIC PLATFORM - MASTER START
echo ==================================================
echo.

echo [1/3] Starting REDIS SERVER...
start "Redis Server" cmd /k "run_redis.bat"
timeout /t 3 /nobreak > nul

echo [2/3] Starting FLASK BACKEND...
start "Flask Backend" cmd /k "run_backend.bat"
timeout /t 2 /nobreak > nul

echo [3/3] Starting CELERY WORKER...
start "Celery Worker" cmd /k "run_celery.bat"

echo.
echo ==================================================
echo   ALL SYSTEMS BOOTED! 
echo   Please keep all three windows open.
echo ==================================================
pause
