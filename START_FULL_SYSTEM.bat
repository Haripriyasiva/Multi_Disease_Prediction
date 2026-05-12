@echo off
title Medical AI Master Starter
echo ==================================================
echo   MEDICAL AI DIAGNOSTIC PLATFORM - SYSTEM RESET
echo ==================================================
echo.

echo Cleaning up previous sessions...
taskkill /f /im python.exe /t >nul 2>&1
taskkill /f /im node.exe /t >nul 2>&1
taskkill /f /im redis-server.exe /t >nul 2>&1
timeout /t 2 /nobreak > nul

echo [1/4] Launching REDIS SERVER...
start "Redis Server" cmd /c "run_redis.bat"

echo [2/4] Launching FLASK BACKEND...
start "Flask Backend" cmd /c "run_backend.bat"

echo [3/4] Launching CELERY WORKER...
start "Celery Worker" cmd /c "run_celery.bat"

echo [4/4] Launching REACT FRONTEND...
start "React Frontend" cmd /c "run_frontend.bat"

echo.
echo ==================================================
echo   ALL SYSTEMS TRIGGERED IMMEDIATELY! 
echo   Windows will open as they become ready.
echo ==================================================
timeout /t 5
exit
