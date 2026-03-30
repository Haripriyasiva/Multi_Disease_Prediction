@echo off
echo Executing Failsafe Clean Sandbox Rebuild...

if exist "ai_env" (
    echo Wiping damaged ai_env...
    rmdir /S /Q "ai_env"
)
echo Building fresh sterile virtual environment: ai_env
python -m venv ai_env

echo Upgrading Pip Installer Engine...
.\ai_env\Scripts\python.exe -m pip install --upgrade pip

echo Provisioning API libraries...
.\ai_env\Scripts\python.exe -m pip install "greenlet>=2.0.0" --only-binary :all:
.\ai_env\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Network or package error occurred during installation.
    pause
    exit /b %errorlevel%
)

echo Sandbox successfully sealed. Booting API...
.\ai_env\Scripts\python.exe server.py
pause
