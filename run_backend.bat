@echo off
echo Checking virtual environment...

if not exist "ai_env" (
    echo Building fresh virtual environment: ai_env
    python -m venv ai_env
    
    echo Upgrading Pip...
    .\ai_env\Scripts\python.exe -m pip install --upgrade pip
    
    echo Connecting to PyPI and installing libraries...
    .\ai_env\Scripts\python.exe -m pip install "greenlet>=2.0.0" --only-binary :all:
    .\ai_env\Scripts\python.exe -m pip install -r requirements.txt
    
    if %errorlevel% neq 0 (
        echo [ERROR] Network or package error occurred during installation.
        pause
        exit /b %errorlevel%
    )
) else (
    echo Virtual environment already exists. Skipping library installation...
)

echo Booting API...
.\ai_env\Scripts\python.exe server.py
pause
