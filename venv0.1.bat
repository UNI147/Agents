@echo off
chcp 65001 >nul
set PYTHON_PATH=C:\Users\Hatul\AppData\Local\Programs\Python\Python312\python.exe
echo ========================================
echo Creating virtual environment
echo ========================================
echo.
:: Check if Python exists
if not exist "%PYTHON_PATH%" (
    echo [ERROR] Python not found at path: %PYTHON_PATH%
    echo Please check the Python path.
    pause
    exit /b 1
)
echo [OK] Python found: %PYTHON_PATH%
"%PYTHON_PATH%" --version
echo.
:: Create new environment
echo [INFO] Creating new virtual environment...
"%PYTHON_PATH%" -m venv agentsenv
if not exist "agentsenv\Scripts\activate.bat" (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment created
echo.
:: Activation
echo [INFO] Activating environment...
call agentsenv\Scripts\activate.bat
echo.
echo ========================================
echo [DONE] Virtual environment is ready
echo ========================================
echo.
pause
exit /b 0