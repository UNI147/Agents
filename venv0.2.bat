@echo off
chcp 65001 >nul
:: Check if environment exists
if not exist "agentsenv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run venv0.1.bat or venv0.3.bat first.
    pause
    exit /b 1
)
echo ========================================
echo Starting environment
echo ========================================
:: Activate environment
call agentsenv\Scripts\activate
echo [OK] Virtual environment activated
echo [OK] Python version:
python --version
echo.
pip install networkx
:: Export current dependencies to requirements.txt
echo [INFO] Updating requirements.txt...
pip freeze > requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to update requirements.txt
    pause
    exit /b 1
)
echo [OK] requirements.txt updated (all installed packages)
echo.
pause
exit /b %ERRORLEVEL%