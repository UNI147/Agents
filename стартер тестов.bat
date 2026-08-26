@echo off
chcp 65001 >nul
echo ========================================
echo Starting environment
echo ========================================
echo.
:: Check if environment exists
if not exist "agentsenv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run venv0.1.bat or venv0.3.bat first.
    echo.
    pause
    exit /b 1
)
echo [INFO] Activating virtual environment...
call agentsenv\Scripts\activate.bat
echo [INFO] Running tests...
echo.
echo ========================================
echo.
python -m pytest tests -v
set EXIT_CODE=%ERRORLEVEL%
echo.
echo ========================================
if %EXIT_CODE% equ 0 (
    echo [DONE] Tests finished successfully
) else (
    echo [ERROR] Tests failed with code: %EXIT_CODE%
)
echo ========================================
echo.
pause
exit /b %EXIT_CODE%