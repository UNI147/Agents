@echo off
echo ========================================
echo Starting environment
echo ========================================
echo.

if not exist "agentsenv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Run venv0.1.bat or venv0.3.bat first.
    echo.
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment...
call agentsenv\Scripts\activate.bat

echo [INFO] Starting simulation...
echo.
echo ========================================
echo.

python run.py
set EXIT_CODE=%ERRORLEVEL%

echo.
echo ========================================
if %EXIT_CODE% equ 0 (
    echo [DONE] Finished successfully
) else (
    echo [ERROR] Failed with code: %EXIT_CODE%
)
echo ========================================
echo.
pause
exit /b %EXIT_CODE%