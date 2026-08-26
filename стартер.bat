@echo off

echo ========================================
echo Запуск в окружении
echo ========================================
echo.

:: Проверка существования окружения
if not exist "agentsenv\Scripts\activate.bat" (
    echo [ОШИБКА] Виртуальное окружение не найдено!
    echo Сначала запустите venv0.1.bat или venv0.3.bat.
    echo.
    pause
    exit /b 1
)

echo [INFO] Активация виртуального окружения...
call agentsenv\Scripts\activate.bat

echo [INFO] Запуск симуляции...
echo.
echo ========================================
echo.

python run.py
set EXIT_CODE=%ERRORLEVEL%

echo.
echo ========================================
if %EXIT_CODE% equ 0 (
    echo [ГОТОВО] Работа завершилось успешно
) else (
    echo [ОШИБКА] Работа провалилось (код: %EXIT_CODE%)
)
echo ========================================
echo.

pause
exit /b %EXIT_CODE%