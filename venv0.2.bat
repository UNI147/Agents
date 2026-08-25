@echo off
chcp 65001 >nul

:: Проверка существования окружения
if not exist "agentsenv\Scripts\activate.bat" (
    echo [ОШИБКА] Виртуальное окружение не найдено!
    echo Сначала запустите venv0.1.bat или venv0.3.bat
    pause
    exit /b 1
)

echo ========================================
echo Запуск окружения
echo ========================================

:: Активация окружения
call agentsenv\Scripts\activate

echo [OK] Виртуальное окружение активировано
echo [OK] Python: 
python --version
echo.

pip install networkx

:: Экспорт актуального списка всех зависимостей в requirements.txt
echo [INFO] Обновление requirements.txt...
pip freeze > requirements.txt
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось обновить requirements.txt
    pause
    exit /b 1
)

echo [OK] requirements.txt обновлён (все установленные пакеты)
echo.

pause
exit /b %EXIT_CODE%