@echo off
chcp 65001 >nul

set PYTHON_PATH=C:\Users\Hatul\AppData\Local\Programs\Python\Python312\python.exe

echo ========================================
echo Пересоздание виртуального окружения
echo ========================================
echo.

:: Проверка существования Python
if not exist "%PYTHON_PATH%" (
    echo [ОШИБКА] Python не найден по пути: %PYTHON_PATH%
    echo Проверьте правильность пути к Python
    pause
    exit /b 1
)

echo [OK] Python найден: %PYTHON_PATH%
"%PYTHON_PATH%" --version
echo.

:: Удаление старого окружения, если существует
if exist "agentsenv" (
    echo [INFO] Удаление старого окружения...
    rmdir /s /q "agentsenv"
    if exist "agentsenv" (
        echo [ОШИБКА] Не удалось полностью удалить старое окружение
        echo Возможно, оно используется другим процессом
        pause
        exit /b 1
    )
    echo [OK] Старое окружение удалено
    echo.
)

:: Создание нового окружения
echo [INFO] Создание нового виртуального окружения...
"%PYTHON_PATH%" -m venv agentsenv

if not exist "agentsenv\Scripts\activate.bat" (
    echo [ОШИБКА] Не удалось создать виртуальное окружение
    pause
    exit /b 1
)

echo [OK] Виртуальное окружение создано
echo.

:: Активация
echo [INFO] Активация окружения...
call agentsenv\Scripts\activate.bat

echo.
echo ========================================
echo [ГОТОВО] Виртуальное окружение пересоздано
echo ========================================
echo.

pause
exit /b 0