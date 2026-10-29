@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 goto :error
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Setup complete. Add GOOGLE_API_KEY to the project .env file, then run run.bat.
exit /b 0

:error
echo.
echo Setup failed. Check the error above.
exit /b 1
